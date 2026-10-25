"""Safe portable serialization for NeuroLab's NumPy feedforward models.

The ``.nlmodel`` container is a ZIP archive holding JSON metadata and a
compressed NumPy archive. Import never deserializes Python objects or enables
NumPy pickle loading.
"""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import PurePosixPath
from typing import Any, Mapping
import zipfile

import numpy as np

from .models import NeuralNetwork
from .models.neural_network import HIDDEN_ACTIVATIONS


FORMAT_NAME = "neurolab-model"
FORMAT_VERSION = 1
MAX_MODEL_FILE_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
_OUTER_FILES = {"metadata.json", "parameters.npz"}


def export_model(network: NeuralNetwork, metadata: Mapping[str, Any] | None = None) -> bytes:
    """Return a validated model as a portable ``.nlmodel`` byte string."""
    _validate_network(network)
    user_metadata = dict(metadata or {})
    _validate_metadata(user_metadata, network.input_feature_count)
    configuration = {
        "hidden_activation": network.hidden_activation,
        "output_activation": "sigmoid",
    }
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": {
            "input_feature_count": int(network.input_feature_count),
            "hidden_layer_sizes": [int(width) for width in network.hidden_layer_sizes],
            "output_size": int(network.output_size),
        },
        "configuration": configuration,
        "metadata": user_metadata,
    }
    try:
        manifest_bytes = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Model metadata must contain JSON-safe finite values: {error}") from error

    parameter_buffer = BytesIO()
    arrays: dict[str, np.ndarray] = {}
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        arrays[f"weight_{index}"] = np.asarray(weight, dtype=np.float64)
        arrays[f"bias_{index}"] = np.asarray(bias, dtype=np.float64)
    np.savez_compressed(parameter_buffer, **arrays)

    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", manifest_bytes)
        archive.writestr("parameters.npz", parameter_buffer.getvalue())
    result = archive_buffer.getvalue()
    if len(result) > MAX_MODEL_FILE_BYTES:
        raise ValueError("Exported model exceeds the 50 MB file limit.")
    return result


def import_model(model_file: bytes | bytearray | BytesIO) -> tuple[NeuralNetwork, dict[str, Any]]:
    """Validate and reconstruct a network from an untrusted ``.nlmodel`` file.

    The returned metadata includes the format manifest fields and user metadata.
    Imported arrays are loaded with ``allow_pickle=False`` and copied into the
    existing :class:`NeuralNetwork` implementation.
    """
    if isinstance(model_file, BytesIO):
        payload = model_file.getvalue()
    elif isinstance(model_file, (bytes, bytearray)):
        payload = bytes(model_file)
    else:
        raise ValueError("Model input must be file bytes from a NeuroLab .nlmodel export.")
    if not payload or len(payload) > MAX_MODEL_FILE_BYTES:
        raise ValueError("Model file is empty or exceeds the 50 MB size limit.")

    try:
        outer = zipfile.ZipFile(BytesIO(payload), "r")
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError("File is not a valid NeuroLab model archive.") from error
    with outer:
        entries = outer.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Model archive contains duplicate entries.")
        if set(names) != _OUTER_FILES:
            raise ValueError("Model archive must contain exactly metadata.json and parameters.npz.")
        for entry in entries:
            if PurePosixPath(entry.filename).is_absolute() or ".." in PurePosixPath(entry.filename).parts:
                raise ValueError("Model archive contains an unsafe path.")
            if entry.flag_bits & 0x1:
                raise ValueError("Encrypted model archive entries are not supported.")
            if entry.file_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Model archive contains an entry that exceeds the size limit.")
        if sum(entry.file_size for entry in entries) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Model archive exceeds the total uncompressed size limit.")
        metadata_bytes = outer.read("metadata.json")
        parameters_bytes = outer.read("parameters.npz")

    try:
        manifest = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Model metadata is not valid UTF-8 JSON.") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "format", "format_version", "created_at", "architecture", "configuration", "metadata"
    }:
        raise ValueError("Model metadata is missing required fields or contains unsupported fields.")
    if manifest["format"] != FORMAT_NAME:
        raise ValueError("This file is not a NeuroLab model export.")
    version = manifest["format_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != FORMAT_VERSION:
        raise ValueError(f"Unsupported model format version: {version!r}.")
    if not isinstance(manifest["created_at"], str):
        raise ValueError("Model creation timestamp must be a string.")
    try:
        datetime.fromisoformat(manifest["created_at"])
    except ValueError as error:
        raise ValueError("Model creation timestamp is invalid.") from error

    architecture = manifest["architecture"]
    if not isinstance(architecture, dict) or set(architecture) != {
        "input_feature_count", "hidden_layer_sizes", "output_size"
    }:
        raise ValueError("Model architecture metadata is missing required fields or contains unsupported fields.")
    input_count = _positive_integer(architecture["input_feature_count"], "input_feature_count")
    output_size = _positive_integer(architecture["output_size"], "output_size")
    hidden = architecture["hidden_layer_sizes"]
    if not isinstance(hidden, list):
        raise ValueError("hidden_layer_sizes must be a JSON list.")
    hidden_sizes = tuple(_positive_integer(width, f"hidden_layer_sizes[{index}]") for index, width in enumerate(hidden))

    configuration = manifest["configuration"]
    if not isinstance(configuration, dict) or set(configuration) != {"hidden_activation", "output_activation"}:
        raise ValueError("Model configuration is missing required fields or contains unsupported fields.")
    activation = configuration["hidden_activation"]
    if activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(f"Unsupported hidden activation: {activation!r}.")
    if configuration["output_activation"] != "sigmoid":
        raise ValueError("This NeuroLab network implementation supports only sigmoid output activation.")
    user_metadata = manifest["metadata"]
    if not isinstance(user_metadata, dict):
        raise ValueError("User metadata must be a JSON object.")

    layer_sizes = (input_count, *hidden_sizes, output_size)
    expected_arrays = {f"{kind}_{index}" for index in range(len(layer_sizes) - 1) for kind in ("weight", "bias")}
    try:
        parameter_archive = zipfile.ZipFile(BytesIO(parameters_bytes), "r")
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError("Parameter data is not a valid NumPy archive.") from error
    with parameter_archive:
        param_entries = parameter_archive.infolist()
        param_names = [entry.filename for entry in param_entries]
        if len(param_names) != len(set(param_names)):
            raise ValueError("Parameter archive contains duplicate arrays.")
        if sum(entry.file_size for entry in param_entries) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Parameter arrays exceed the total size limit.")
        # npz stores each array as <name>.npy; reject unexpected members before loading.
        expected_files = {name + ".npy" for name in expected_arrays}
        if set(param_names) != expected_files:
            raise ValueError("Parameter archive has missing or unexpected weight/bias arrays.")

    try:
        loaded = np.load(BytesIO(parameters_bytes), allow_pickle=False)
        with loaded:
            weights: list[np.ndarray] = []
            biases: list[np.ndarray] = []
            for index, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                weight = np.asarray(loaded[f"weight_{index}"])
                bias = np.asarray(loaded[f"bias_{index}"])
                if weight.shape != (fan_out, fan_in):
                    raise ValueError(f"weight_{index} has shape {weight.shape}; expected {(fan_out, fan_in)}.")
                if bias.shape != (fan_out, 1):
                    raise ValueError(f"bias_{index} has shape {bias.shape}; expected {(fan_out, 1)}.")
                if weight.dtype.kind not in "iuf" or bias.dtype.kind not in "iuf":
                    raise ValueError("Model parameters must be real numeric arrays.")
                if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                    raise ValueError("Model parameters must contain only finite values.")
                weights.append(np.asarray(weight, dtype=float).copy())
                biases.append(np.asarray(bias, dtype=float).copy())
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not safely read model parameter arrays: {error}") from error

    _validate_metadata(user_metadata, input_count)
    network = NeuralNetwork(
        input_feature_count=input_count, hidden_layer_sizes=hidden_sizes,
        output_size=output_size, random_seed=0, hidden_activation=activation,
    )
    network.weights = weights
    network.biases = biases
    network.activations = []
    network.weighted_sums = []
    network._gradients_available = False
    return network, manifest


def _validate_network(network: NeuralNetwork) -> None:
    if not isinstance(network, NeuralNetwork):
        raise ValueError("Only NeuroLab NeuralNetwork instances can be exported.")
    if network.hidden_activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(f"Unsupported hidden activation: {network.hidden_activation!r}.")
    expected_sizes = (network.input_feature_count, *network.hidden_layer_sizes, network.output_size)
    if len(network.weights) != len(expected_sizes) - 1 or len(network.biases) != len(expected_sizes) - 1:
        raise ValueError("Model parameter counts do not match its architecture.")
    for index, (fan_in, fan_out) in enumerate(zip(expected_sizes[:-1], expected_sizes[1:])):
        weight = np.asarray(network.weights[index])
        bias = np.asarray(network.biases[index])
        if weight.shape != (fan_out, fan_in) or bias.shape != (fan_out, 1):
            raise ValueError(f"Layer {index} parameter shapes do not match the model architecture.")
        if weight.dtype.kind not in "iuf" or bias.dtype.kind not in "iuf":
            raise ValueError("Model parameters must be real numeric arrays.")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError("Model parameters must contain only finite values.")


def _validate_metadata(metadata: Mapping[str, Any], input_count: int) -> None:
    if "trained" in metadata and not isinstance(metadata["trained"], bool):
        raise ValueError("Metadata field 'trained' must be a boolean.")
    preprocessing = metadata.get("preprocessing")
    if preprocessing is None:
        return
    if not isinstance(preprocessing, dict) or preprocessing.get("type") != "standard_scaler":
        raise ValueError("preprocessing metadata must describe a supported standard_scaler or be omitted.")
    if set(preprocessing) != {"type", "mean", "scale"}:
        raise ValueError("standard_scaler preprocessing needs exactly type, mean, and scale fields.")
    mean = np.asarray(preprocessing["mean"], dtype=float)
    scale = np.asarray(preprocessing["scale"], dtype=float)
    if mean.shape != (input_count,) or scale.shape != (input_count,):
        raise ValueError("StandardScaler mean and scale must match the model input dimensions.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("StandardScaler parameters must be finite and scales must be positive.")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)
