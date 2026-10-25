"""Safe model archive, inference, and Model Manager verification."""

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neurolab.model_manager import available_models, parse_csv_features, predict_with_model
from src.neurolab.model_persistence import export_model, import_model
from src.neurolab.models import NeuralNetwork


def test_model_export_import_preserves_parameters_configuration_and_predictions():
    network = NeuralNetwork(2, (5, 3), 1, random_seed=41, hidden_activation="tanh")
    preprocessing = {"type": "standard_scaler", "mean": [1.0, -2.0], "scale": [2.0, 0.5]}
    metadata = {"trained": True, "model_label": "Round trip", "preprocessing": preprocessing,
                "configuration": {"optimizer": "Adam", "epochs_completed": 8}}
    samples = np.array([[0.2, 0.4], [1.0, -3.0], [2.2, 1.1]])
    original_output, original_classes = predict_with_model(network, samples, metadata)

    restored, manifest = import_model(export_model(network, metadata))
    restored_output, restored_classes = predict_with_model(restored, samples, manifest["metadata"])
    assert restored.input_feature_count == network.input_feature_count
    assert restored.hidden_layer_sizes == network.hidden_layer_sizes
    assert restored.output_size == network.output_size
    assert restored.hidden_activation == network.hidden_activation
    assert manifest["configuration"]["output_activation"] == "sigmoid"
    assert manifest["metadata"]["trained"] is True
    assert manifest["metadata"]["configuration"]["epochs_completed"] == 8
    for expected, actual in zip(network.weights + network.biases, restored.weights + restored.biases):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(original_output, restored_output, atol=1e-14)
    np.testing.assert_array_equal(original_classes, restored_classes)


def test_export_rejects_nonfinite_or_incompatible_parameters():
    network = NeuralNetwork(2, (3,), 1, random_seed=4)
    network.weights[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        export_model(network)
    network.weights[0][0, 0] = 0.0
    network.biases[0] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="shapes"):
        export_model(network)


def test_import_rejects_bad_archives_versions_shapes_and_nonfinite_arrays():
    network = NeuralNetwork(2, (3,), 1, random_seed=5)
    blob = export_model(network, {"trained": True})
    manifest, arrays = _read_archive(blob)

    with pytest.raises(ValueError, match="valid NeuroLab model archive"):
        import_model(b"not a model")

    changed = dict(manifest)
    changed["format_version"] = 77
    with pytest.raises(ValueError, match="Unsupported model format version"):
        import_model(_build_archive(changed, arrays))

    wrong_shape = {key: value.copy() for key, value in arrays.items()}
    wrong_shape["weight_0"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="weight_0 has shape"):
        import_model(_build_archive(manifest, wrong_shape))

    nonfinite = {key: value.copy() for key, value in arrays.items()}
    nonfinite["bias_0"][0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        import_model(_build_archive(manifest, nonfinite))

    extra = {**arrays, "rogue": np.zeros((1,))}
    with pytest.raises(ValueError, match="missing or unexpected"):
        import_model(_build_archive(manifest, extra))
    missing = dict(arrays)
    del missing["bias_0"]
    with pytest.raises(ValueError, match="missing or unexpected"):
        import_model(_build_archive(manifest, missing))


def test_prediction_dimensions_and_csv_input_are_validated():
    network = NeuralNetwork(2, (), 1, random_seed=2)
    output, classes = predict_with_model(network, np.array([0.1, 0.2]))
    assert output.shape == (1, 1)
    assert classes.shape == (1,)
    with pytest.raises(ValueError, match="expects 2 features"):
        predict_with_model(network, np.array([[1.0, 2.0, 3.0]]))
    parsed = parse_csv_features(b"0.1,0.2\n0.3,0.4\n", expected_dimensions=2)
    np.testing.assert_allclose(parsed, [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="exactly 2 feature columns"):
        parse_csv_features(b"0.1,0.2,0.3\n", expected_dimensions=2)
    with pytest.raises(ValueError, match="without a header"):
        parse_csv_features(b"feature1,feature2\n0.1,0.2\n", expected_dimensions=2)


def test_model_manager_page_is_available_from_navigation():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_model_manager").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Model Manager"]


def test_session_models_are_listed_only_when_the_run_completed_training():
    from types import SimpleNamespace

    trained = SimpleNamespace(network=NeuralNetwork(2, (4,), 1, random_seed=9), epochs_completed=3)
    incomplete = SimpleNamespace(network=NeuralNetwork(2, (), 1, random_seed=8), epochs_completed=0)
    session = {
        "network_playground_result": {"result": trained, "run_number": 2, "dataset_name": "XOR",
                                       "config": {"epochs_completed": 3}},
        "experiment_wizard_result": {"result": incomplete},
    }
    models = available_models(session)
    assert list(models) == ["network-playground-2"]
    assert models["network-playground-2"]["trained"] is True


def _read_archive(blob):
    with zipfile.ZipFile(BytesIO(blob)) as outer:
        manifest = json.loads(outer.read("metadata.json"))
        with np.load(BytesIO(outer.read("parameters.npz")), allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    return manifest, arrays


def _build_archive(manifest, arrays):
    parameter_buffer = BytesIO()
    np.savez_compressed(parameter_buffer, **arrays)
    result = BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(manifest).encode("utf-8"))
        archive.writestr("parameters.npz", parameter_buffer.getvalue())
    return result.getvalue()
