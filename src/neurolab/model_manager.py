"""Session-aware Model Manager for NeuroLab's exported and trained networks."""

from hashlib import sha256
from io import StringIO
import json
import re
from typing import Any, Mapping

import numpy as np
import streamlit as st

from .model_persistence import export_model, import_model
from .models import NeuralNetwork


IMPORTED_MODELS_KEY = "model_manager_imported_models"
PREDICTION_KEY = "model_manager_prediction"


def available_models(session_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect real trained runs and explicitly loaded imports from page state."""
    models: dict[str, dict[str, Any]] = {}
    playground = session_state.get("network_playground_result")
    if isinstance(playground, dict) and _is_training_result(playground.get("result")):
        config = _json_safe(playground.get("config", {}))
        scaler = playground.get("scaler")
        _add_model(models, f"network-playground-{playground.get('run_number', 1)}", 
                   f"Network Playground · Run {playground.get('run_number', 1)} · {playground.get('dataset_name', 'dataset')}",
                   playground["result"].network, "Network Playground", True, config, scaler)

    wizard = session_state.get("experiment_wizard_result")
    if isinstance(wizard, dict) and _is_training_result(wizard.get("result")):
        wizard_config = session_state.get("experiment_wizard", {})
        details = wizard_config.get("details", {}) if isinstance(wizard_config, dict) else {}
        _add_model(models, "experiment-wizard-run", f"Experiment Wizard · {details.get('name', 'completed run')}",
                   wizard["result"].network, "Experiment Wizard", True,
                   _json_safe(wizard_config), wizard.get("split").scaler if wizard.get("split") else None)

    optimizer_lab = session_state.get("optimizer_lab_result")
    comparison = optimizer_lab.get("comparison") if isinstance(optimizer_lab, dict) else None
    if comparison is not None and isinstance(getattr(comparison, "runs", None), dict):
        for name, run in comparison.runs.items():
            if _is_training_result(run):
                run_number = optimizer_lab.get("run_number", 1)
                model_id = f"optimizer-{run_number}-{name.lower()}"
                label = f"Optimizer Lab · Run {run_number} · {name}"
                _add_model(models, model_id, label, run.network, "Optimizer Lab", True,
                           _json_safe(optimizer_lab.get("configuration", {})), comparison.split.scaler)

    for imported in session_state.get(IMPORTED_MODELS_KEY, []):
        model_id = imported.get("id")
        if model_id:
            user_metadata = imported.get("manifest", {}).get("metadata", {})
            is_trained = bool(user_metadata.get("trained", False))
            _add_model(models, f"imported-{model_id}", imported.get("label", "Imported model"),
                       imported["network"], "Imported model", is_trained,
                       user_metadata.get("configuration", {}), _preprocessing_from_metadata(user_metadata))
    return models


def predict_with_model(network: NeuralNetwork, features: np.ndarray,
                       metadata: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate outputs and labels without training; apply saved train scaler if present."""
    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Prediction input must be a non-empty 2D sample matrix.")
    if values.shape[1] != network.input_feature_count:
        raise ValueError(f"This model expects {network.input_feature_count} features per sample; received {values.shape[1]}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Prediction inputs must contain only finite numeric values.")
    config = dict(metadata or {})
    _validate_prediction_preprocessing(config, network.input_feature_count)
    preprocessing = config.get("preprocessing")
    model_values = values
    if preprocessing is not None:
        mean = np.asarray(preprocessing["mean"], dtype=float)
        scale = np.asarray(preprocessing["scale"], dtype=float)
        model_values = (values - mean) / scale
    outputs = network.forward(model_values)
    if network.output_size == 1:
        classes = (outputs[:, 0] >= 0.5).astype(int)
    else:
        classes = np.argmax(outputs, axis=1)
    return outputs, classes


def parse_csv_features(file_bytes: bytes, expected_dimensions: int, max_rows: int = 1000) -> np.ndarray:
    """Parse a small headerless numeric CSV containing feature columns only."""
    if not isinstance(file_bytes, bytes) or not file_bytes:
        raise ValueError("Upload a non-empty CSV file of numeric feature values.")
    if len(file_bytes) > 1024 * 1024:
        raise ValueError("CSV input must be no larger than 1 MB.")
    try:
        text = file_bytes.decode("utf-8")
        values = np.loadtxt(StringIO(text), delimiter=",", dtype=float, ndmin=2)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("CSV must contain only numeric values, without a header row.") from error
    if values.shape[0] > max_rows:
        raise ValueError(f"CSV may contain at most {max_rows} samples.")
    if values.ndim != 2 or values.shape[1] != expected_dimensions:
        raise ValueError(f"CSV must contain exactly {expected_dimensions} feature columns.")
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("CSV must contain at least one sample and all values must be finite.")
    return values


def _is_training_result(value: Any) -> bool:
    return value is not None and getattr(value, "network", None) is not None and getattr(value, "epochs_completed", 0) > 0


def _add_model(models: dict[str, dict[str, Any]], model_id: str, label: str,
               network: NeuralNetwork, source: str, trained: bool,
               configuration: Any, scaler: Any) -> None:
    metadata = {"trained": bool(trained), "source": source, "configuration": _json_safe(configuration)}
    if isinstance(scaler, Mapping):
        metadata["preprocessing"] = _json_safe(scaler)
    elif scaler is not None:
        metadata["preprocessing"] = {
            "type": "standard_scaler",
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        }
    models[model_id] = {"id": model_id, "label": label, "network": network, "source": source,
                        "trained": bool(trained), "metadata": metadata}


def _preprocessing_from_metadata(metadata: Mapping[str, Any]) -> Any:
    return metadata.get("preprocessing") if isinstance(metadata, Mapping) else None


def _validate_prediction_preprocessing(metadata: Mapping[str, Any], dimensions: int) -> None:
    preprocessing = metadata.get("preprocessing")
    if preprocessing is None:
        return
    if not isinstance(preprocessing, Mapping) or preprocessing.get("type") != "standard_scaler":
        raise ValueError("Imported model has unsupported preprocessing metadata.")
    mean, scale = np.asarray(preprocessing.get("mean"), dtype=float), np.asarray(preprocessing.get("scale"), dtype=float)
    if mean.shape != (dimensions,) or scale.shape != (dimensions,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Imported model preprocessing dimensions or values are invalid.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _architecture_label(network: NeuralNetwork) -> str:
    layer_widths = [network.input_feature_count, *network.hidden_layer_sizes, network.output_size]
    return " → ".join(map(str, layer_widths))


def _parameter_count(network: NeuralNetwork) -> int:
    return int(sum(weight.size + bias.size for weight, bias in zip(network.weights, network.biases)))


def _render_model_details(model: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> None:
    network: NeuralNetwork = model["network"]
    cols = st.columns(4)
    cols[0].metric("Input features", str(network.input_feature_count))
    cols[1].metric("Output units", str(network.output_size))
    cols[2].metric("Trainable parameters", f"{_parameter_count(network):,}")
    cols[3].metric("Model status", "Trained" if model.get("trained") else "Untrained")
    st.write({
        "Architecture": _architecture_label(network),
        "Hidden activation": network.hidden_activation,
        "Output activation": "sigmoid",
        "Source": model.get("source", "Imported model"),
    })
    metadata = manifest or {"metadata": model.get("metadata", {})}
    st.json(_json_safe(metadata))


def render_model_manager() -> None:
    """Render model export, explicit import, loaded model details, and inference."""
    st.markdown('<div class="eyebrow">EXPORT · IMPORT · PREDICT</div>', unsafe_allow_html=True)
    st.title("Model Manager")
    st.markdown("Export trained NumPy networks, safely restore a NeuroLab model file, and inspect predictions without training.")

    st.markdown("### Export a trained model")
    models = available_models(st.session_state)
    exportable = {model_id: model for model_id, model in models.items() if model["trained"]}
    if exportable:
        selected_id = st.selectbox("Available trained model", tuple(exportable), format_func=lambda model_id: exportable[model_id]["label"], key="model_export_selection")
        selected = exportable[selected_id]
        _render_model_details(selected)
        try:
            file_bytes = export_model(selected["network"], selected["metadata"])
            filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", selected["label"]).strip("_").lower() or "neurolab_model"
            st.download_button("Download model file", file_bytes, f"{filename}.nlmodel", "application/zip", key="model_export_download")
        except (TypeError, ValueError) as error:
            st.error(f"Model could not be exported: {error}")
    else:
        st.info("Train a model in Network Playground, Optimizer Lab, or Experiment Wizard before exporting it.")

    st.markdown("### Import a NeuroLab model")
    uploaded = st.file_uploader("Choose a .nlmodel file", type=["nlmodel", "zip"], key="model_import_upload")
    import_error = None
    imported_model = None
    manifest = None
    uploaded_bytes = None
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        try:
            imported_model, manifest = import_model(uploaded_bytes)
            user_metadata = manifest["metadata"]
            temporary = {"network": imported_model, "source": "Model file", "trained": bool(user_metadata.get("trained", False))}
            st.success("Model file is valid. Review its details, then explicitly load it into this session.")
            _render_model_details(temporary, manifest)
            if st.button("Load imported model into session", type="primary", key="model_import_confirm"):
                digest = sha256(uploaded_bytes).hexdigest()
                loaded = st.session_state.setdefault(IMPORTED_MODELS_KEY, [])
                if any(item.get("id") == digest for item in loaded):
                    st.info("This exact model file is already loaded in this session.")
                else:
                    loaded.append({"id": digest, "label": user_metadata.get("model_label", uploaded.name),
                                   "network": imported_model, "manifest": manifest})
                    st.success("Imported model loaded. Existing trained models were left unchanged.")
                st.rerun()
        except (TypeError, ValueError, OSError) as error:
            import_error = str(error)
            st.error(f"Model import rejected: {import_error}")

    st.markdown("### Loaded models")
    models = available_models(st.session_state)
    if models:
        selected_loaded_id = st.selectbox("Inspect model", tuple(models), format_func=lambda model_id: models[model_id]["label"], key="model_details_selection")
        _render_model_details(models[selected_loaded_id])
    else:
        st.info("No model is currently available in this session. Imported models appear here after you confirm loading.")

    st.markdown("### Predict with a model")
    if not models:
        st.caption("Load a model or complete training before making predictions.")
        return
    selected_prediction_id = st.selectbox("Model for prediction", tuple(models), format_func=lambda model_id: models[model_id]["label"], key="model_prediction_selection")
    selected_prediction_model = models[selected_prediction_id]
    network: NeuralNetwork = selected_prediction_model["network"]
    if not selected_prediction_model["trained"]:
        st.warning("This model is marked untrained. Predictions are available for inspection but do not indicate accuracy.")
    input_mode = st.radio("Input source", ("Enter one sample", "Upload CSV batch"), horizontal=True, key="model_prediction_mode")
    prediction_features: np.ndarray | None = None
    if input_mode == "Enter one sample":
        cols = st.columns(min(network.input_feature_count, 4))
        sample = []
        for index in range(network.input_feature_count):
            sample.append(cols[index % len(cols)].number_input(
                f"Feature {index + 1}", value=0.0, format="%.5f",
                key=f"model_input_{selected_prediction_id}_{index}",
            ))
        prediction_features = np.asarray(sample, dtype=float).reshape(1, -1)
    else:
        input_file = st.file_uploader("CSV with numeric feature columns only (no header)", type=["csv"], key="model_prediction_csv")
        if input_file is not None:
            try:
                prediction_features = parse_csv_features(input_file.getvalue(), network.input_feature_count)
                st.caption(f"Ready to predict {len(prediction_features)} samples.")
            except ValueError as error:
                st.error(str(error))

    if st.button("Generate predictions", type="primary", key="model_prediction_run") and prediction_features is not None:
        try:
            outputs, predicted_classes = predict_with_model(network, prediction_features, selected_prediction_model["metadata"])
            st.session_state[PREDICTION_KEY] = {
                "model_id": selected_prediction_id, "features": prediction_features.copy(),
                "outputs": outputs.copy(), "predicted_classes": predicted_classes.copy(),
            }
        except (ValueError, FloatingPointError) as error:
            st.error(f"Prediction could not be generated: {error}")

    prediction = st.session_state.get(PREDICTION_KEY)
    if prediction and prediction.get("model_id") == selected_prediction_id:
        outputs = prediction["outputs"]
        if network.output_size == 1:
            rows = [{"Sample": i + 1, "P(class 1)": float(row[0]), "P(class 0)": float(1 - row[0]),
                     "Predicted class": int(prediction["predicted_classes"][i])}
                    for i, row in enumerate(outputs)]
            st.caption("For a single sigmoid output, the displayed class-0 probability is 1 − P(class 1).")
        else:
            rows = [{"Sample": i + 1, **{f"Sigmoid score {j + 1}": float(value) for j, value in enumerate(row)},
                     "Predicted class": int(prediction["predicted_classes"][i])}
                    for i, row in enumerate(outputs)]
            st.caption("Multiple sigmoid outputs are independent scores, not normalized softmax probabilities; the displayed class is their argmax.")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("An imported model's predictions do not establish that it is accurate.")
