"""Shared explicit Save Experiment controls for completed training runs."""

import hashlib
import json
from typing import Any

import streamlit as st

from .experiment_store import ExperimentStore


SAVED_EXPERIMENTS_KEY = "saved_experiment_fingerprints"


def experiment_fingerprint(candidate: dict[str, object]) -> str:
    """Return a stable digest for one run's dataset, configuration, and history."""
    def normalize(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Cannot fingerprint value of type {type(value).__name__}.")

    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"), default=normalize, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_save_experiment_control(candidates: dict[str, dict[str, object]], key: str) -> None:
    """Render a save form for actual run histories supplied by an existing page."""
    with st.expander("Save Experiment", expanded=False):
        if not candidates:
            st.info("A completed training run is required before it can be saved.")
            return
        candidate_label = (
            st.selectbox("Run to save", tuple(candidates), key=f"{key}_candidate")
            if len(candidates) > 1 else next(iter(candidates))
        )
        candidate = candidates[candidate_label]
        try:
            fingerprint = experiment_fingerprint(candidate)
        except (TypeError, ValueError) as error:
            st.error(f"This run cannot be saved because its record data is invalid: {error}")
            return
        saved_records = st.session_state.setdefault(SAVED_EXPERIMENTS_KEY, {})
        saved_record_id = saved_records.get(fingerprint)
        if saved_record_id is not None:
            st.success(f"This run is already saved as experiment #{saved_record_id}.")
            return
        default_name = f"{candidate['dataset']} · {candidate['configuration'].get('optimizer', candidate_label)}"
        name = st.text_input("Experiment name", value=default_name, key=f"{key}_name")
        notes = st.text_area("Description or notes (optional)", key=f"{key}_notes")
        if st.button("Save Experiment", key=f"{key}_save", type="primary"):
            try:
                record_id = ExperimentStore().save(
                    name=name,
                    dataset=candidate["dataset"],
                    dataset_configuration=candidate["dataset_configuration"],
                    configuration=candidate["configuration"],
                    history=candidate["history"],
                    notes=notes,
                )
                saved_records[fingerprint] = record_id
                st.success(f"Saved experiment #{record_id}. Open Experiments & Reports to view or export it.")
            except (TypeError, ValueError, OSError) as error:
                st.error(f"Experiment could not be saved: {error}")
