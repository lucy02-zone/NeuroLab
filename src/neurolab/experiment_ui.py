"""Shared explicit Save Experiment controls for completed training runs."""

import streamlit as st

from .experiment_store import ExperimentStore


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
                st.success(f"Saved experiment #{record_id}. Open Experiments & Reports to view or export it.")
            except (TypeError, ValueError, OSError) as error:
                st.error(f"Experiment could not be saved: {error}")
