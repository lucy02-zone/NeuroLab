"""Streamlit history, comparison, and export page for saved NeuroLab runs."""

import matplotlib.pyplot as plt
import streamlit as st

from .experiment_store import (
    SORT_FIELDS,
    ExperimentStore,
    compare_experiments,
    history_csv,
    markdown_report,
    summary_csv,
)
from .visualization import BLUE, CYAN, GRID, MUTED, NAVY, TEXT, VIOLET


PENDING_DELETE_KEY = "experiment_pending_delete_id"


def _history_figure(records: list[dict[str, object]], metric: str, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.0, 4.2), facecolor=NAVY)
    axis.set_facecolor(NAVY)
    colors = (BLUE, CYAN, VIOLET)
    for index, record in enumerate(records):
        history = record["history"]
        axis.plot(history["epoch"], history[metric], linewidth=2.1,
                  color=colors[index % len(colors)], label=f"#{record['id']} {record['name']}")
    axis.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    axis.set_xlabel("Epoch", color=MUTED)
    axis.set_ylabel("Loss", color=MUTED)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, linewidth=0.65, alpha=0.55)
    axis.legend(facecolor="#141f30", edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout()
    return figure


def _summary_row(record: dict[str, object]) -> dict[str, str]:
    config, metrics = record["configuration"], record["metrics"]
    architecture = config.get("hidden_layer_sizes", config.get("architecture", []))
    architecture_text = "Input → " + (" → ".join(map(str, architecture)) + " → " if architecture else "") + "Output"
    accuracy = metrics.get("final_validation_accuracy")
    if accuracy is None:
        accuracy = metrics.get("final_training_accuracy")
    return {
        "ID": str(record["id"]), "Experiment": record["name"], "Date (UTC)": record["created_at"],
        "Dataset": record["dataset"], "Optimizer": str(config.get("optimizer", "—")),
        "Architecture": architecture_text, "Final loss": _format_metric(metrics.get("final_training_loss")),
        "Final accuracy": _format_metric(accuracy, percent=True),
    }


def _format_metric(value: object, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}" if percent else f"{value:.5f}"


def render_experiments_reports() -> None:
    """Render saved experiment history and report controls."""
    st.markdown('<div class="eyebrow">REVIEW AND EXPORT REAL TRAINING RUNS</div>', unsafe_allow_html=True)
    st.title("Experiments & Reports")
    st.markdown("Saved experiment records are stored locally in SQLite and contain configuration and numeric history, not serialized model objects.")
    store = ExperimentStore()
    all_records = store.list(sort_by="Date", descending=True)

    st.markdown("### 1 · Experiment history")
    if not all_records:
        st.info("No saved experiments yet. Run a model in Network Playground or Optimizer Lab, then use its **Save Experiment** control.")
        st.markdown("### 2 · Experiment details")
        st.caption("Save a completed run to make its configuration and metrics available here.")
        return

    controls = st.columns(4)
    datasets = ["All", *sorted({record["dataset"] for record in all_records})]
    optimizers = ["All", *sorted({str(record["configuration"].get("optimizer", "Unknown")) for record in all_records})]
    with controls[0]:
        dataset_filter = st.selectbox("Filter dataset", datasets, key="reports_dataset_filter")
    with controls[1]:
        optimizer_filter = st.selectbox("Filter optimizer", optimizers, key="reports_optimizer_filter")
    with controls[2]:
        sort_by = st.selectbox("Sort by", tuple(SORT_FIELDS), key="reports_sort")
    with controls[3]:
        descending = st.selectbox("Order", ("Descending", "Ascending"), key="reports_order") == "Descending"
    records = store.list(dataset=dataset_filter, optimizer=optimizer_filter, sort_by=sort_by, descending=descending)
    if records:
        st.table([_summary_row(record) for record in records])
    else:
        st.info("No saved experiments match these filters.")

    st.markdown("### 2 · Experiment details")
    detail_options = {f"#{record['id']} · {record['name']} · {record['dataset']}": record for record in records or all_records}
    detail_label = st.selectbox("View saved experiment", tuple(detail_options), key="reports_detail")
    selected = detail_options[detail_label]
    detail_columns = st.columns(2, gap="large")
    with detail_columns[0]:
        st.markdown("#### Configuration")
        st.json({"dataset": selected["dataset"], "dataset_configuration": selected["dataset_configuration"],
                 "experiment": selected["configuration"], "task_type": selected["task_type"]})
        if selected["notes"]:
            st.markdown("#### Notes")
            st.write(selected["notes"])
    with detail_columns[1]:
        st.markdown("#### Final metrics")
        st.table([{"Metric": label, "Value": _format_metric(selected["metrics"].get(field), percent=percent)}
                  for label, field, percent in (
                      ("Training loss", "final_training_loss", False),
                      ("Validation loss", "final_validation_loss", False),
                      ("Training accuracy", "final_training_accuracy", True),
                      ("Validation accuracy", "final_validation_accuracy", True),
                      ("Epochs completed", "epochs_completed", False),
                  )])
        st.markdown("#### Epoch history")
        history_rows = []
        for index, epoch in enumerate(selected["history"]["epoch"]):
            history_rows.append({"Epoch": epoch, **{
                metric.replace("_", " ").title(): selected["history"].get(metric, [None] * len(selected["history"]["epoch"]))[index]
                for metric in ("training_loss", "validation_loss", "training_accuracy", "validation_accuracy")
            }})
        st.dataframe(history_rows, hide_index=True, width="stretch")

    st.markdown("### 3 · Comparison workspace")
    comparison_labels = st.multiselect(
        "Select two or more saved experiments",
        tuple(f"#{record['id']} · {record['name']} · {record['dataset']}" for record in all_records),
        key="reports_compare_selection",
    )
    by_label = {f"#{record['id']} · {record['name']} · {record['dataset']}": record for record in all_records}
    if len(comparison_labels) >= 2:
        comparison = compare_experiments([by_label[label] for label in comparison_labels])
        if comparison["metrics_comparable"]:
            st.success("These runs share the same task, dataset, and dataset/split configuration. Their recorded metrics can be compared directly.")
        else:
            st.warning("Final metrics are not directly comparable: " + " ".join(comparison["reasons"]))
        st.table(comparison["rows"])
        chosen = [by_label[label] for label in comparison_labels]
        if comparison["metrics_comparable"]:
            _render_comparison_charts(chosen)
        else:
            st.caption("Configuration details are shown above. Metric charts are hidden because the recorded evaluation bases differ.")
    else:
        st.caption("Choose at least two records to compare architectures, settings, metrics, and (when comparable) training curves.")

    st.markdown("### 4 · Export and deletion")
    exports = st.columns(3)
    with exports[0]:
        st.download_button("Download summary CSV", summary_csv(all_records), "neurolab_experiments.csv", "text/csv", key="reports_summary_csv")
    with exports[1]:
        st.download_button("Download epoch history CSV", history_csv(all_records), "neurolab_training_history.csv", "text/csv", key="reports_history_csv")
    with exports[2]:
        report = markdown_report(selected)
        st.download_button("Download selected Markdown report", report, f"experiment_{selected['id']}.md", "text/markdown", key="reports_markdown")

    st.markdown("#### Delete a saved experiment")
    delete_id = int(selected["id"])
    if st.button("Delete selected experiment", key="reports_delete_request"):
        st.session_state[PENDING_DELETE_KEY] = delete_id
    if st.session_state.get(PENDING_DELETE_KEY) == delete_id:
        st.warning(f"Delete **{selected['name']}** permanently from the local experiment history?")
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm Delete", type="primary", key="reports_delete_confirm"):
            store.delete(delete_id)
            del st.session_state[PENDING_DELETE_KEY]
            st.success("Experiment deleted.")
            st.rerun()
        if cancel.button("Cancel", key="reports_delete_cancel"):
            del st.session_state[PENDING_DELETE_KEY]
            st.rerun()


def _render_comparison_charts(records: list[dict[str, object]]) -> None:
    for metric, title in (("training_loss", "Training loss comparison"), ("validation_loss", "Validation loss comparison")):
        if all(metric in record["history"] for record in records):
            figure = _history_figure(records, metric, title)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        elif metric == "validation_loss":
            st.caption("A validation-loss curve is unavailable for at least one selected experiment.")
