"""Validated SQLite persistence and safe exports for completed experiments."""

import csv
from contextlib import contextmanager
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SORT_FIELDS = {
    "Date": ("created_at", True),
    "Training loss": ("final_training_loss", False),
    "Validation loss": ("final_validation_loss", False),
    "Training accuracy": ("final_training_accuracy", True),
    "Validation accuracy": ("final_validation_accuracy", True),
}


def default_database_path() -> Path:
    """Return the project's local database path, optionally overridden for deployment/tests."""
    configured = os.environ.get("NEUROLAB_EXPERIMENT_DB")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "data" / "experiments.sqlite3"


class ExperimentStore:
    """Store experiment metadata and numeric histories without model pickles."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path) if database_path is not None else default_database_path()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    dataset_configuration TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    history TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                )"""
            )

    def save(
        self,
        *,
        name: str,
        dataset: str,
        dataset_configuration: dict[str, Any],
        configuration: dict[str, Any],
        history: dict[str, list[int | float]],
        notes: str = "",
    ) -> int:
        """Validate genuine recorded history, derive final metrics, and persist JSON data."""
        clean_name = name.strip() if isinstance(name, str) else ""
        clean_dataset = dataset.strip() if isinstance(dataset, str) else ""
        if not clean_name:
            raise ValueError("Experiment name is required.")
        if not clean_dataset:
            raise ValueError("Dataset name is required.")
        if not isinstance(dataset_configuration, dict) or not isinstance(configuration, dict):
            raise ValueError("Dataset and experiment configuration must be dictionaries.")
        if not isinstance(history, dict):
            raise ValueError("Training history must be a dictionary of recorded epoch metrics.")
        epochs = history.get("epoch")
        training_loss = history.get("training_loss")
        if not isinstance(epochs, list) or not epochs or not isinstance(training_loss, list) or len(training_loss) != len(epochs):
            raise ValueError("Completed experiment history must contain matching non-empty epoch and training_loss lists.")
        if any(not isinstance(epoch, (int, float)) for epoch in epochs):
            raise ValueError("Epoch values must be numeric.")
        for metric_name, values in history.items():
            if not isinstance(values, list) or len(values) != len(epochs):
                raise ValueError(f"History metric {metric_name!r} must contain one value per epoch.")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                raise ValueError(f"History metric {metric_name!r} must contain only numeric values.")
        task_type = "classification" if "training_accuracy" in history else "regression"
        metrics = {
            "epochs_completed": int(epochs[-1]),
            "final_training_loss": float(training_loss[-1]),
            "final_validation_loss": _final(history, "validation_loss"),
            "final_training_accuracy": _final(history, "training_accuracy"),
            "final_validation_accuracy": _final(history, "validation_accuracy"),
        }
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO experiments
                   (name, created_at, dataset, dataset_configuration, configuration, task_type, metrics, history, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clean_name,
                    timestamp,
                    clean_dataset,
                    _json(dataset_configuration),
                    _json(configuration),
                    task_type,
                    _json(metrics),
                    _json(history),
                    notes.strip() if isinstance(notes, str) else "",
                ),
            )
            return int(cursor.lastrowid)

    def list(
        self,
        *,
        dataset: str | None = None,
        optimizer: str | None = None,
        sort_by: str = "Date",
        descending: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching records, with missing metric values always sorted last."""
        if sort_by not in SORT_FIELDS:
            raise ValueError(f"sort_by must be one of {tuple(SORT_FIELDS)}.")
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM experiments").fetchall()
        records = [_decode(row) for row in rows]
        if dataset and dataset != "All":
            records = [record for record in records if record["dataset"] == dataset]
        if optimizer and optimizer != "All":
            records = [record for record in records if record["configuration"].get("optimizer") == optimizer]
        field, default_descending = SORT_FIELDS[sort_by]
        descending = default_descending if descending is None else descending
        if field == "created_at":
            records.sort(key=lambda record: record[field], reverse=descending)
        else:
            valued = [record for record in records if record["metrics"].get(field) is not None]
            missing = [record for record in records if record["metrics"].get(field) is None]
            valued.sort(key=lambda record: record["metrics"][field], reverse=descending)
            records = valued + missing
        return records

    def get(self, experiment_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM experiments WHERE id = ?", (int(experiment_id),)).fetchone()
        return _decode(row) if row is not None else None

    def delete(self, experiment_id: int) -> bool:
        """Delete one record; callers should gate this behind an explicit confirmation UI."""
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM experiments WHERE id = ?", (int(experiment_id),))
            return cursor.rowcount == 1


def compare_experiments(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe configuration differences and whether final metrics share a task and split."""
    if len(records) < 2:
        raise ValueError("Select at least two experiments to compare.")
    first = records[0]
    same_task = all(record["task_type"] == first["task_type"] for record in records)
    same_dataset = all(record["dataset"] == first["dataset"] for record in records)
    same_data_config = all(record["dataset_configuration"] == first["dataset_configuration"] for record in records)
    same_loss = all(record["configuration"].get("loss") == first["configuration"].get("loss") for record in records)
    reasons = []
    if not same_task:
        reasons.append("Task types differ (classification vs regression).")
    if not same_dataset:
        reasons.append("Datasets differ.")
    if not same_data_config:
        reasons.append("Dataset or train/validation split configuration differs.")
    if not same_loss:
        reasons.append("Loss functions differ, so loss values are on different scales.")
    rows = []
    for record in records:
        config = record["configuration"]
        metrics = record["metrics"]
        rows.append({
            "Experiment": record["name"], "Dataset": record["dataset"], "Task": record["task_type"],
            "Architecture": _architecture(config), "Activation": _activation(config),
            "Loss": config.get("loss", "—"), "Optimizer": config.get("optimizer", "—"),
            "Learning rate": config.get("learning_rate"),
            "Epochs completed": metrics.get("epochs_completed"),
            "Training loss": metrics.get("final_training_loss"),
            "Validation loss": metrics.get("final_validation_loss"),
            "Training accuracy": metrics.get("final_training_accuracy"),
            "Validation accuracy": metrics.get("final_validation_accuracy"),
        })
    return {"rows": rows, "metrics_comparable": same_task and same_dataset and same_data_config and same_loss,
            "accuracy_comparable": same_task and same_dataset and same_data_config,
            "reasons": reasons}


def summary_csv(records: list[dict[str, Any]]) -> str:
    """Serialize experiment summaries with CSV quoting for commas/newlines safely."""
    columns = ["id", "name", "created_at", "dataset", "optimizer", "architecture", "activation",
               "loss", "learning_rate", "batch_size", "epochs_completed", "final_training_loss",
               "final_validation_loss", "final_training_accuracy", "final_validation_accuracy", "notes"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        config, metrics = record["configuration"], record["metrics"]
        writer.writerow({
            "id": record["id"], "name": _safe_csv_text(record["name"]), "created_at": record["created_at"],
            "dataset": _safe_csv_text(record["dataset"]), "optimizer": _safe_csv_text(config.get("optimizer", "")),
            "architecture": _safe_csv_text(_architecture(config)), "activation": _safe_csv_text(_activation(config)),
            "loss": _safe_csv_text(config.get("loss", "")), "learning_rate": config.get("learning_rate", ""),
            "batch_size": config.get("batch_size", ""), **metrics, "notes": _safe_csv_text(record["notes"]),
        })
    return output.getvalue()


def history_csv(records: list[dict[str, Any]]) -> str:
    """Export actual per-epoch metrics; missing metrics remain blank."""
    columns = ["experiment_id", "experiment_name", "epoch", "training_loss", "validation_loss",
               "training_accuracy", "validation_accuracy"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        history = record["history"]
        for index, epoch in enumerate(history["epoch"]):
            row = {"experiment_id": record["id"], "experiment_name": _safe_csv_text(record["name"]), "epoch": epoch}
            for metric in columns[3:]:
                values = history.get(metric)
                row[metric] = values[index] if values is not None else ""
            writer.writerow(row)
    return output.getvalue()


def markdown_report(record: dict[str, Any]) -> str:
    """Create a portable Markdown report from one stored experiment record."""
    config = record["configuration"]
    metrics = record["metrics"]
    lines = [f"# {record['name']}", "", f"- **Saved:** {record['created_at']}",
             f"- **Dataset:** {record['dataset']}", f"- **Task:** {record['task_type']}",
             f"- **Optimizer:** {config.get('optimizer', '—')}", f"- **Architecture:** {_architecture(config)}",
             f"- **Activation:** {_activation(config)}", f"- **Loss:** {config.get('loss', '—')}",
             f"- **Learning rate:** {config.get('learning_rate', '—')}",
             f"- **Epochs completed:** {metrics.get('epochs_completed', '—')}", "", "## Dataset configuration", "",
             "```json", json.dumps(record["dataset_configuration"], indent=2, ensure_ascii=False), "```", "",
             "## Experiment configuration", "", "```json",
             json.dumps(config, indent=2, ensure_ascii=False), "```", "", "## Final metrics", "",
             "| Metric | Value |", "|---|---:|"]
    for label, key in (("Training loss", "final_training_loss"), ("Validation loss", "final_validation_loss"),
                       ("Training accuracy", "final_training_accuracy"), ("Validation accuracy", "final_validation_accuracy")):
        value = metrics.get(key)
        lines.append(f"| {label} | {value if value is not None else 'Not available'} |")
    if record["notes"]:
        lines.extend(["", "## Notes", "", record["notes"]])
    lines.extend(["", "## Epoch history", "", "```csv", history_csv([record]).strip(), "```", ""])
    return "\n".join(lines)


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]), "name": row["name"], "created_at": row["created_at"],
        "dataset": row["dataset"], "dataset_configuration": json.loads(row["dataset_configuration"]),
        "configuration": json.loads(row["configuration"]), "task_type": row["task_type"],
        "metrics": json.loads(row["metrics"]), "history": json.loads(row["history"]), "notes": row["notes"],
    }


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Experiment data must contain only finite JSON-compatible values.") from error


def _final(history: dict[str, list[int | float]], name: str) -> float | None:
    values = history.get(name)
    return float(values[-1]) if values else None


def _architecture(configuration: dict[str, Any]) -> str:
    widths = configuration.get("hidden_layer_sizes", configuration.get("architecture", []))
    if isinstance(widths, (list, tuple)):
        return "Input → " + (" → ".join(map(str, widths)) + " → " if widths else "") + "Output"
    return str(widths or "—")


def _activation(configuration: dict[str, Any]) -> str:
    hidden = configuration.get("hidden_activation", configuration.get("activation", "—"))
    output = configuration.get("output_activation")
    return f"{hidden} / {output}" if output else str(hidden)


def _safe_csv_text(value: Any) -> Any:
    """Prevent user-provided text from being interpreted as a spreadsheet formula."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
