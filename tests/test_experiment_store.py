import csv
import io

import pytest

from src.neurolab.experiment_store import (
    ExperimentStore,
    compare_experiments,
    history_csv,
    markdown_report,
    summary_csv,
)


def _save(store, *, name="Trial", dataset="moons", validation_fraction=0.25, history=None, learning_rate=0.02):
    return store.save(
        name=name,
        dataset=dataset,
        dataset_configuration={"n_samples": 30, "noise": 0.1, "validation_fraction": validation_fraction,
                              "split_seed": 7, "standardize": True},
        configuration={"hidden_layer_sizes": [4, 3], "hidden_activation": "tanh", "output_activation": "sigmoid",
                       "optimizer": "Adam", "optimizer_config": {"beta1": 0.9, "beta2": 0.999},
                       "learning_rate": learning_rate, "batch_size": 8, "loss": "binary_cross_entropy",
                       "random_seed": 7},
        history=history or {
            "epoch": [1, 2], "training_loss": [0.7, 0.5], "validation_loss": [0.8, 0.6],
            "training_accuracy": [0.55, 0.75], "validation_accuracy": [0.5, 0.7],
        },
        notes="First line, with comma\nsecond line",
    )


def test_save_retrieve_and_survive_a_new_store_instance(tmp_path):
    path = tmp_path / "runs.sqlite3"
    first_store = ExperimentStore(path)
    record_id = _save(first_store, name="Café, run one")
    retrieved = ExperimentStore(path).get(record_id)

    assert retrieved["name"] == "Café, run one"
    assert retrieved["metrics"]["final_training_loss"] == 0.5
    assert retrieved["metrics"]["final_validation_accuracy"] == 0.7
    assert retrieved["task_type"] == "classification"
    assert retrieved["history"]["epoch"] == [1, 2]


def test_filter_sort_and_missing_metrics_are_preserved(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    _save(store, name="A", learning_rate=0.1)
    _save(store, name="B", dataset="circles", learning_rate=0.2,
          history={"epoch": [1], "training_loss": [0.3]})

    assert [record["name"] for record in store.list(dataset="moons", optimizer="Adam")] == ["A"]
    ordered = store.list(sort_by="Training loss", descending=False)
    assert [record["name"] for record in ordered] == ["B", "A"]
    assert ordered[0]["metrics"]["final_validation_loss"] is None
    assert ordered[0]["metrics"]["final_validation_accuracy"] is None


def test_comparison_explains_different_validation_splits_and_keeps_missing_values_none(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    first_id = _save(store, name="Baseline")
    second_id = _save(store, name="Changed split", validation_fraction=0.3,
                      history={"epoch": [1], "training_loss": [0.4]})
    result = compare_experiments([store.get(first_id), store.get(second_id)])

    assert not result["metrics_comparable"]
    assert any("split configuration differs" in reason for reason in result["reasons"])
    assert result["rows"][1]["Validation loss"] is None
    assert result["rows"][1]["Validation accuracy"] is None


def test_exports_quote_special_characters_and_do_not_fill_missing_metrics(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    record_id = _save(store, name="=unsafe, experiment")
    record = store.get(record_id)

    summary_rows = list(csv.DictReader(io.StringIO(summary_csv([record]))))
    epoch_rows = list(csv.DictReader(io.StringIO(history_csv([record]))))
    report = markdown_report(record)

    assert summary_rows[0]["name"] == "'=unsafe, experiment"
    assert summary_rows[0]["notes"] == "First line, with comma\nsecond line"
    assert epoch_rows[0]["validation_loss"] == "0.8"
    assert "## Epoch history" in report
    assert '"random_seed": 7' in report


def test_invalid_experiment_history_and_non_json_values_are_rejected(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    with pytest.raises(ValueError, match="name is required"):
        _save(store, name=" ")
    with pytest.raises(ValueError, match="matching non-empty"):
        _save(store, history={"epoch": [], "training_loss": []})
    with pytest.raises(ValueError, match="finite JSON"):
        store.save(name="Bad", dataset="xor", dataset_configuration={"noise": float("nan")},
                   configuration={"optimizer": "SGD"}, history={"epoch": [1], "training_loss": [0.1]})
    assert store.list() == []


def test_delete_removes_only_requested_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    first = _save(store, name="Remove")
    second = _save(store, name="Keep")

    assert store.delete(first)
    assert store.get(first) is None
    assert store.get(second)["name"] == "Keep"
    assert not store.delete(first)
