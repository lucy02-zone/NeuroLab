"""A compact integration check for NeuroLab's data-to-model lifecycle."""

import numpy as np

from src.neurolab.datasets import make_classification_dataset, split_classification_data
from src.neurolab.experiment_store import ExperimentStore
from src.neurolab.model_persistence import export_model, import_model
from src.neurolab.training import train_model


def test_dataset_training_history_evaluation_save_and_model_round_trip(tmp_path):
    features, labels = make_classification_dataset("moons", n_samples=80, noise=0.08, random_seed=17)
    split = split_classification_data(features, labels, validation_fraction=0.25, random_seed=17)
    result = train_model(
        split.X_train,
        split.y_train,
        hidden_layer_sizes=(5,),
        epochs=3,
        batch_size=12,
        optimizer="adam",
        learning_rate=0.01,
        validation_data=split.validation_data,
        random_seed=17,
    )

    assert result.epochs_completed == 3
    assert len(result.history["training_loss"]) == 3
    assert len(result.history["validation_loss"]) == 3
    assert len(result.history["training_accuracy"]) == 3
    assert len(result.history["validation_accuracy"]) == 3
    assert np.all(np.isfinite(result.network.forward(split.X_validation)))

    store = ExperimentStore(tmp_path / "workflow.sqlite3")
    record_id = store.save(
        name="integration run",
        dataset="moons",
        dataset_configuration={"samples": 80, "seed": 17},
        configuration={"optimizer": "adam", "hidden_layer_sizes": [5], "loss": "binary_cross_entropy"},
        history=result.history,
    )
    record = store.get(record_id)
    assert record is not None
    assert record["metrics"]["epochs_completed"] == 3
    assert record["metrics"]["final_validation_accuracy"] == result.history["validation_accuracy"][-1]

    samples = split.X_validation[:4]
    before = result.network.forward(samples)
    exported = export_model(result.network, {"trained": True, "source": "integration test"})
    restored, manifest = import_model(exported)
    after = restored.forward(samples)
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-12)
    assert manifest["metadata"]["trained"] is True

