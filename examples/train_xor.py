"""Train a small NeuroLab model on a generated XOR dataset."""

from src.neurolab.datasets import make_classification_dataset, split_classification_data
from src.neurolab.training import train_model


def main() -> None:
    features, labels = make_classification_dataset("xor", n_samples=400, noise=0.05, random_seed=21)
    split = split_classification_data(features, labels, validation_fraction=0.2, random_seed=21)
    result = train_model(
        split.X_train,
        split.y_train,
        hidden_layer_sizes=(8, 8),
        hidden_activation="tanh",
        epochs=200,
        batch_size=32,
        optimizer="adam",
        learning_rate=0.01,
        validation_data=split.validation_data,
        random_seed=21,
    )
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Final training loss: {result.history['training_loss'][-1]:.4f}")
    print(f"Final validation loss: {result.history['validation_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {result.history['validation_accuracy'][-1]:.1%}")


if __name__ == "__main__":
    main()
