"""Run a small, untrained forward-propagation example."""

import numpy as np

from src.neurolab.models import NeuralNetwork


def main() -> None:
    network = NeuralNetwork(
        input_feature_count=2,
        hidden_layer_sizes=[4],
        output_size=1,
        random_seed=7,
    )
    samples = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
    predictions = network.forward(samples)
    print(f"Output shape: {predictions.shape}")
    print("Randomly initialized predictions (untrained):")
    print(predictions)


if __name__ == "__main__":
    main()
