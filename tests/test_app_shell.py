"""Basic checks for the initial NeuroLab application shell."""

from src.neurolab.dashboard import FEATURES
from src.neurolab.navigation import NAVIGATION


def test_navigation_contains_requested_sections():
    assert NAVIGATION == (
        "Dashboard",
        "Network Playground",
        "Dataset Explorer",
        "Network Visualizer",
        "Activation Functions",
        "Loss Functions",
        "Optimizer Lab",
        "Backpropagation Explorer",
        "Experiments & Reports",
        "Learn DNN",
    )


def test_dashboard_feature_cards_describe_planned_capabilities():
    assert len(FEATURES) == 6
    assert all(title and description for _, title, description, _, _ in FEATURES)
