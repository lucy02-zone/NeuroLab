"""Basic checks for the initial NeuroLab application shell."""

from src.neurolab.dashboard import FEATURES
from src.neurolab.navigation import NAVIGATION


def test_navigation_contains_requested_sections():
    assert NAVIGATION == (
        "Dashboard",
        "Network Playground",
        "Dataset Explorer",
        "Experiment Wizard",
        "Model Manager",
        "Math Calculator",
        "Quiz & Challenge Center",
        "Network Visualizer",
        "Activation Functions",
        "Loss Functions",
        "Optimizer Lab",
        "Backpropagation Explorer",
        "Experiments & Reports",
        "Learn DNN",
    )


def test_dashboard_feature_cards_describe_current_capabilities():
    assert len(FEATURES) == 6
    assert all(title and description for _, title, description, _, _ in FEATURES)


def test_every_navigation_page_loads_without_streamlit_errors():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=45).run()
    for section in NAVIGATION:
        if section != "Dashboard":
            key = "nav_" + section.lower().replace(" ", "_").replace("&", "and")
            next(button for button in app.button if button.key == key).click().run()
        assert not app.exception, f"Page failed to render: {section}: {[error.message for error in app.exception]}"
