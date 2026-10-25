"""Interactive data inspection tools for NeuroLab's synthetic datasets."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.neurolab.datasets import make_classification_dataset, split_classification_data


DATASETS = {
    "XOR": "xor",
    "Two interleaving moons": "moons",
    "Concentric circles": "circles",
    "Spiral classification": "spiral",
}
CLASS_COLORS = ("#2F4156", "#397F8B")
CLASS_MARKERS = ("o", "^")


def calculate_dataset_statistics(features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Summarize generated feature values and class counts from the supplied data."""
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("labels must contain one value per sample.")
    classes, counts = np.unique(y, return_counts=True)
    return {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "classes": int(classes.size),
        "class_counts": {int(label): int(count) for label, count in zip(classes, counts)},
        "class_percentages": {int(label): float(count / X.shape[0] * 100) for label, count in zip(classes, counts)},
        "missing_values": int(np.isnan(X).sum()),
        "mean": np.nanmean(X, axis=0),
        "std": np.nanstd(X, axis=0),
        "minimum": np.nanmin(X, axis=0),
        "maximum": np.nanmax(X, axis=0),
    }


def _style_axis(axis: plt.Axes, title: str) -> None:
    """Apply the same light plotting style used in Network Playground."""
    axis.set_facecolor("#FFFFFF")
    axis.set_title(title, loc="left", color="#17263F", fontsize=11, pad=10)
    axis.tick_params(colors="#61758B", labelsize=8)
    axis.xaxis.label.set_color("#61758B")
    axis.yaxis.label.set_color("#61758B")
    for spine in axis.spines.values():
        spine.set_color("#D8E2EC")
    axis.grid(color="#D8E2EC", linewidth=0.6, alpha=0.75)


def _plot_scatter(features: np.ndarray, labels: np.ndarray, title: str, validation: bool = False) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="#F3F6FA")
    _style_axis(axis, title)
    y = np.asarray(labels).reshape(-1)
    for class_index, (color, marker) in enumerate(zip(CLASS_COLORS, CLASS_MARKERS)):
        mask = y == class_index
        if not np.any(mask):
            continue
        axis.scatter(
            features[mask, 0], features[mask, 1], c=color,
            marker="x" if validation else marker, s=32 if validation else 25,
            label=f"Class {class_index}", alpha=0.88,
            edgecolors="none" if validation else "#FFFFFF", linewidths=0.5,
        )
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.legend(facecolor="#FFFFFF", edgecolor="#D8E2EC", labelcolor="#17263F")
    figure.tight_layout()
    return figure


def _plot_split(split, standardize: bool) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="#F3F6FA")
    _style_axis(axis, "Stratified train / validation split")
    feature_label = "Standardized feature" if standardize else "Feature"
    for X, y, split_name, marker in (
        (split.X_train, split.y_train, "Training", "o"),
        (split.X_validation, split.y_validation, "Validation", "x"),
    ):
        y = np.asarray(y).reshape(-1)
        for class_index, color in enumerate(CLASS_COLORS):
            mask = y == class_index
            if np.any(mask):
                axis.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                             label=f"{split_name} · class {class_index}", s=28, alpha=0.82)
    axis.set_xlabel(f"{feature_label} 1")
    axis.set_ylabel(f"{feature_label} 2")
    axis.legend(facecolor="#FFFFFF", edgecolor="#D8E2EC", labelcolor="#17263F", fontsize=8)
    figure.tight_layout()
    return figure


def _plot_feature_analysis(features: np.ndarray, labels: np.ndarray) -> tuple[plt.Figure, plt.Figure]:
    y = np.asarray(labels).reshape(-1)
    distributions, axes = plt.subplots(1, features.shape[1], figsize=(8.0, 3.5), facecolor="#F3F6FA")
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        _style_axis(axis, f"Feature {index + 1} distribution")
        for class_index, color in enumerate(CLASS_COLORS):
            values = features[y == class_index, index]
            if values.size:
                axis.hist(values, bins=24, alpha=0.55, color=color, label=f"Class {class_index}")
        axis.set_xlabel(f"Feature {index + 1}")
        axis.set_ylabel("Samples")
        axis.legend(facecolor="#FFFFFF", edgecolor="#D8E2EC", labelcolor="#17263F", fontsize=8)
    distributions.tight_layout()

    correlation, axis = plt.subplots(figsize=(4.0, 3.5), facecolor="#F3F6FA")
    _style_axis(axis, "Feature correlation")
    matrix = np.corrcoef(features, rowvar=False)
    if not np.all(np.isfinite(matrix)):
        axis.text(0.5, 0.5, "Correlation is undefined\nfor a constant feature",
                  transform=axis.transAxes, ha="center", va="center", color="#17263F")
        axis.set_axis_off()
    else:
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="PuBuGn")
        axis.set_xticks(range(features.shape[1]), [f"F{i + 1}" for i in range(features.shape[1])])
        axis.set_yticks(range(features.shape[1]), [f"F{i + 1}" for i in range(features.shape[1])])
        for row in range(features.shape[1]):
            for column in range(features.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="#17263F")
        correlation.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    correlation.tight_layout()
    return distributions, correlation


@st.cache_data(show_spinner=False)
def _cached_dataset(name: str, samples: int, noise: float, seed: int,
                    class_1_fraction: float, circle_factor: float, spiral_turns: float):
    return make_classification_dataset(
        name, n_samples=samples, noise=noise, random_seed=seed,
        class_1_fraction=class_1_fraction, circle_factor=circle_factor, spiral_turns=spiral_turns,
    )


def _show_figure(figure: plt.Figure) -> None:
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _set_explorer_step(step: int) -> None:
    """Move the Dataset Explorer wizard to the requested step."""
    st.session_state["dataset_explorer_step"] = step


def render_dataset_explorer() -> None:
    """Render the guided dataset setup, analysis, and split preview."""
    with st.container(key="dataset_explorer_page"):
        _render_dataset_explorer_content()


def _render_dataset_explorer_content() -> None:
    st.markdown(
        """<section class="hero playground-hero">
            <div class="hero-copy">
                <div class="hero-kicker">DATA LAB · EXPLORE YOUR INPUTS</div>
                <h1>Dataset Explorer<span class="hero-period">.</span></h1>
                <div class="hero-tagline">See how data shapes a learning problem.</div>
                <p class="hero-description">Inspect synthetic datasets, class balance, noise, feature distributions, and train/validation splits.</p>
                <div class="hero-chip"><span class="status-dot"></span>Explore data before training</div>
            </div>
            <div class="hero-art" aria-hidden="true">
                <span class="orbit orbit-one"></span><span class="orbit orbit-two"></span>
                <span class="neuron n1"></span><span class="neuron n2"></span><span class="neuron n3"></span>
                <span class="neuron n4"></span><span class="neuron n5"></span><span class="neuron n6"></span>
                <span class="neuron n7"></span><span class="neuron n8"></span><span class="neuron n9"></span>
            </div>
        </section>""",
        unsafe_allow_html=True,
    )
    active_step = max(1, min(3, int(st.session_state.get("dataset_explorer_step", 1))))
    step_names = ("DATASET SETUP", "DATASET ANALYSIS", "TRAIN / VALIDATION")
    st.progress(active_step / 3, text=f"STEP {active_step} OF 3 · {step_names[active_step - 1]}")

    with st.expander("Step 1 · Dataset setup", expanded=active_step == 1):
        control_col, chart_col = st.columns([0.82, 1.18], gap="large")
        with control_col:
            dataset_label = st.selectbox("Dataset", tuple(DATASETS), key="explorer_dataset")
            samples = st.slider("Number of samples", 40, 2000, 300, 20, key="explorer_samples")
            noise = st.slider("Noise level", 0.0, 0.5, 0.08, 0.01, key="explorer_noise")
            seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=21, step=1, key="explorer_seed")
            class_1_fraction = st.slider(
                "Target share for class 1", 0.1, 0.9, 0.5, 0.05, key="explorer_class_1_fraction",
                help="The generator targets this share; the exact counts are shown in the analysis step.",
            )
            circle_factor = 0.5
            spiral_turns = 2.0
            if dataset_label == "Concentric circles":
                circle_factor = st.slider("Inner-circle radius factor", 0.15, 0.85, 0.5, 0.05, key="explorer_circle_factor")
            if dataset_label == "Spiral classification":
                spiral_turns = st.slider("Spiral turns", 1.0, 4.0, 2.0, 0.25, key="explorer_spiral_turns")
            validation_fraction = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05, key="explorer_validation_fraction")
            standardize = st.checkbox(
                "Preview training-only standardization", value=False, key="explorer_standardize",
                help="When enabled, the scaler is fitted on the training split and then applied to validation data.",
            )
        features, labels = _cached_dataset(
            DATASETS[dataset_label], int(samples), float(noise), int(seed),
            float(class_1_fraction), float(circle_factor), float(spiral_turns),
        )
        with chart_col:
            _show_figure(_plot_scatter(features, labels, f"{dataset_label} · {samples} samples"))
        if active_step == 1:
            _, next_column = st.columns([1, 0.42])
            with next_column:
                st.button("Continue to analysis →", type="primary", width="stretch", key="explorer_next_setup",
                          on_click=_set_explorer_step, args=(2,))

    with st.expander("Step 2 · Dataset analysis", expanded=active_step == 2):
        stats = calculate_dataset_statistics(features, labels)
        metrics = st.columns(5)
        metrics[0].metric("Samples", f"{stats['samples']:,}")
        metrics[1].metric("Features", str(stats["features"]))
        metrics[2].metric("Classes", str(stats["classes"]))
        metrics[3].metric("Class 0 / 1", f"{stats['class_counts'].get(0, 0):,} / {stats['class_counts'].get(1, 0):,}")
        metrics[4].metric("Missing values", f"{stats['missing_values']:,}")
        class_columns = st.columns(max(1, len(stats["class_counts"])))
        for column, (label, count) in zip(class_columns, stats["class_counts"].items()):
            column.caption(f"Class {label}: {count:,} samples · {stats['class_percentages'][label]:.1f}%")
        feature_rows = [
            {
                "Feature": f"Feature {index + 1}",
                "Mean": f"{stats['mean'][index]:.4f}",
                "Std. dev.": f"{stats['std'][index]:.4f}",
                "Minimum": f"{stats['minimum'][index]:.4f}",
                "Maximum": f"{stats['maximum'][index]:.4f}",
            }
            for index in range(int(stats["features"]))
        ]
        st.dataframe(feature_rows, hide_index=True, width="stretch")

        noise_col, point_col = st.columns([1.3, 0.7], gap="large")
        with noise_col:
            reference_features, reference_labels = _cached_dataset(
                DATASETS[dataset_label], int(samples), 0.0, int(seed), float(class_1_fraction),
                float(circle_factor), float(spiral_turns),
            )
            left, right = st.columns(2)
            with left:
                _show_figure(_plot_scatter(reference_features, reference_labels, "Same seed · zero noise"))
            with right:
                _show_figure(_plot_scatter(features, labels, f"Same seed · noise {noise:.2f}"))
            st.caption("Compare the zero-noise reference with the selected noise level. The point positions are generated data.")
        with point_col:
            point_number = st.number_input("Inspect sample number", 1, int(samples), 1, key="explorer_point_index")
            point_index = int(point_number) - 1
            st.markdown(f"**Sample {point_number}**")
            st.write({f"Feature {i + 1}": float(value) for i, value in enumerate(features[point_index])})
            st.write({"Class label": int(labels[point_index])})
            st.info("More noise can make classes overlap. Unequal class counts can make accuracy alone misleading; compare class distributions too.")

        hist_col, corr_col = st.columns([1.6, 0.8], gap="large")
        distribution_figure, correlation_figure = _plot_feature_analysis(features, labels)
        with hist_col:
            _show_figure(distribution_figure)
        with corr_col:
            _show_figure(correlation_figure)
        st.caption("Histograms show generated feature values by class. Correlation describes linear association between features, not class separability.")
        if active_step == 2:
            back_column, next_column = st.columns(2)
            with back_column:
                st.button("← Back to dataset", width="stretch", key="explorer_back_analysis",
                          on_click=_set_explorer_step, args=(1,))
            with next_column:
                st.button("Continue to split preview →", type="primary", width="stretch", key="explorer_next_analysis",
                          on_click=_set_explorer_step, args=(3,))

    with st.expander("Step 3 · Train / validation split", expanded=active_step == 3):
        try:
            split = split_classification_data(
                features, labels, validation_fraction=float(validation_fraction),
                random_seed=int(seed), standardize=bool(standardize),
            )
            split_figure, split_chart = st.columns([1.2, 0.8], gap="large")
            with split_figure:
                _show_figure(_plot_split(split, bool(standardize)))
            train_counts = Counter(np.asarray(split.y_train).reshape(-1).tolist())
            validation_counts = Counter(np.asarray(split.y_validation).reshape(-1).tolist())
            with split_chart:
                st.metric("Training samples", f"{len(split.y_train):,}")
                st.metric("Validation samples", f"{len(split.y_validation):,}")
                st.dataframe([
                    {"Class": f"Class {label}", "Training": train_counts.get(label, 0),
                     "Validation": validation_counts.get(label, 0)}
                    for label in sorted(set(train_counts) | set(validation_counts))
                ], hide_index=True, width="stretch")
                if standardize and split.scaler is not None:
                    st.caption("StandardScaler was fitted on training samples only.")
            st.caption("Validation data is held out to evaluate generalization; it is not used to update model weights.")
        except ValueError as error:
            st.warning(f"The split could not be created: {error}")

        st.markdown("**Use this dataset in Network Playground**")
        st.caption("Transfer the current dataset, sample count, noise, seed, class balance, and dataset-specific shape settings.")
        back_column, transfer_column = st.columns([0.42, 0.58])
        with back_column:
            st.button("← Back to analysis", width="stretch", key="explorer_back_split",
                      on_click=_set_explorer_step, args=(2,))
        with transfer_column:
            if st.button("Open in Network Playground →", type="primary", width="stretch", key="explorer_transfer"):
                st.session_state["play_dataset"] = dataset_label
                st.session_state["play_samples"] = int(samples)
                st.session_state["play_noise"] = min(0.5, float(noise))
                st.session_state["play_seed"] = int(seed)
                st.session_state["play_class_balance"] = f"{int(round(class_1_fraction * 100))}% class 1"
                st.session_state["play_circle_factor"] = float(circle_factor)
                st.session_state["play_spiral_turns"] = float(spiral_turns)
                st.session_state["play_validation_fraction"] = float(validation_fraction)
                st.session_state["network_playground_step"] = 1
                st.session_state["navigation_selection"] = "Network Playground"
                st.rerun()
