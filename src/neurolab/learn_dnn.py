"""Eight-module interactive learning center for deep neural networks."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .experiment_store import ExperimentStore, default_database_path
from .models import Adam, NeuralNetwork, SGD, SGDMomentum, activations, losses
from .visualization import BLUE, CYAN, GRID, TEXT, VIOLET, style_axis


MODULES = (
    "1 · Introduction to Neural Networks",
    "2 · Forward Propagation",
    "3 · Activation Functions",
    "4 · Loss Functions",
    "5 · Backpropagation",
    "6 · Gradient Descent and Optimizers",
    "7 · Overfitting and Generalization",
    "8 · Model Evaluation",
)
MODULE_IDS = tuple(f"module_{index}" for index in range(1, 9))
COMPLETED_KEY = "learn_dnn_completed_modules"
FEEDBACK_PREFIX = "learn_dnn_quiz_feedback_"


def evaluate_quiz(selected_answer: str, correct_answer: str) -> bool:
    """Return whether a learner selected the exact expected quiz choice."""
    return selected_answer == correct_answer


def update_progress(completed: set[str] | list[str] | tuple[str, ...], module_id: str, passed: bool) -> set[str]:
    """Return the completed module set, adding a module only after a correct answer."""
    result = set(completed)
    if passed and module_id in MODULE_IDS:
        result.add(module_id)
    return result


def forward_example(inputs: np.ndarray, hidden_bias: float = 0.1) -> dict[str, np.ndarray | float]:
    """Run a fixed two-input, two-hidden-unit network through the real model API."""
    network = NeuralNetwork(2, [2], 1, random_seed=0, hidden_activation="tanh")
    network.weights[0][...] = np.array([[0.5, -0.25], [-0.4, 0.6]])
    network.biases[0][...] = np.array([[hidden_bias], [-0.1]])
    network.weights[1][...] = np.array([[0.7, -0.3]])
    network.biases[1][...] = np.array([[0.05]])
    prediction = network.forward(np.asarray(inputs, dtype=float))
    return {
        "network": network,
        "hidden_z": network.weighted_sums[0].copy(),
        "hidden_a": network.activations[1].copy(),
        "output_z": network.weighted_sums[1].copy(),
        "prediction": prediction.copy(),
    }


def classification_metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, object]:
    """Calculate binary classification metrics and a [[TN,FP],[FN,TP]] matrix."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_predicted)
    if truth.ndim != 1 or predicted.ndim != 1 or truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("y_true and y_predicted must be non-empty one-dimensional arrays with matching shapes.")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(predicted, (0, 1))):
        raise ValueError("Binary labels must contain only 0 and 1.")
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / truth.size, "precision": precision, "recall": recall,
            "f1": f1, "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=int)}


def _style_axis(axis: plt.Axes, title: str) -> None:
    style_axis(axis, title)
    axis.axhline(0, color=GRID, linewidth=0.8)


def _mark_navigation(destination: str) -> None:
    """Navigation callback runs before Streamlit instantiates the keyed radio."""
    st.session_state["navigation_selection"] = destination


def _quiz(module_id: str, question: str, options: tuple[str, ...], correct: str, explanation: str) -> None:
    st.markdown("#### Knowledge check")
    answer = st.radio(question, options, key=f"learn_quiz_{module_id}")
    if st.button("Check answer", key=f"learn_check_{module_id}"):
        passed = evaluate_quiz(answer, correct)
        st.session_state[f"{FEEDBACK_PREFIX}{module_id}"] = {"passed": passed, "explanation": explanation}
        if passed:
            current = st.session_state.get(COMPLETED_KEY, [])
            st.session_state[COMPLETED_KEY] = sorted(update_progress(current, module_id, True), key=MODULE_IDS.index)
    feedback = st.session_state.get(f"{FEEDBACK_PREFIX}{module_id}")
    if feedback:
        if feedback["passed"]:
            st.success("Correct. " + feedback["explanation"])
        else:
            st.error("Not quite. " + feedback["explanation"])


def _objectives(items: tuple[str, ...]) -> None:
    st.markdown("#### Learning objectives")
    for item in items:
        st.markdown(f"- {item}")


def _practical_link(label: str, destination: str, key: str) -> None:
    st.button(f"Open {label}", key=key, on_click=_mark_navigation, args=(destination,))


def _render_module_1() -> None:
    _objectives(("Describe what an artificial neuron computes.", "Identify weights, bias, and network layers.",
                 "Distinguish shallow and deep networks."))
    st.markdown("#### Concept")
    st.write("An artificial neuron (a perceptron in its simplest form) combines input features using learned weights, adds a bias, and passes the result through an activation. "
             "Input layers carry features, hidden layers build intermediate representations, and output layers produce predictions. "
             "A shallow network has one hidden layer; a deep network has multiple hidden layers and can compose more stages of representation.")
    st.markdown("#### Mathematical formula")
    st.latex(r"z = \sum_{j=1}^{n} w_jx_j + b,\qquad a = f(z)")
    st.markdown("#### Worked example · one sigmoid neuron")
    controls = st.columns(5)
    x1 = controls[0].number_input("x₁", value=0.6, step=0.1, key="learn_m1_x1")
    x2 = controls[1].number_input("x₂", value=0.2, step=0.1, key="learn_m1_x2")
    w1 = controls[2].number_input("w₁", value=0.8, step=0.1, key="learn_m1_w1")
    w2 = controls[3].number_input("w₂", value=-0.4, step=0.1, key="learn_m1_w2")
    bias = controls[4].number_input("Bias b", value=0.1, step=0.1, key="learn_m1_bias")
    network = NeuralNetwork(2, [], 1, random_seed=0)
    network.weights[0][...] = [[w1, w2]]
    network.biases[0][...] = [[bias]]
    output = network.forward(np.array([x1, x2]))
    z_value = float(network.weighted_sums[0][0, 0])
    st.code(f"z = {w1:.3f}·{x1:.3f} + {w2:.3f}·{x2:.3f} + {bias:.3f} = {z_value:.6f}\n"
            f"a = sigmoid(z) = {float(output[0, 0]):.6f}", language="text")
    st.caption("This is a one-neuron calculation performed by NeuroLab's NeuralNetwork.forward().")
    _practical_link("Network Playground", "Network Playground", "learn_link_network_1")
    _quiz(MODULE_IDS[0], "What does a bias let a neuron do?",
          ("Shift its activation threshold", "Normalize all input features", "Choose the optimizer"),
          "Shift its activation threshold", "The bias shifts the weighted sum before activation.")


def _render_module_2() -> None:
    _objectives(("Compute a weighted sum for a layer.", "Explain the roles of the bias and activation.",
                 "Read cached values from multiple layers."))
    st.markdown("#### Concept")
    st.write("Forward propagation carries an input through the network from left to right. Each layer first computes a weighted sum and bias, "
             "then applies its activation. A layer's activation becomes the next layer's input.")
    st.markdown("#### Mathematical formula")
    st.latex(r"z = Wx + b")
    st.latex(r"a = f(z)")
    st.markdown("#### Worked example · two inputs, two hidden neurons, one output")
    controls = st.columns(3)
    x1 = controls[0].slider("Input x₁", -1.0, 1.0, 0.4, 0.1, key="learn_m2_x1")
    x2 = controls[1].slider("Input x₂", -1.0, 1.0, -0.2, 0.1, key="learn_m2_x2")
    hidden_bias = controls[2].slider("First hidden bias", -1.0, 1.0, 0.1, 0.1, key="learn_m2_bias")
    values = forward_example(np.array([x1, x2]), hidden_bias)
    net: NeuralNetwork = values["network"]
    first_z, first_a = values["hidden_z"], values["hidden_a"]
    final_z, prediction = values["output_z"], values["prediction"]
    st.code(
        "Layer 1: z = W·x + b\n"
        f"  W = {np.array2string(net.weights[0], precision=2)}  b = {np.array2string(net.biases[0][:, 0], precision=2)}\n"
        f"  z¹ = {np.array2string(first_z[:, 0], precision=5)}\n"
        f"  a¹ = tanh(z¹) = {np.array2string(first_a[:, 0], precision=5)}\n"
        "Output layer: z = W·a¹ + b\n"
        f"  z² = {float(final_z[0, 0]):.6f}\n  prediction = sigmoid(z²) = {float(prediction[0, 0]):.6f}", language="text")
    st.caption("Weights, biases, sums, and activations above are the live values cached by the network's forward pass.")
    _practical_link("Network Playground", "Network Playground", "learn_link_network_2")
    _quiz(MODULE_IDS[1], "What becomes the input to the next layer?",
          ("The current layer's activation", "The loss value", "The learning rate"),
          "The current layer's activation", "The next layer uses A from the current layer as its previous activation.")


def _render_module_3() -> None:
    _objectives(("Compare common activation shapes and output ranges.", "Relate activation derivatives to gradient flow.",
                 "Recognize softmax as a vector-valued function."))
    st.markdown("#### Concept and formulas")
    descriptions = (
        ("Sigmoid", r"\sigma(x)=\frac{1}{1+e^{-x}}", "(0, 1)", r"\sigma'(x)=\sigma(x)(1-\sigma(x))", "Smooth probability-like output; can saturate at large magnitudes."),
        ("Tanh", r"\tanh(x)", "(-1, 1)", r"1-\tanh^2(x)", "Zero-centered outputs; can saturate for large magnitudes."),
        ("ReLU", r"\max(0,x)", "[0, ∞)", r"1\ (x>0),\ 0\ (x\leq0)", "Simple hidden-layer function; negative inputs have zero gradient."),
        ("Leaky ReLU", r"\max(\alpha x,x)", "(-∞, ∞)", r"1\ (x>0),\ \alpha\ (x\leq0)", "Keeps a small negative-side gradient; alpha controls its slope."),
        ("Softmax", r"\frac{e^{z_i}}{\sum_j e^{z_j}}", "(0, 1), sums to 1", r"J=\operatorname{diag}(p)-pp^T", "Transforms a vector of logits into class probabilities; couples the output units."),
    )
    for name, formula, output_range, derivative, use in descriptions:
        with st.expander(f"{name} · output range {output_range}", expanded=name == "Sigmoid"):
            st.latex(formula)
            st.markdown(f"**Derivative:** {derivative}")
            st.write(use)
    st.markdown("#### Interactive activation curves")
    input_range = st.slider("Plot range", 1.0, 10.0, 5.0, 0.5, key="learn_m3_range")
    alpha = st.slider("Leaky ReLU α", 0.0, 0.3, 0.05, 0.01, key="learn_m3_alpha")
    x_values = np.linspace(-input_range, input_range, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor="#F3F6FA")
    for axis, title in zip(axes, ("Activation outputs", "Elementwise derivatives")):
        _style_axis(axis, title)
    functions = (("Sigmoid", activations.sigmoid, activations.sigmoid_derivative),
                 ("Tanh", activations.tanh, activations.tanh_derivative),
                 ("ReLU", activations.relu, activations.relu_derivative),
                 ("Leaky ReLU", lambda x: activations.leaky_relu(x, alpha),
                  lambda x: activations.leaky_relu_derivative(x, alpha)))
    for label, function, derivative in functions:
        axes[0].plot(x_values, function(x_values), label=label, linewidth=2)
        axes[1].plot(x_values, derivative(x_values), label=label, linewidth=2)
    for axis in axes:
        axis.legend(facecolor="#FFFFFF", edgecolor="#C8D9E6", labelcolor="#2F4156")
        axis.set_xlabel("Input x")
    axes[0].set_ylabel("f(x)")
    axes[1].set_ylabel("f′(x)")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown("#### Softmax example")
    logits = np.array([1.2, 0.3, -0.5])
    probabilities = activations.softmax(logits)
    st.code(f"logits = {logits}\nsoftmax(logits) = {np.array2string(probabilities, precision=5)}\n"
            f"probability sum = {probabilities.sum():.6f}", language="text")
    figure, axis = plt.subplots(figsize=(6.0, 2.8), facecolor="#F3F6FA")
    axis.set_facecolor("#FFFFFF")
    axis.bar(("Class 1", "Class 2", "Class 3"), probabilities, color=(BLUE, CYAN, VIOLET))
    _style_axis(axis, "Softmax probabilities for the example logits")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Probability")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    _practical_link("Activation Functions", "Activation Functions", "learn_link_activation")
    _quiz(MODULE_IDS[2], "Which activation maps a vector of logits to probabilities that sum to one?",
          ("ReLU", "Softmax", "Tanh"), "Softmax", "Softmax normalizes exponentiated logits across a class axis.")


def _render_module_4() -> None:
    _objectives(("Distinguish regression losses from classification losses.", "Calculate a loss from targets and predictions.",
                 "Choose a loss that matches an output configuration."))
    st.markdown("#### Concept and formulas")
    formulas = (
        ("Mean Squared Error · regression", r"L=\frac{1}{N}\sum_i(y_i-\hat y_i)^2", "Squares errors, so larger misses have greater influence."),
        ("Mean Absolute Error · regression", r"L=\frac{1}{N}\sum_i|y_i-\hat y_i|", "Averages absolute errors and is less sensitive to large outliers."),
        ("Binary Cross-Entropy · binary classification", r"L=-\frac{1}{N}\sum_i[y_i\log(p_i)+(1-y_i)\log(1-p_i)]", "Scores binary targets against sigmoid probabilities."),
        ("Categorical Cross-Entropy · multiclass classification", r"L=-\frac{1}{N}\sum_i\sum_c y_{ic}\log(p_{ic})", "Scores one-hot labels against a probability vector, typically from softmax."),
    )
    for title, formula, explanation in formulas:
        st.markdown(f"**{title}**")
        st.latex(formula)
        st.caption(explanation)
    st.markdown("#### Interactive numerical example")
    loss_name = st.selectbox("Loss example", ("MSE", "MAE", "Binary cross-entropy", "Categorical cross-entropy"), key="learn_m4_loss")
    if loss_name in ("MSE", "MAE"):
        cols = st.columns(2)
        target = cols[0].number_input("Regression target", value=1.0, step=0.1, key="learn_m4_reg_target")
        prediction = cols[1].number_input("Regression prediction", value=0.3, step=0.1, key="learn_m4_reg_prediction")
        y, p = np.array([target]), np.array([prediction])
        value = losses.mean_squared_error(y, p) if loss_name == "MSE" else losses.mean_absolute_error(y, p)
    elif loss_name == "Binary cross-entropy":
        cols = st.columns(2)
        target = cols[0].selectbox("Binary target", (0.0, 1.0), key="learn_m4_binary_target")
        prediction = cols[1].slider("Predicted probability", 0.01, 0.99, 0.7, 0.01, key="learn_m4_binary_prediction")
        y, p = np.array([target]), np.array([prediction])
        value = losses.binary_cross_entropy(y, p)
    else:
        cols = st.columns(3)
        class_count = cols[0].selectbox("Class count", (3, 4, 5), key="learn_m4_classes")
        target_class = cols[1].selectbox("Target class", tuple(range(class_count)), key="learn_m4_target_class")
        target_probability = cols[2].slider("Probability on target", 0.1, 0.9, 0.7, 0.05, key="learn_m4_target_probability")
        y = np.zeros((1, class_count)); y[0, target_class] = 1
        p = np.full((1, class_count), (1 - target_probability) / (class_count - 1)); p[0, target_class] = target_probability
        value = losses.categorical_cross_entropy(y, p)
    st.code(f"target = {np.array2string(y, precision=3)}\nprediction = {np.array2string(p, precision=3)}", language="text")
    st.metric("Calculated loss", f"{value:.6f}")
    _practical_link("Loss Functions", "Loss Functions", "learn_link_loss")
    _quiz(MODULE_IDS[3], "Which loss is designed for a one-hot class target and a class-probability vector?",
          ("Mean Absolute Error", "Categorical Cross-Entropy", "Mean Squared Error"),
          "Categorical Cross-Entropy", "Categorical cross-entropy compares one-hot classes with a probability distribution.")


def _render_module_5() -> None:
    _objectives(("Trace the chain rule from output to earlier layers.", "Interpret dL/dZ, dW, and db.",
                 "Distinguish gradient inspection from an optimizer update."))
    st.markdown("#### Concept")
    st.write("Backpropagation applies the chain rule in reverse. The output error is converted into a pre-activation gradient, "
             "then each hidden layer receives the downstream signal through the next layer's weights and its own activation derivative. "
             "Weight gradients combine dL/dZ with the previous activation; bias gradients sum dL/dZ across samples.")
    st.markdown("#### Mathematical formulas")
    st.latex(r"\frac{\partial L}{\partial A^{[l]}}=(W^{[l+1]})^T\frac{\partial L}{\partial Z^{[l+1]}}")
    st.latex(r"\frac{\partial L}{\partial Z^{[l]}}=\frac{\partial L}{\partial A^{[l]}}\odot f'(Z^{[l]})")
    st.latex(r"dW^{[l]}=dZ^{[l]}(A^{[l-1]})^T,\qquad db^{[l]}=\sum_{samples}dZ^{[l]}")
    st.markdown("#### Worked example · actual NumPy gradients")
    cols = st.columns(3)
    x1 = cols[0].number_input("Backprop x₁", value=0.2, step=0.1, key="learn_m5_x1")
    x2 = cols[1].number_input("Backprop x₂", value=-0.4, step=0.1, key="learn_m5_x2")
    target = cols[2].selectbox("Target", (0.0, 1.0), key="learn_m5_target")
    net = NeuralNetwork(2, [2], 1, random_seed=5, hidden_activation="tanh")
    prediction = net.forward(np.array([x1, x2]))
    gradient_data = net.backward(np.array([target]), loss="binary_cross_entropy")
    st.write(f"Prediction **{prediction[0, 0]:.6f}** · BCE loss **{gradient_data['loss']:.6f}**")
    for layer_index, d_z in enumerate(gradient_data["preactivation_gradients"], start=1):
        with st.expander(f"Layer {layer_index} gradient values", expanded=True):
            st.write("dL/dA", gradient_data["activation_gradients"][layer_index - 1])
            st.write("f′(Z)", gradient_data["activation_derivatives"][layer_index - 1])
            st.write("dL/dZ", d_z)
            st.write("dW", gradient_data["weight_gradients"][layer_index - 1])
            st.write("db", gradient_data["bias_gradients"][layer_index - 1])
    st.caption("Gradients above come from NeuralNetwork.backward(). Inspecting them does not change weights or biases.")
    _practical_link("Backpropagation Explorer", "Backpropagation Explorer", "learn_link_backprop")
    _quiz(MODULE_IDS[4], "Why does backpropagation use the weights of the next layer?",
          ("To propagate the downstream gradient to the preceding activations", "To shuffle the training data", "To calculate the output probabilities"),
          "To propagate the downstream gradient to the preceding activations",
          "Multiplication by the next layer's transposed weights maps the gradient back to the preceding layer.")


def _render_module_6() -> None:
    _objectives(("Explain gradient descent and learning rate.", "Compare SGD, Momentum, and Adam updates.",
                 "Recognize optimizer state in Momentum and Adam."))
    st.markdown("#### Concept and update equations")
    st.write("An optimizer changes parameters in a direction intended to reduce loss. The learning rate sets update size. "
             "SGD follows the current gradient, Momentum accumulates a velocity, and Adam adapts each parameter's step using first and second moments.")
    st.latex(r"\theta_{t+1}=\theta_t-\eta\nabla_\theta L")
    st.latex(r"v_t=\beta v_{t-1}+(1-\beta)g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t")
    st.latex(r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2")
    st.latex(r"\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\quad "
             r"\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}")
    st.markdown("#### Interactive one-step update")
    cols = st.columns(4)
    parameter_value = cols[0].number_input("Starting parameter θ", value=1.0, step=0.1, key="learn_m6_parameter")
    gradient_value = cols[1].number_input("Gradient g", value=0.4, step=0.1, key="learn_m6_gradient")
    learning_rate = cols[2].slider("Learning rate η", 0.01, 0.5, 0.1, 0.01, key="learn_m6_lr")
    selected = cols[3].selectbox("Optimizer", ("SGD", "Momentum", "Adam"), key="learn_m6_optimizer")
    parameter = np.array([parameter_value], dtype=float)
    gradient = np.array([gradient_value], dtype=float)
    if selected == "SGD":
        optimizer = SGD(learning_rate)
    elif selected == "Momentum":
        momentum = st.slider("Momentum coefficient β", 0.0, 0.99, 0.9, 0.01, key="learn_m6_momentum")
        optimizer = SGDMomentum(learning_rate, momentum=momentum)
    else:
        beta1 = st.slider("Adam β₁", 0.0, 0.99, 0.9, 0.01, key="learn_m6_beta1")
        beta2 = st.slider("Adam β₂", 0.0, 0.999, 0.999, 0.001, key="learn_m6_beta2")
        optimizer = Adam(learning_rate, beta1=beta1, beta2=beta2)
    before = float(parameter[0])
    optimizer.step([parameter], [gradient])
    st.code(f"θ before = {before:.6f}\ngradient = {gradient_value:.6f}\n"
            f"θ after one {selected} update = {parameter[0]:.6f}", language="text")
    st.caption("This is an actual one-step update using NeuroLab's optimizer class. Momentum and Adam start with fresh state in this example.")
    _practical_link("Optimizer Lab", "Optimizer Lab", "learn_link_optimizer")
    _quiz(MODULE_IDS[5], "What does a larger learning rate usually change?",
          ("The size of parameter updates", "The number of output classes", "The activation function"),
          "The size of parameter updates", "The learning rate scales how far an optimizer moves along its update direction.")


def _render_module_7() -> None:
    _objectives(("Describe underfitting and overfitting.", "Explain why validation data estimates generalization.",
                 "Recognize regularization, dropout, and early stopping."))
    st.markdown("#### Concept")
    st.write("Underfitting occurs when a model cannot capture patterns in its training data. Overfitting occurs when it fits training details "
             "that do not generalize, often shown by improving training performance while validation performance stalls or worsens. "
             "Validation data estimates performance on examples not used for parameter updates.")
    st.markdown("#### Regularization examples")
    st.latex(r"L_{L1}=L+\lambda\sum_j|w_j|,\qquad L_{L2}=L+\lambda\sum_jw_j^2")
    st.write("L1 encourages sparse weights; L2 penalizes large weights smoothly. Dropout randomly masks units during training. "
             "Early stopping ends training when validation performance stops improving. These are general techniques; this small NeuroLab trainer does not currently apply dropout or L1/L2 penalties automatically.")
    st.markdown("#### Interactive regularization penalty")
    cols = st.columns(2)
    weights = np.array([
        cols[0].number_input("Weight w₁", value=0.8, step=0.1, key="learn_m7_w1"),
        cols[0].number_input("Weight w₂", value=-0.3, step=0.1, key="learn_m7_w2"),
        cols[0].number_input("Weight w₃", value=0.1, step=0.1, key="learn_m7_w3"),
    ])
    strength = cols[1].slider("Regularization strength λ", 0.0, 1.0, 0.2, 0.01, key="learn_m7_lambda")
    st.code(f"weights = {weights}\nL1 penalty = λ·Σ|w| = {strength * np.abs(weights).sum():.6f}\n"
            f"L2 penalty = λ·Σw² = {strength * np.square(weights).sum():.6f}", language="text")

    database = default_database_path()
    if database.exists():
        records = ExperimentStore(database).list(sort_by="Date", descending=True)
    else:
        records = []
    st.markdown("#### Saved training and validation history")
    if records:
        labels = {f"#{record['id']} · {record['name']} · {record['dataset']}": record for record in records}
        chosen = st.selectbox("Genuine saved experiment", tuple(labels), key="learn_m7_saved_experiment")
        record = labels[chosen]
        history = record["history"]
        figure, axis = plt.subplots(figsize=(8.0, 3.8), facecolor="#F3F6FA")
        _style_axis(axis, "Stored loss history · actual experiment")
        axis.plot(history["epoch"], history["training_loss"], color=BLUE, label="Training loss", linewidth=2.1)
        if "validation_loss" in history:
            axis.plot(history["epoch"], history["validation_loss"], color=CYAN, label="Validation loss", linewidth=2.1)
        axis.set_xlabel("Epoch"); axis.set_ylabel("Loss")
        axis.legend(facecolor="#FFFFFF", edgecolor="#C8D9E6", labelcolor="#2F4156")
        figure.tight_layout(); st.pyplot(figure, width="stretch"); plt.close(figure)
    else:
        st.info("No saved experiment history is available yet, so no training curves are shown.")
    _practical_link("Experiments & Reports", "Experiments & Reports", "learn_link_reports")
    _quiz(MODULE_IDS[6], "What is the validation set primarily used to estimate?",
          ("Generalization to held-out examples", "The training batch size", "The number of hidden neurons"),
          "Generalization to held-out examples", "Validation examples are held out from parameter updates and help monitor generalization.")


def _render_module_8() -> None:
    _objectives(("Interpret accuracy, precision, recall, and F1.", "Read a binary confusion matrix.",
                 "Distinguish training metrics from validation metrics."))
    st.markdown("#### Concept and formulas")
    st.write("Accuracy measures all correct classifications. Precision asks how many predicted positives were correct; recall asks how many actual positives were found. "
             "F1 is the harmonic mean of precision and recall. A confusion matrix shows true negatives, false positives, false negatives, and true positives.")
    st.latex(r"Accuracy=\frac{TP+TN}{TP+TN+FP+FN},\quad Precision=\frac{TP}{TP+FP}")
    st.latex(r"Recall=\frac{TP}{TP+FN},\qquad F1=2\frac{Precision\cdot Recall}{Precision+Recall}")
    st.markdown("#### Interactive, illustrative classification example")
    st.caption("The probabilities below are a small teaching example, not the output of a trained experiment.")
    y_true = np.array([1, 0, 1, 1, 0, 0])
    probabilities = np.array([0.90, 0.70, 0.40, 0.80, 0.20, 0.30])
    threshold = st.slider("Positive-class threshold", 0.1, 0.9, 0.5, 0.05, key="learn_m8_threshold")
    y_predicted = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(y_true, y_predicted)
    cols = st.columns(4)
    for column, metric, label in zip(cols, ("accuracy", "precision", "recall", "f1"), ("Accuracy", "Precision", "Recall", "F1-score")):
        column.metric(label, f"{metrics[metric]:.1%}")
    st.markdown("Confusion matrix rows are actual classes (0, 1); columns are predicted classes (0, 1).")
    st.table([{"Actual \\ Predicted": "Class 0", "Class 0": int(metrics["confusion_matrix"][0, 0]),
               "Class 1": int(metrics["confusion_matrix"][0, 1])},
              {"Actual \\ Predicted": "Class 1", "Class 0": int(metrics["confusion_matrix"][1, 0]),
               "Class 1": int(metrics["confusion_matrix"][1, 1])}])
    st.code(f"true labels      = {y_true}\nprobabilities    = {probabilities}\n"
            f"threshold        = {threshold:.2f}\npredicted labels = {y_predicted}", language="text")
    st.write("Training metrics are measured on examples used for parameter updates. Validation metrics use held-out examples and are a better signal of generalization.")
    _practical_link("Network Playground", "Network Playground", "learn_link_network_8")
    _quiz(MODULE_IDS[7], "If false negatives are costly, which metric is especially useful to monitor?",
          ("Recall", "Training duration", "Number of weights"), "Recall",
          "Recall measures the fraction of actual positives that the model correctly identifies.")


RENDERERS = (_render_module_1, _render_module_2, _render_module_3, _render_module_4,
             _render_module_5, _render_module_6, _render_module_7, _render_module_8)


def render_learn_dnn() -> None:
    """Render module selector, progress, lessons, quizzes, and practical links."""
    with st.container(key="learn_dnn_page"):
        _render_learning_center()


def _render_learning_center() -> None:
    st.markdown(
        """<section class="hero playground-hero learn-hero">
            <div class="hero-copy">
                <div class="hero-kicker">PRINCIPLES, EXPLAINED STEP BY STEP</div>
                <h1>Learn DNN<span class="hero-period">.</span></h1>
                <div class="hero-tagline">Build the intuition behind neural networks.</div>
                <p class="hero-description">Explore core concepts through concise lessons, live calculations, and knowledge checks.</p>
                <div class="hero-chip"><span class="status-dot"></span>Eight guided modules · Interactive examples</div>
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
    with st.container(border=True, key="learn_dnn_controls_panel"):
        selected = st.selectbox("Learning module", MODULES, key="learn_dnn_module")
        module_index = MODULES.index(selected)
        completed = set(st.session_state.get(COMPLETED_KEY, []))
        progress_col, reset_col = st.columns([4, 1], vertical_alignment="center")
        with progress_col:
            st.progress(len(completed) / len(MODULES), text=f"{len(completed)} of {len(MODULES)} modules completed")
        with reset_col:
            if st.button("Reset progress", key="learn_dnn_reset", width="stretch"):
                st.session_state[COMPLETED_KEY] = []
                for module_id in MODULE_IDS:
                    st.session_state.pop(f"{FEEDBACK_PREFIX}{module_id}", None)
                st.rerun()

    with st.container(border=True, key="learn_dnn_lesson_panel"):
        st.markdown(f'<div class="learn-module-title"><span>MODULE {module_index + 1:02d}</span><h2>{selected}</h2></div>', unsafe_allow_html=True)
        RENDERERS[module_index]()
