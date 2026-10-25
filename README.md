# NeuroLab

**Build. Train. Visualize. Understand.**

NeuroLab is an interactive learning workspace for Principles of Deep Neural Networks. It combines small NumPy implementations of neural network mathematics with Streamlit pages that let students inspect datasets, model layers, gradients, optimizer updates, and evaluation results.

## Features

- Network Playground for synthetic binary classification, architecture setup, training, and decision boundaries.
- Dataset Explorer for XOR, moons, circles, and spiral data, including sample statistics and leakage-safe splitting.
- Activation and Loss Functions pages with interactive curves, equations, and numeric examples.
- NumPy feedforward network with forward propagation, backpropagation, and SGD, Momentum, and Adam updates.
- Optimizer Lab for reproducible comparisons from equivalent initial parameters.
- Backpropagation Explorer, Network Visualizer, and Math Calculator for layer values and gradients.
- Experiment Wizard, evaluation tools, and local SQLite experiment reports.
- Safe `.nlmodel` export/import using JSON metadata and NumPy arrays; imported files do not execute Python code.
- Quiz & Challenge Center with practice, timed, daily, numerical, and mistake-review modes.

## Technology

Python, NumPy, Streamlit, Matplotlib, scikit-learn, and pytest.

## Project layout

```text
app.py                         Streamlit entry point
src/neurolab/                  Pages, datasets, training, persistence, visualizations
src/neurolab/models/           NumPy network, activations, losses, optimizers
tests/                         Unit and Streamlit application tests
examples/                      Forward-pass and XOR training examples
data/experiments.sqlite3       Local experiment history (created at runtime, ignored by Git)
requirements.txt               Python dependencies
```

## Setup and launch

From the project root, create and activate a virtual environment, install dependencies, then launch Streamlit:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` after creating the environment.

## Deploy on Vercel

The repository includes `Dockerfile.vercel`, which packages the Streamlit server as a Vercel container function. Push the project to a Git provider connected to Vercel, import the repository, and deploy it. Vercel detects the Dockerfile automatically; no frontend build command or output directory is needed. The container listens on Vercel's `PORT` (defaulting to port 80).

You can also deploy from the project root with the Vercel CLI:

```bash
npm install --global vercel
vercel login
vercel
```

Use the preview URL to check the app before deploying to production with `vercel --prod`.

The runtime dependency list is in `requirements-vercel.txt`; `pytest` remains a development dependency and is not installed in the deployed image. Experiment records use SQLite in `/tmp` on Vercel, so they are temporary and may disappear when an instance is replaced or scaled down. For durable shared experiment history, configure a hosted database in a later deployment step. Session state and loaded models are also per browser session.

## Tests

With the environment active:

```bash
pytest -q
```

Or run without activation from the project root:

```bash
.venv/bin/python -m pytest -q
```

## Neural network mathematics

For each layer, NeuroLab computes the weighted sum and bias, then applies an activation:

```text
Z[l] = W[l] @ A[l-1] + b[l]
A[l] = activation(Z[l])
```

The current feedforward model uses a configurable hidden activation and a sigmoid output layer. Backpropagation applies the chain rule to calculate parameter gradients; the loss derivative's averaging convention is already included, so the batch gradient is not divided twice. Optimizers then update the model parameters. MSE and MAE are available as regression losses; binary cross-entropy is the supported classification training loss for the current sigmoid-output engine.

## Example workflow

1. Open **Experiment Wizard** or **Network Playground** and choose a synthetic dataset, architecture, activation, optimizer, and training settings.
2. Start training explicitly. Inspect the recorded train/validation history, accuracy where applicable, and the learned boundary.
3. Use **Model Evaluation Lab** to inspect generalization metrics, then save a completed run for **Experiments & Reports**.
4. Open **Model Manager** to export a trained model. Import a `.nlmodel` file and confirm loading before generating predictions.

The command-line XOR example can be run with:

```bash
python -m examples.train_xor
```

## Screenshots

For GitHub, capture the app with Streamlit running and save selected images under a tracked `docs/images/` folder. Link them with relative Markdown paths, for example `![Network Playground](docs/images/network-playground.png)`. Do not include private experiment data or local database files in screenshots.

## Known limitations

- The model engine's output layer is sigmoid. It does not implement a softmax output head, so categorical cross-entropy is a calculator/loss utility only and is not a supported training configuration.
- Multiple sigmoid outputs are independent values rather than a normalized multiclass probability distribution.
- Quiz progress and live models held in Streamlit session state are temporary and are not restored after the session ends. Experiment metadata and histories are stored locally in SQLite; model parameter files must be downloaded separately.
- The timed quiz checks its deadline on page updates or when **Refresh timer** is pressed; it does not run a background clock.
- The synthetic dataset tools are designed for small educational demonstrations, not production-scale model development.
