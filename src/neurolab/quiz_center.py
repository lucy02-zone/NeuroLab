"""Question bank, scoring helpers, and interactive quiz center for NeuroLab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
from typing import Any, Iterable

import numpy as np
import streamlit as st

from .learn_dnn import classification_metrics
from .models import activations, losses


CATEGORIES = (
    "Neural network fundamentals",
    "Forward propagation",
    "Activation functions",
    "Loss functions",
    "Backpropagation and the chain rule",
    "Gradient descent and optimizers",
    "Overfitting, regularization, and generalization",
    "Model evaluation metrics",
)
DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")
MODES = ("Practice Mode", "Timed Quiz", "Daily Challenge", "Review Mistakes")


@dataclass(frozen=True)
class Question:
    """A multiple-choice or numeric quiz question with feedback metadata."""

    question_id: str
    category: str
    difficulty: str
    prompt: str
    explanation: str
    options: tuple[str, ...] = ()
    correct_index: int | None = None
    numeric_answer: float | None = None
    tolerance: float = 1e-3
    formula: str = ""

    @property
    def is_numeric(self) -> bool:
        return self.numeric_answer is not None


def _question_bank() -> tuple[Question, ...]:
    """Return curated questions; numerical answers use real project utilities."""
    return (
        Question("fund-b-1", CATEGORIES[0], "Beginner", "What does a neuron's bias do?",
                 "The bias shifts the weighted sum before activation, changing the neuron's threshold.",
                 ("Shifts the activation threshold", "Normalizes every feature", "Chooses the optimizer", "Measures accuracy"), 0,
                 formula="z = Σ(wᵢxᵢ) + b"),
        Question("fund-i-1", CATEGORIES[0], "Intermediate", "What distinguishes a deep network from a shallow one?",
                 "A deep network composes multiple hidden layers of learned representations.",
                 ("It must use sigmoid", "It has multiple hidden layers", "It always has more output classes", "It requires a GPU"), 1),
        Question("fund-a-1", CATEGORIES[0], "Advanced", "For a layer with nᵢ inputs and nₒ neurons, what is the weight matrix shape using W @ A?",
                 "Each output neuron has one row of weights, so the matrix is (nₒ, nᵢ).",
                 ("(nᵢ, nₒ)", "(nₒ, nᵢ)", "(nᵢ+nₒ, 1)", "(nₒ, nₒ)"), 1),
        Question("forward-b-1", CATEGORIES[1], "Beginner", "Given x=[2, 1], w=[3, -1], and b=0.5, what is z?",
                 "The weighted sum is 3·2 + (−1)·1 + 0.5 = 5.5.",
                 ("2.5", "4.5", "5.5", "6.5"), 2, formula="z = wᵀx + b"),
        Question("forward-i-1", CATEGORIES[1], "Intermediate", "A layer has W=[[1,2],[-1,3]], A=[[2],[1]], b=[[0.5],[-0.5]]. What is the first component of Z?",
                 "The first row gives 1·2 + 2·1 + 0.5 = 4.5.", ("3.5", "4.0", "4.5", "5.5"), 2,
                 formula="Z = W @ A + b"),
        Question("forward-a-1", CATEGORIES[1], "Advanced", "Calculate the weighted sum for x=[0.2,0.8], w=[-0.5,1.25], and b=-0.1.",
                 "Compute (−0.5)(0.2) + (1.25)(0.8) − 0.1 = 0.8.", numeric_answer=float(np.dot([-0.5, 1.25], [0.2, 0.8]) - 0.1), formula="z = wᵀx + b"),
        Question("act-b-1", CATEGORIES[2], "Beginner", "What is ReLU(−3)?",
                 "ReLU returns max(0,x), so negative inputs map to zero.", ("−3", "0", "1", "−1"), 1,
                 formula="ReLU(x) = max(0,x)"),
        Question("act-i-1", CATEGORIES[2], "Intermediate", "What is sigmoid(0)?",
                 "1/(1+e⁰) = 1/2.", ("0", "0.5", "1", "−0.5"), 1,
                 formula="σ(x)=1/(1+e⁻ˣ)"),
        Question("act-a-1", CATEGORIES[2], "Advanced", "Evaluate tanh(0.4) to three decimal places.",
                 f"The NumPy tanh implementation gives {np.tanh(0.4):.6f}.", numeric_answer=float(activations.tanh(0.4)), tolerance=5e-4,
                 formula="tanh(0.4)"),
        Question("loss-b-1", CATEGORIES[3], "Beginner", "Which loss is commonly used for regression and squares prediction errors?",
                 "MSE averages squared errors, giving larger errors more influence.", ("Binary cross-entropy", "Mean squared error", "Categorical cross-entropy", "Accuracy"), 1),
        Question("loss-i-1", CATEGORIES[3], "Intermediate", "For target 1 and prediction 0.8, what is binary cross-entropy (approximately)?",
                 "For y=1, BCE is −log(p); −log(0.8) is approximately 0.2231.", numeric_answer=float(losses.binary_cross_entropy(np.array([1.0]), np.array([0.8]))), tolerance=1e-3,
                 formula="−[y log(p)+(1−y)log(1−p)]"),
        Question("loss-a-1", CATEGORIES[3], "Advanced", "For targets [1,0] and predictions [0.8,0.25], what is mean squared error?",
                 "Squared errors are 0.04 and 0.0625; their mean is 0.05125.", numeric_answer=losses.mean_squared_error(np.array([1., 0.]), np.array([.8, .25])), tolerance=1e-5,
                 formula="mean((y−p)²)"),
        Question("back-b-1", CATEGORIES[4], "Beginner", "What does backpropagation use to compute parameter gradients?",
                 "It applies the chain rule from the loss backward through the network.",
                 ("The chain rule", "The confusion matrix", "Only the forward pass", "A random search"), 0),
        Question("back-i-1", CATEGORIES[4], "Intermediate", "For L=(w−y)²/2 with w=2 and y=0.5, what is dL/dw?",
                 "Differentiating gives w−y = 1.5.", numeric_answer=1.5, formula="dL/dw = w−y"),
        Question("back-a-1", CATEGORIES[4], "Advanced", "For mean MSE over two values, targets [1,0] and predictions [0.8,0.2], what is the derivative for the first prediction?",
                 "The project averages over two elements: 2(0.8−1)/2 = −0.2.", numeric_answer=float(losses.mean_squared_error_derivative(np.array([1., 0.]), np.array([.8, .2]))[0]),
                 tolerance=1e-6, formula="dL/dpᵢ = 2(pᵢ−yᵢ)/n"),
        Question("opt-b-1", CATEGORIES[5], "Beginner", "With gradient descent, positive gradient and positive learning rate, which way does a parameter move?",
                 "The update subtracts learning_rate × gradient, so a positive gradient decreases the parameter.",
                 ("It increases", "It decreases", "It stays fixed", "It becomes the gradient"), 1,
                 formula="θ ← θ − η∇L"),
        Question("opt-i-1", CATEGORIES[5], "Intermediate", "A weight is 1.0, its gradient is 0.4, and learning rate is 0.1. What is the SGD-updated weight?",
                 "1.0 − 0.1·0.4 = 0.96.", numeric_answer=1.0 - 0.1 * 0.4, formula="w ← w − ηg"),
        Question("opt-a-1", CATEGORIES[5], "Advanced", "What does Adam combine?",
                 "Adam tracks first and second gradient moments and applies bias correction.",
                 ("Only a fixed decay schedule", "Momentum-like first moments and squared-gradient second moments", "A train/test split and dropout", "Only the sign of the gradient"), 1),
        Question("gen-b-1", CATEGORIES[6], "Beginner", "Which pattern can indicate overfitting?",
                 "Training performance improves while validation performance worsens or stalls.",
                 ("Both train and validation losses fall", "Training loss falls while validation loss rises", "Both datasets have the same size", "The model has one input"), 1),
        Question("gen-i-1", CATEGORIES[6], "Intermediate", "What does L2 regularization encourage?",
                 "The squared-weight penalty discourages excessively large parameter values.",
                 ("Larger weights", "Smaller weight magnitudes", "More training samples", "Perfect training accuracy"), 1),
        Question("gen-a-1", CATEGORIES[6], "Advanced", "Why should preprocessing be fitted only on training data?",
                 "Fitting transforms on validation data leaks information about that held-out set into training.",
                 ("To prevent validation information leakage", "To make every feature binary", "To guarantee lower training loss", "To avoid shuffling"), 0),
        Question("eval-b-1", CATEGORIES[7], "Beginner", "Accuracy is the fraction of what?",
                 "Accuracy is the number of correct predictions divided by all predictions.",
                 ("Correct predictions among all predictions", "True positives among predicted positives", "True positives among actual positives", "Predicted positives among actual positives"), 0),
        Question("eval-i-1", CATEGORIES[7], "Intermediate", "There are 8 correct predictions out of 10. What is accuracy?",
                 "Accuracy is 8/10 = 0.8 (80%).", numeric_answer=float(classification_metrics(np.array([1]*10), np.array([1]*8+[0]*2))["accuracy"]), formula="(TP+TN)/N"),
        Question("eval-a-precision", CATEGORIES[7], "Advanced", "For true labels [1,1,0,0,1] and predicted labels [1,0,1,0,1], what is precision?",
                 "There are 2 true positives and 1 false positive, so precision is 2/(2+1) = 0.667.",
                 numeric_answer=float(classification_metrics(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1]))["precision"]),
                 tolerance=1e-3, formula="Precision = TP/(TP+FP)"),
        Question("eval-a-recall", CATEGORIES[7], "Advanced", "For true labels [1,1,0,0,1] and predicted labels [1,0,1,0,1], what is recall?",
                 "There are 2 true positives and 1 false negative, so recall is 2/(2+1) = 0.667.",
                 numeric_answer=float(classification_metrics(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1]))["recall"]),
                 tolerance=1e-3, formula="Recall = TP/(TP+FN)"),
        Question("eval-a-1", CATEGORIES[7], "Advanced", "If precision=0.75 and recall=0.60, what is F1 (approximately)?",
                 "F1 = 2PR/(P+R) = 2·0.75·0.60/1.35 ≈ 0.667.", numeric_answer=2*.75*.60/(.75+.60), tolerance=1e-3, formula="F1=2PR/(P+R)"),
    )


QUESTION_BANK = _question_bank()


def validate_question_bank(questions: Iterable[Question] = QUESTION_BANK) -> None:
    """Raise a useful error if question IDs, labels, answers, or explanations are invalid."""
    seen: set[str] = set()
    for question in questions:
        if not question.question_id or question.question_id in seen:
            raise ValueError("Question IDs must be non-empty and unique.")
        seen.add(question.question_id)
        if question.category not in CATEGORIES or question.difficulty not in DIFFICULTIES:
            raise ValueError(f"Question {question.question_id} has an unknown category or difficulty.")
        if not question.prompt.strip() or not question.explanation.strip():
            raise ValueError(f"Question {question.question_id} needs prompt and explanation text.")
        if question.is_numeric:
            if question.options or question.correct_index is not None or not np.isfinite(question.numeric_answer):
                raise ValueError(f"Numeric question {question.question_id} has invalid answer fields.")
            if question.tolerance < 0 or not np.isfinite(question.tolerance):
                raise ValueError(f"Question {question.question_id} has an invalid numeric tolerance.")
        elif len(question.options) != 4 or question.correct_index not in range(4):
            raise ValueError(f"Multiple-choice question {question.question_id} must have four options and a valid answer index.")


def filter_questions(category: str, difficulty: str, questions: Iterable[Question] = QUESTION_BANK) -> list[Question]:
    """Filter the bank by category and difficulty; use All for either filter."""
    if category != "All categories" and category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    if difficulty != "All levels" and difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return [q for q in questions if (category == "All categories" or q.category == category)
            and (difficulty == "All levels" or q.difficulty == difficulty)]


def select_daily_questions(challenge_date: date, count: int = 5, questions: Iterable[Question] = QUESTION_BANK) -> list[Question]:
    """Select a stable daily set by hashing the chosen local calendar date."""
    if count < 1:
        raise ValueError("Daily challenge question count must be at least one.")
    items = list(questions)
    if not items:
        return []
    seed = int(hashlib.sha256(challenge_date.isoformat().encode("ascii")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(items))[: min(count, len(items))]
    return [items[int(index)] for index in indexes]


def validate_answer(question: Question, answer: int | str | float) -> bool:
    """Check a selected option index or numeric answer using per-question tolerance."""
    if question.is_numeric:
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and abs(value - float(question.numeric_answer)) <= question.tolerance)
    try:
        return int(answer) == question.correct_index
    except (TypeError, ValueError):
        return False


def calculate_progress(answers: dict[str, dict[str, Any]], questions: Iterable[Question] = QUESTION_BANK) -> dict[str, Any]:
    """Summarize actual submitted responses, including accuracy by category and level."""
    bank = {q.question_id: q for q in questions}
    valid = [(bank[key], value) for key, value in answers.items() if key in bank]
    attempted = len(valid)
    correct = sum(bool(value.get("correct")) for _, value in valid)
    by_category: dict[str, dict[str, int]] = {name: {"attempted": 0, "correct": 0} for name in CATEGORIES}
    by_difficulty: dict[str, dict[str, int]] = {name: {"attempted": 0, "correct": 0} for name in DIFFICULTIES}
    for question, answer in valid:
        for bucket in (by_category[question.category], by_difficulty[question.difficulty]):
            bucket["attempted"] += 1
            bucket["correct"] += int(bool(answer.get("correct")))
    return {"attempted": attempted, "correct": correct,
            "accuracy": (100.0 * correct / attempted) if attempted else 0.0,
            "by_category": by_category, "by_difficulty": by_difficulty,
            "incorrect_ids": [q.question_id for q, value in valid if not value.get("correct")]}


def timed_quiz_expired(started_at: datetime, duration_seconds: int, now: datetime | None = None) -> bool:
    """Return whether the quiz deadline has passed, using timestamps kept in session state."""
    if duration_seconds <= 0:
        raise ValueError("Timed quiz duration must be positive.")
    current = now or datetime.now(started_at.tzinfo)
    return current >= started_at + timedelta(seconds=duration_seconds)


def _submit(question: Question, answer: int | float, mode: str, attempt_key: str) -> None:
    result = validate_answer(question, answer)
    response = {"correct": result, "answer": answer, "mode": mode, "submitted_at": datetime.now().isoformat(timespec="seconds")}
    st.session_state.setdefault("quiz_center_answers", {})[question.question_id] = response
    st.session_state.setdefault("quiz_center_history", []).append({"question_id": question.question_id, **response})
    st.session_state[attempt_key] = {"correct": result, "answer": answer}


def render_quiz_center() -> None:
    """Render practice, timed, daily, and mistake-review learning modes."""
    validate_question_bank()
    st.markdown('<div class="eyebrow">TEST YOUR DNN INTUITION</div>', unsafe_allow_html=True)
    st.title("Quiz & Challenge Center")
    st.write("Practice the ideas behind neural networks, check your calculations, and review what you missed.")
    answers: dict[str, dict[str, Any]] = st.session_state.setdefault("quiz_center_answers", {})
    history: list[dict[str, Any]] = st.session_state.setdefault("quiz_center_history", [])
    progress = calculate_progress(answers)
    top = st.columns(3)
    top[0].metric("Questions attempted", progress["attempted"])
    top[1].metric("Correct answers", progress["correct"])
    top[2].metric("Session accuracy", f"{progress['accuracy']:.1f}%")
    st.caption("Progress and completion history are stored in this browser session only; they reset when the session ends.")
    with st.expander("Progress by topic and difficulty"):
        category_rows = [{"Category": name, "Attempted": values["attempted"], "Correct": values["correct"]}
                         for name, values in progress["by_category"].items() if values["attempted"]]
        difficulty_rows = [{"Difficulty": name, "Attempted": values["attempted"], "Correct": values["correct"]}
                           for name, values in progress["by_difficulty"].items() if values["attempted"]]
        if category_rows:
            st.dataframe(category_rows, hide_index=True, width="stretch")
            st.dataframe(difficulty_rows, hide_index=True, width="stretch")
        else:
            st.caption("Topic and difficulty breakdowns appear after your first submitted answer.")
        completions = st.session_state.get("quiz_completion_history", [])
        if completions:
            st.markdown("**Completed challenges this session**")
            st.dataframe(completions, hide_index=True, width="stretch")

    mode = st.selectbox("Challenge mode", MODES, key="quiz_center_mode")
    if mode == "Review Mistakes":
        mistake_ids = progress["incorrect_ids"]
        st.subheader("Review mistakes")
        if not mistake_ids:
            st.info("No incorrect answers to review yet. Try a question in Practice Mode or a challenge.")
            return
        bank = {q.question_id: q for q in QUESTION_BANK}
        for question_id in mistake_ids:
            question = bank[question_id]
            response = answers[question_id]
            with st.expander(f"{question.category} · {question.difficulty}: {question.prompt}"):
                if question.is_numeric:
                    st.write(f"Your answer: `{response['answer']}` · Correct answer: `{question.numeric_answer:.6g}`")
                else:
                    st.write(f"Your answer: {question.options[int(response['answer'])]} · Correct answer: {question.options[question.correct_index]}")
                st.info(question.explanation)
        return

    if mode == "Daily Challenge":
        challenge_date = st.date_input("Challenge date", value=date.today(), key="quiz_daily_date")
        selected_questions = select_daily_questions(challenge_date, 5)
        st.caption(f"Five questions selected deterministically for {challenge_date.isoformat()}.")
        quiz_key = f"daily:{challenge_date.isoformat()}"
        _render_question_sequence(selected_questions, mode, quiz_key)
    else:
        category_col, difficulty_col = st.columns(2)
        category = category_col.selectbox("Category", ("All categories", *CATEGORIES), key="quiz_category")
        difficulty = difficulty_col.selectbox("Difficulty", ("All levels", *DIFFICULTIES), key="quiz_difficulty")
        candidates = filter_questions(category, difficulty)
        if mode == "Timed Quiz":
            duration_minutes = st.select_slider("Time limit", options=(1, 2, 3, 5, 10), value=3, key="quiz_duration")
            if st.button("Start / Restart timed quiz", key="quiz_start_timed"):
                st.session_state["quiz_timed_started_at"] = datetime.now().isoformat()
                st.session_state["quiz_timed_deadline"] = datetime.now().timestamp() + int(duration_minutes) * 60
                st.session_state["quiz_timed_question_ids"] = [q.question_id for q in candidates]
                st.session_state["quiz_timed_responses"] = {}
                st.session_state["quiz_sequence_timed"] = 0
                st.session_state["quiz_completed_timed"] = False
                st.session_state["quiz_completion_saved_timed"] = False
            deadline = st.session_state.get("quiz_timed_deadline")
            if deadline is None:
                st.info("Choose a topic and difficulty, then start the quiz when you are ready.")
                return
            remaining = max(0, int(deadline - datetime.now().timestamp()))
            st.progress(remaining / (duration_minutes * 60), text=f"Time remaining: {remaining // 60:02d}:{remaining % 60:02d}")
            st.caption("The timer refreshes when the page updates; use Refresh timer to check the deadline without submitting an answer.")
            st.button("Refresh timer", key="quiz_refresh_timer")
            if remaining <= 0:
                st.warning("Time is up. Your submitted answers and score are kept below.")
                by_id = {q.question_id: q for q in QUESTION_BANK}
                timed_questions = [by_id[qid] for qid in st.session_state.get("quiz_timed_question_ids", []) if qid in by_id]
                _render_final_summary(st.session_state.get("quiz_timed_responses", {}), timed_questions, "timed")
                return
            _render_question_sequence(timed_questions, mode, "timed", timed=True)
        else:
            _render_question_sequence(candidates, mode, "practice")


def _render_question_sequence(questions: list[Question], mode: str, attempt_key: str, timed: bool = False) -> None:
    if not questions:
        st.warning("No questions match that selection.")
        return
    sequence_key = f"quiz_sequence_{attempt_key}"
    if st.session_state.get(sequence_key) not in range(len(questions)):
        st.session_state[sequence_key] = 0
    index = st.session_state[sequence_key]
    question = questions[index]
    st.progress((index + 1) / len(questions), text=f"Question {index + 1} of {len(questions)} · {question.difficulty}")
    st.markdown(f"**{question.category}**")
    st.subheader(question.prompt)
    if question.formula:
        st.latex(question.formula)
    widget_key = f"quiz_answer_{attempt_key}_{question.question_id}"
    if question.is_numeric:
        answer: int | float = st.number_input("Your numerical answer", value=0.0, format="%.6f", key=widget_key)
    else:
        choice = st.radio("Choose one answer", question.options, key=widget_key)
        answer = question.options.index(choice)
    response_key = f"{attempt_key}:{question.question_id}"
    if st.button("Submit answer", key=f"quiz_submit_{response_key}"):
        _submit(question, answer, mode, response_key)
        if timed:
            st.session_state.setdefault("quiz_timed_responses", {})[question.question_id] = st.session_state[response_key]
    submitted = st.session_state.get(response_key)
    if submitted:
        if submitted["correct"]:
            st.success("Correct. " + question.explanation)
        else:
            st.error("Not quite. " + question.explanation)
            if not question.is_numeric:
                st.caption("Correct answer: " + question.options[question.correct_index])
        if st.button("Next question" if index < len(questions) - 1 else "Finish challenge", key=f"quiz_next_{response_key}"):
            if index < len(questions) - 1:
                st.session_state[sequence_key] = index + 1
            else:
                st.session_state[f"quiz_completed_{attempt_key}"] = True
    if st.session_state.get(f"quiz_completed_{attempt_key}"):
        scoped = st.session_state.get("quiz_timed_responses", {}) if timed else {
            q.question_id: st.session_state.get(f"{attempt_key}:{q.question_id}", {}) for q in questions
        }
        _render_final_summary(scoped, questions, attempt_key)


def _render_final_summary(responses: dict[str, dict[str, Any]], questions: list[Question], completion_key: str = "") -> None:
    lookup = {q.question_id: q for q in questions}
    usable = [response for qid, response in responses.items() if qid in lookup]
    correct = sum(bool(response.get("correct")) for response in usable)
    attempted = len(usable)
    st.subheader("Challenge result")
    st.metric("Score", f"{correct} / {attempted}")
    st.metric("Accuracy", f"{(100 * correct / attempted) if attempted else 0:.1f}%")
    if attempted < len(questions):
        st.caption(f"{len(questions) - attempted} question(s) were unanswered.")
    saved_key = f"quiz_completion_saved_{completion_key or 'timed'}"
    if not st.session_state.get(saved_key):
        st.session_state.setdefault("quiz_completion_history", []).append({
            "completed_at": datetime.now().isoformat(timespec="seconds"), "attempted": attempted,
            "correct": correct, "accuracy": (100 * correct / attempted) if attempted else 0.0,
        })
        st.session_state[saved_key] = True
