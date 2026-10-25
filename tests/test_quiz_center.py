from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.neurolab.quiz_center import (
    CATEGORIES,
    DIFFICULTIES,
    QUESTION_BANK,
    Question,
    calculate_progress,
    filter_questions,
    select_daily_questions,
    timed_quiz_expired,
    validate_answer,
    validate_question_bank,
)


def test_question_bank_has_unique_ids_and_valid_answers():
    validate_question_bank()
    assert {q.category for q in QUESTION_BANK} == set(CATEGORIES)
    assert {q.difficulty for q in QUESTION_BANK} == set(DIFFICULTIES)
    assert len({q.question_id for q in QUESTION_BANK}) == len(QUESTION_BANK)


def test_question_bank_validation_rejects_bad_choice_and_duplicate_id():
    invalid = Question("bad", CATEGORIES[0], "Beginner", "Question?", "Explanation.", ("a", "b", "c", "d"), 4)
    with pytest.raises(ValueError, match="valid answer index"):
        validate_question_bank((invalid,))
    first = QUESTION_BANK[0]
    with pytest.raises(ValueError, match="unique"):
        validate_question_bank((first, first))


def test_category_and_difficulty_filtering():
    selected = filter_questions(CATEGORIES[2], "Advanced")
    assert len(selected) == 1
    assert selected[0].category == CATEGORIES[2]
    assert selected[0].difficulty == "Advanced"
    assert len(filter_questions("All categories", "All levels")) == len(QUESTION_BANK)


def test_multiple_choice_and_numerical_answer_validation_with_tolerance():
    choice = next(q for q in QUESTION_BANK if q.question_id == "fund-b-1")
    assert validate_answer(choice, 0)
    assert not validate_answer(choice, 1)
    numeric = next(q for q in QUESTION_BANK if q.question_id == "loss-i-1")
    assert validate_answer(numeric, numeric.numeric_answer + numeric.tolerance * 0.5)
    assert not validate_answer(numeric, numeric.numeric_answer + numeric.tolerance * 2)


def test_daily_challenge_selection_is_stable_for_a_given_date():
    selected_a = select_daily_questions(date(2026, 10, 25), count=5)
    selected_b = select_daily_questions(date(2026, 10, 25), count=5)
    assert [q.question_id for q in selected_a] == [q.question_id for q in selected_b]
    assert len(selected_a) == 5
    assert len({q.question_id for q in selected_a}) == 5


def test_progress_calculations_and_incorrect_tracking():
    q1, q2 = QUESTION_BANK[:2]
    result = calculate_progress({q1.question_id: {"correct": True}, q2.question_id: {"correct": False}})
    assert result["attempted"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 50
    assert result["incorrect_ids"] == [q2.question_id]
    assert result["by_category"][q1.category] == {"attempted": 2, "correct": 1}
    assert result["by_difficulty"][q2.difficulty] == {"attempted": 1, "correct": 0}


def test_timed_quiz_expiration_boundary():
    started = datetime(2026, 10, 25, 12, 0, 0)
    assert not timed_quiz_expired(started, 60, started + timedelta(seconds=59))
    assert timed_quiz_expired(started, 60, started + timedelta(seconds=60))
    with pytest.raises(ValueError, match="positive"):
        timed_quiz_expired(started, 0)


def test_quiz_page_renders_and_daily_mode_is_available():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_quiz_and_challenge_center").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Quiz & Challenge Center"]
    mode_selector = next(widget for widget in app.selectbox if widget.key == "quiz_center_mode")
    assert "Daily Challenge" in mode_selector.options

