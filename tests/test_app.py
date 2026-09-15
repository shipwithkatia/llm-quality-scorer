"""
UI-level regression tests using Streamlit's AppTest harness.

These exist because scorer.py's unit tests cannot see the class of bug that
lives in app.py: Streamlit re-runs the whole script on any widget
interaction, so state that is only a local variable disappears.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAYLOAD = {
    "scores": {k: 4 for k in
               ["relevance", "accuracy", "coherence", "completeness", "format_and_tone"]},
    "rationale": {k: "ok" for k in
                  ["relevance", "accuracy", "coherence", "completeness", "format_and_tone"]},
    "flags": ["one flag"],
    "summary": "fine",
}


def _mock_api_message():
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(PAYLOAD)
    message = MagicMock()
    message.content = [block]
    message.stop_reason = "end_turn"
    return message


def _scored_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.text_area[0].set_value("Is collagen healthy?")
    at.text_area[1].set_value("Collagen isn't harmful, but...")
    at.sidebar.text_input[0].set_value("fake-key-for-test")
    at.button[0].click().run()
    return at


def test_results_render_after_scoring():
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_api_message()
        at = _scored_app()
    assert any("Overall score" in s.value for s in at.subheader)
    assert not at.exception


def test_results_survive_an_unrelated_rerun():
    """Editing a text field re-runs the script without pressing the button.
    Results must not vanish."""
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_api_message()
        at = _scored_app()
        at.text_area[1].set_value("Collagen isn't harmful, but... (typo fixed)").run()
    assert any("Overall score" in s.value for s in at.subheader), \
        "results were wiped by a rerun -- state is not persisted"
    assert not at.exception


def test_missing_key_shows_an_error_not_a_crash():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.text_area[0].set_value("p")
    at.text_area[1].set_value("r")
    at.button[0].click().run()
    assert at.error, "no error shown when the API key is missing"
    assert not at.exception


def test_fresh_result_carries_no_stale_warning():
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_api_message()
        at = _scored_app()
    assert not any("edited since" in w.value for w in at.warning)


def test_edited_input_marks_the_shown_score_as_stale():
    """Keeping results across reruns must not let an old score sit silently
    beside text it was not computed from -- for a scoring tool that is worse
    than losing the result."""
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_api_message()
        at = _scored_app()
        at.text_area[1].set_value("a completely different response").run()
    assert any("Overall score" in s.value for s in at.subheader), "result was lost"
    assert any("edited since" in w.value for w in at.warning), \
        "stale score shown with no warning"
