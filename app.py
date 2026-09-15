"""
app.py

Streamlit front end for the LLM Output Quality Scorer.
Run locally with: streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from scorer import MAX_INPUT_CHARS, RUBRIC, ScorerError, score_output

load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env file, if present

st.set_page_config(page_title="LLM Output Quality Scorer", page_icon="🧪", layout="centered")

st.title("🧪 LLM Output Quality Scorer")
st.caption(
    "Paste a prompt and an LLM's response to it. Claude scores the response "
    "against a five-dimension rubric and explains why."
)

st.info(
    "Don't paste confidential, personal, or sensitive text. Anything you "
    "enter here is sent to Anthropic's API to be scored and isn't under "
    "this app's control once submitted.",
    icon="🔒",
)

with st.expander("What's being measured?"):
    for dim, description in RUBRIC.items():
        st.markdown(f"**{dim.replace('_', ' ').title()}** — {description}")

with st.expander("Known limitations"):
    st.markdown(
        "- **The judge is also an LLM.** Treat scores as a directional "
        "signal, not ground truth.\n"
        "- **Prompt injection.** Text in the 'response' field is treated "
        "as untrusted data, but a sufficiently crafted input could still "
        "influence scoring. Flags on unexpected/suspicious content are "
        "shown when the model notices something odd.\n"
        f"- **Input length is capped at {MAX_INPUT_CHARS:,} characters** "
        "per field to keep cost and load predictable."
    )

api_key_input = st.sidebar.text_input(
    "Anthropic API key",
    type="password",
    placeholder="sk-ant-...",
    help="Used for this session only. Not stored, not logged, not sent anywhere but Anthropic's API.",
)
st.sidebar.caption(
    "**Required.** This deployment has no server-side key, so bring your own: "
    "the app can't call Claude without one. Get a key at "
    "[console.anthropic.com](https://console.anthropic.com/settings/keys) -- "
    "new accounts include free credit. Your key is used for this session only "
    "and is never stored."
)
effective_key = api_key_input or os.environ.get("ANTHROPIC_API_KEY")

prompt = st.text_area(
    "Prompt given to the LLM",
    height=120,
    max_chars=MAX_INPUT_CHARS,
    placeholder="e.g. Explain photosynthesis to a 10-year-old.",
)
response = st.text_area(
    "LLM's response to evaluate",
    height=200,
    max_chars=MAX_INPUT_CHARS,
    placeholder="Paste the response you want scored...",
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("Score it", type="primary", use_container_width=True)

# Streamlit re-runs the whole script on ANY widget interaction. Without
# stashing the outcome, a user who edits a field after scoring silently
# loses their results. Keep it in session_state and render from there.
if run:
    st.session_state["scorer_error"] = None
    st.session_state["scorer_result"] = None
    st.session_state["scored_inputs"] = (prompt, response)
    if not effective_key:
        st.session_state["scorer_error"] = (
            "Add your Anthropic API key in the sidebar to score. This demo "
            "has no shared key -- the sidebar says where to get one."
        )
    elif not prompt.strip() or not response.strip():
        st.session_state["scorer_error"] = "Both the prompt and the response are required."
    else:
        with st.spinner("Scoring against the rubric..."):
            try:
                st.session_state["scorer_result"] = score_output(
                    prompt, response, api_key=effective_key
                )
            except ScorerError as e:
                st.session_state["scorer_error"] = str(e)

# Streamlit re-runs the whole script on ANY widget interaction, so the outcome
# is kept in session_state rather than in a local that a rerun would discard.
# But a kept result is only valid for the text it was computed from: if the
# fields have changed since, say so rather than letting an old score sit
# silently next to new text.
result = st.session_state.get("scorer_result")
error = st.session_state.get("scorer_error")
is_stale = st.session_state.get("scored_inputs") not in (None, (prompt, response))

if error:
    st.error(error)

if result:
    if is_stale:
        st.warning(
            "The prompt or response has been edited since this score was "
            "computed. Press **Score it** again to rescore.",
            icon="\u26a0\ufe0f",
        )

    st.subheader(f"Overall score: {result.overall} / 5")
    st.write(result.summary)

    df = pd.DataFrame(
        {
            "Dimension": [d.replace("_", " ").title() for d in result.scores],
            "Score": list(result.scores.values()),
        }
    ).set_index("Dimension")
    st.bar_chart(df, y="Score")

    st.subheader("Rationale")
    for dim, text in result.rationale.items():
        st.markdown(f"**{dim.replace('_', ' ').title()} ({result.scores[dim]}/5):** {text}")

    if result.flags:
        st.subheader("Flags")
        for flag in result.flags:
            st.warning(flag)

st.divider()
st.caption(
    "Built with Claude's API. Scores are directional, not ground truth — "
    "use them to spot patterns across many outputs, not as a single source "
    "of truth on any one response."
)
