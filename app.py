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
    help=(
        "Not stored anywhere, and never pre-filled with a server-side key "
        "even if one is configured for this deployment -- see the caption "
        "below."
    ),
)
st.sidebar.caption(
    "Leave blank to use this deployment's configured key, if the person "
    "hosting it set one up server-side."
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

if run:
    if not effective_key:
        st.error("Add your Anthropic API key in the sidebar first.")
    elif not prompt.strip() or not response.strip():
        st.error("Both the prompt and the response are required.")
    else:
        with st.spinner("Scoring against the rubric..."):
            try:
                result = score_output(prompt, response, api_key=effective_key)
            except ScorerError as e:
                st.error(str(e))
                result = None

        if result:
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
