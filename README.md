# LLM Output Quality Scorer

I built a tool that scores an LLM's response to a prompt against a five-part rubric, using Claude as the judge, so I could compare outputs without re-reading every response by hand.

## The Problem

When I changed a prompt or swapped a model, I had no consistent way to tell if the new output was actually better — just re-reading responses and eyeballing it. That doesn't scale past a couple of comparisons, and it's not something I could hand to someone else and get a consistent answer back.

## The Solution

I paste in a prompt and the response I want to check. Claude scores it 1–5 across five dimensions — relevance, accuracy, coherence, completeness, format/tone — and writes a short reason for each score, plus an overall average.

## Demo

**[Live demo →](#)** *(link added once deployed)*

Example of what it returns:

```
Input:  Prompt + an LLM's response to it
Output: Overall score: 4.2 / 5
        Bar chart across the five dimensions
        One-sentence rationale per dimension
        Flags for anything that looked unverified or suspicious
```

## How to Use

1. `git clone https://github.com/shipwithkatia/llm-quality-scorer.git && cd llm-quality-scorer`
2. `pip install -r requirements.txt` then `cp .env.example .env` and add an Anthropic API key — see `SECURITY.md` if you don't have one yet, especially the spend-limit step
3. `streamlit run app.py`

### Example

```
Input:  Prompt: "Explain photosynthesis to a 10-year-old."
        Response: "Plants use sunlight, water, and carbon dioxide to make food..."
Output: Overall score: 4.4 / 5 — clear and age-appropriate, one unverified detail flagged
```

## How It Works

`app.py` (Streamlit) collects a prompt/response pair. `scorer.py` wraps both in tagged delimiters — so the response can't be mistaken for instructions — and sends them to Claude with a system prompt defining the rubric. Before anything reaches the screen, the JSON that comes back is checked: every dimension present, every score numeric and in range. That validation logic has 15 tests covering it, runnable without a real API key (`python -m pytest tests/`). See `SECURITY.md` for the full write-up of API-key handling.

## Tradeoffs and Decisions

- **Why Streamlit over a custom HTML/JS frontend:** Streamlit gave me a working form-and-chart interface without hand-writing any frontend code — the right call for a five-field scoring tool I needed running quickly. The cost is less control over layout and styling than a hand-built page would give me.
- **What I'd do differently:** add even a simple CSV log of past scores. Right now every check is one-off, with no way to see if a prompt change made things better or worse over time.

## What I Learned

Building the validation step made the honest limits of the approach obvious: an LLM judging another LLM's output isn't objective the way a unit test is. The judge can miss exactly the kind of error it's supposed to catch, and its own output needed bounds-checking before I could trust it. That changed how I describe this tool — as a directional signal, not a verdict.

I also caught myself only defending against half the problem. I kept asking "what if something is missing from the model's response?" but never asked the reverse: what if the model added something extra? An undocumented extra dimension in the scores would have silently skewed the overall average with no warning at all — the same validation instinct needs to run in both directions, not just one.

## Next Steps

- [ ] Batch mode: score a CSV of prompt/response pairs at once
- [ ] Track scores over time to catch model or prompt regressions
- [ ] Compare two responses to the same prompt side by side

## Built With

- Python
- Streamlit
- Anthropic API (Claude)

## License

MIT — see [LICENSE](LICENSE).

---
Built by Katia Engalycheva | [GitHub](https://github.com/shipwithkatia) | [LinkedIn](https://www.linkedin.com/in/katiaengalycheva/)
