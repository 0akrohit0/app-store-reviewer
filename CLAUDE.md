# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## What this project is

**Play Store Reviews Analyzer**: a single-page Streamlit dashboard that scrapes Google Play reviews for an app ID, saves them locally, and runs simple NLP over them:

- **Scrape**: `google_play_scraper.reviews()` fetches one page of up to `count` reviews (no pagination through `continuation_token`).
- **Persist**: writes `data/<app_id>_<YYYYMMDD_HHMMSS>.json` and a matching `.csv`.
- **Analyze**:
  - NLTK VADER sentiment (compound score bucketed into Negative < -0.1 ≤ Neutral ≤ 0.1 < Positive)
  - text cleanup (lowercase, strip punctuation, tokenize, remove stopwords, lemmatize)
  - keyword-regex "feature request" detection
  - CountVectorizer top phrases, TF-IDF keywords, word clouds
- **Executive summary** (`reviewer/summary.py`, offline, no LLM): main issues from 1–2★ reviews and what users like from 4–5★ reviews.
  - Themes are 1–3 word phrases that are much more common in one group than the other, ranked by reach (upvotes dampened) × distinctiveness.
  - Overlapping themes are merged, and each theme is quoted with the sentence that mentions it.
  - Groups with fewer than 5 reviews get no themes.
- **AI Report** (`reviewer/ai_report.py` + `reviewer/pdf_report.py`, Claude API): the PM-facing report, in the app and as a downloadable PDF. See "AI report pipeline" below.
- **Display**: six tabs: 🤖 AI Report, Summary (offline executive summary), Overview (rating histogram/trend), Sentiment, Key Phrases (word clouds), Detailed Reviews (filters).

Status: MVP. Most code is still in `app.py` (a top-to-bottom Streamlit script). `reviewer/` is the start of the extracted, Streamlit-free logic, tested under `tests/`. There is no linting and no CI yet.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # app deps + pytest
streamlit run app.py            # http://localhost:8501
pytest                          # unit tests (config in pyproject.toml); never call the real API
```

The AI Report needs Anthropic credentials: `ANTHROPIC_API_KEY` in a `.env` file, in the environment, or in `.streamlit/secrets.toml` (`.env` and secrets are gitignored). A run costs roughly $1–2 per 1000 reviews, and the app shows an estimate before running.

On first run, NLTK downloads `vader_lexicon`, `punkt`, `stopwords`, `wordnet` and `averaged_perceptron_tagger` (cached via `st.cache_resource`), so the machine needs network access.

## Code map (`app.py`)

| Section | Functions / notes |
|---|---|
| Setup | `st.set_page_config` (must remain the **first** Streamlit call), global SSL override, NLTK downloads, `FEATURE_REQUEST_PHRASES`, module-level `sia`, `lemmatizer`, `stop_words` |
| Sidebar inputs | app ID, count, lang, country, sort order, Fetch button |
| Data | `scrape_reviews`, `save_reviews` |
| NLP | `preprocess_text`, `analyze_sentiment`, `detect_feature_requests`, `extract_key_phrases`, `extract_keywords` |
| Insights | `generate_insights` (returns a dict of counts and DataFrames), `generate_summary` (markdown) |
| Fetch | `if fetch_button and app_id:` scrapes, saves and analyzes, then stores the results in `st.session_state['analysis']` |
| UI | `if 'analysis' in st.session_state:` renders all tabs from session state, so widget reruns keep the dashboard |

Review record schema (from the scraper, and also the CSV columns): `reviewId, userName, userImage, content, score, thumbsUpCount, reviewCreatedVersion, at, replyContent, repliedAt`. Analysis adds the columns `sentiment_score, sentiment, processed_text, theme_text, is_feature_request`. `theme_text` is the same as `processed_text` but keeps the negations not/no/nor ("not working"), and the executive summary uses it.

### AI report pipeline (`reviewer/ai_report.py`)

`generate_report(df, app_id, context, client)` returns an `AIReport`, which serializes to JSON and is saved as `data/reports/<dataset>_ai_report.json` so reopening a dataset doesn't re-bill:
1. `discover_taxonomy`: Claude reads a sample (top-upvoted plus random) and proposes issue, like and request themes.
2. `classify_reviews`: every review is tagged with theme ids in chunks of `CHUNK_SIZE`, in parallel. The taxonomy is in a **cached** system prompt, and the first chunk runs alone to write the cache.
3. `compute_stats`: **all numbers are computed in Python** (counts, shares, upvotes, trend, quotes). Never let the model produce numbers that end up in the report.
4. `synthesize`: Claude writes the summary, priorities and deep-dives from the stats, and cites evidence by review id. Ids and theme ids are validated against the data, and quotes are filled in verbatim.

Rules:
- Every call goes through `_call()`: `claude-opus-5-5`, streaming, structured output via Pydantic schemas, effort set per step, and server-side `fallbacks="default"`. It raises `ReportError` on refusal, truncation or invalid output.
- Prices and the model ID are constants at the top of the file. Update them together.
- Prompt changes alter report outputs. Check them on a real dataset, and keep the system prompts byte-stable so caching keeps working (no timestamps or per-run values in `system`).
- `tests/test_ai_report.py` has a `FakeClient` that answers by schema. Extend it rather than hitting the API in tests.

`reviewer/pdf_report.py`: `build_pdf(report)` returns PDF bytes (reportlab, DejaVu font from matplotlib, matplotlib charts). Characters the font can't draw (emoji, Indic scripts) are dropped from the PDF only.

`reviewer/summary.py`: `build_executive_summary(df)` returns an `ExecutiveSummary` with `issues` and `likes` lists of `Theme`. `format_headline()` turns it into the one-paragraph markdown headline. The tunables are module constants at the top of the file (`GENERIC_WORDS`, `MIN_LIFT`, `MIN_GROUP_SIZE`, …). After changing them, check the output on a few files in `data/`, not just the unit tests.

## Known issues / tech debt (fix deliberately, not opportunistically)

1. ~~The dashboard disappears when you use a filter.~~ Fixed: results now live in `st.session_state`. Keep any new tab or widget inside the session-state render block.
2. **TLS verification is disabled globally** (`ssl._create_default_https_context = ssl._create_unverified_context`). `certifi` is imported but unused. Replace this with proper certifi-based context rather than leaving verification off.
3. **Sentiment binning edge case**: `pd.cut(bins=[-1, -0.1, 0.1, 1])` is left-exclusive, so a compound score of exactly `-1.0` becomes `NaN`. Use `include_lowest=True`.
4. **Feature-request detection is very noisy**: generic words like `need`, `want`, `update` and `feature` match most complaints.
5. Unused imports: `numpy`, `matplotlib`, `seaborn`, `go`, `certifi`. A bare `except:` in `extract_keywords` swallows all errors.
6. The lang/country choices are hard-coded to 5 Western locales, but the saved data is mostly Indian apps (`in` is not offered).
7. **`data/` (~24 MB) and `.DS_Store` are still tracked in git.** The saved reviews contain real user names and avatar URLs (PII). `.gitignore` now ignores new output, but the existing files need `git rm --cached` (an owner decision).
8. **The pinned `requirements.txt` does not install on Python 3.13** (the 2023 pins fail to build). The current code was verified with the latest releases (Streamlit 1.67). Re-pin and record the Python version in `.python-version`. Streamlit also warns that `use_container_width` is deprecated; switch to `width='stretch'`.
9. The old keyword "App Review Summary" (`generate_summary`) is weak: on a 1.4★ app it lists "ad" under both likes and dislikes. The executive summary supersedes it; consider removing it.

## Target structure (move toward this incrementally)

Separate pure logic from Streamlit so the logic can be tested without a browser:

```
app.py                 # Streamlit UI only: widgets, layout, session_state
reviewer/
  summary.py           # ✅ offline executive summary (issues / likes themes)
  ai_report.py         # ✅ Claude-powered PM report pipeline
  pdf_report.py        # ✅ PDF rendering of the AI report
  scraper.py           # fetch (with pagination), returns list[dict]
  storage.py           # save/load JSON+CSV; load previously saved runs
  analysis.py          # sentiment, preprocessing, feature requests, keywords (pure, DataFrame in → DataFrame out)
  insights.py          # aggregate metrics + summary text
tests/
  fixtures/            # small, anonymized review samples (never real user data)
  test_analysis.py ...
```

Rules for the refactor:
- Modules under `reviewer/` must not import `streamlit`.
- Pure functions take and return data and do no I/O. Network and disk access stay in `scraper.py` and `storage.py`.
- Move one function group per PR. Behavior must stay the same unless the PR says otherwise.

## Development practices

- **Branches and PRs**: never commit directly to `main`. Use short-lived branches (`feat/…`, `fix/…`, `refactor/…`) and keep each PR to one focused change. Write commit messages in the imperative mood, explaining *why*.
- **Tests**: use `pytest`. Any change to `analysis.py` or `insights.py` logic comes with tests (write the failing test first for bug fixes). Mock the scraper in tests; tests must never hit the Play Store.
- **Lint/format**: adopt `ruff` (lint and format). Run `ruff check . && ruff format . && pytest` before you push.
- **Dependencies**: pin exact versions in `requirements.txt` and add only what you use. When you remove a feature, remove its dependency too.
- **Config over constants**: thresholds (the ±0.1 sentiment cutoffs), phrase lists and locale lists belong in one clearly named place, not scattered through the UI code.
- **Performance**: cache expensive pure steps with `st.cache_data` and models/resources with `st.cache_resource`. Avoid per-row `iterrows()` rendering for large review sets; paginate instead.
- **Secrets and data**: no API keys in code (use `.streamlit/secrets.toml` or env vars, and gitignore them). Treat scraped reviews as user data: don't paste real reviews into issues, prompts or test fixtures without anonymizing them.

## Working with AI (Claude) in this repo

- **Understand before editing**: read the relevant section of `app.py` and this file first. Don't re-architect while fixing a bug; log out-of-scope issues under "Known issues" instead.
- **Plan non-trivial work**: for anything beyond a small fix, write a short plan first (files to touch, approach, test strategy) and get agreement before implementing.
- **Small, verifiable steps**: make one logical change, then verify it. Run `pytest` once tests exist. For UI changes, run `streamlit run app.py` and actually exercise the affected tab, including widget interactions (see issue 1). Report what was verified and what wasn't.
- **Match the codebase**: snake_case functions, a short comment above each function, plain pandas/plotly. Don't introduce new frameworks or abstractions without discussion.
- **No silent behavior changes**: if a change alters outputs (sentiment labels, counts, file names or formats in `data/`), call it out explicitly in the PR description.
- **Don't touch `data/`**: it's generated output. Never edit, regenerate or bulk-delete it without being asked.
- **Destructive or outward-facing actions** (force-push, deleting files, pushing, publishing) need explicit confirmation.
- **LLM features**: follow the AI report pipeline's pattern. Keep the model, prompts and prices in one module, read the API key from env/secrets, put numbers in code and wording in the model, validate every id the model returns, and keep an offline path (`summary.py`) working without a key. Get approval before running anything that spends API credits.
- **Keep this file current**: when a known issue is fixed, or the structure, commands or conventions change, update CLAUDE.md in the same PR.
