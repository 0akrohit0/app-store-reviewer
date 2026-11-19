# Play Store Reviews Analyzer

A Streamlit app for product managers. It fetches Google Play reviews for any app and turns them into insights: what users complain about, what they like, and what to fix first. It can produce a **PDF report** written by Claude.

## What you get

| Tab | What it shows | Needs an API key? |
|---|---|---|
| 🤖 **AI Report** | Executive summary, prioritised issues (P0/P1/P2) with exact counts and trends, deep-dives with recommendations, what users like, feature requests and risks. **Downloadable as a PDF.** | Yes (Anthropic) |
| **Summary** | Offline executive summary: main issues from 1–2★ reviews and what users like from 4–5★ reviews, with quotes | No |
| **Overview** | Rating distribution, average rating over time, summary stats | No |
| **Sentiment Analysis** | Positive / neutral / negative breakdown (NLTK VADER) | No |
| **Key Phrases** | Word clouds and top phrases by sentiment | No |
| **Detailed Reviews** | Every review, filterable by sentiment and rating | No |

Everything except the AI Report works offline, without an API key.

## Quick start

**Requirements:** Python 3.13 (versions in `requirements.txt` were verified together on 3.13) and git.

```bash
git clone https://github.com/0akrohit0/app-store-reviewer.git
cd app-store-reviewer

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # opens http://localhost:8501
```

On first launch the app downloads NLTK language data to `~/nltk_data` (about 110 MB on disk). This needs internet access, takes a minute, and only happens once.

### Enable the AI Report (optional)

1. Get an API key from the [Anthropic Console](https://console.anthropic.com/) and make sure the account has credit.
2. Copy the template and add your key:
   ```bash
   cp .env.example .env
   # then edit .env:  ANTHROPIC_API_KEY=sk-ant-...
   ```
   `.env` is gitignored, so never commit it. Alternatively, set `ANTHROPIC_API_KEY` in your environment or in `.streamlit/secrets.toml`.
3. Restart `streamlit run app.py`.

## How to use it

1. **Enter the App ID** in the sidebar. It's the `id=` part of the app's Play Store URL:
   `play.google.com/store/apps/details?id=`**`com.theporter.android.driverapp`**
2. Choose how many reviews to fetch, the language, the country and the sort order, then click **Fetch Reviews**.
3. Explore the tabs. For the AI Report:
   - Optionally describe the context, e.g. *"Reviewers are delivery drivers. I own onboarding and earnings. What should we fix first?"*
   - Check the cost estimate, click **Generate AI report**, and wait 1–3 minutes.
   - Click **📄 Download PDF**.

If you get "No reviews found", check the App ID. It usually starts with `com.` and is case-sensitive.

### AI Report: how it works and what it costs

Claude (`claude-opus-5-5`) proposes themes from a sample of reviews, then tags **every** review with them. **All numbers in the report are computed in code** from those tags, and quotes are verbatim, so the model doesn't make up statistics. Claude then writes the summary, priorities and recommendations from those numbers.

- **Cost:** roughly $1–2 per 1,000 reviews. The app shows an estimate before you run and the actual cost afterwards.
- **No double charges:** reports are saved in `data/reports/`, so reopening the same dataset doesn't cost anything. Click **Regenerate** to rerun.

## Your data

- Fetched reviews are saved to `data/<app_id>_<timestamp>.json` and `.csv`. AI reports go in `data/reports/`.
- Reviews contain reviewers' public names and avatar links. New files in `data/` are gitignored; keep it that way and don't commit scraped data.
- With the AI Report, review text is sent to the Anthropic API.

## Project structure

```
app.py                  Streamlit UI: sidebar, fetching, tabs
reviewer/
  summary.py            Offline executive summary (no API key needed)
  ai_report.py          Claude pipeline: themes → tagging → stats → written report
  pdf_report.py         Builds the PDF (reportlab + matplotlib charts)
tests/                  pytest suite (uses a fake Claude client, so it never calls the API)
data/                   Fetched reviews and saved reports (new files are gitignored)
requirements.txt        App dependencies (pinned)
requirements-dev.txt    + pytest and ruff
.env.example            Template for your API key
CLAUDE.md               Architecture notes, known issues and conventions for contributors (and AI assistants)
```

## Development

```bash
pip install -r requirements-dev.txt
pytest          # tests
ruff check .    # lint
```

Read `CLAUDE.md` before making changes. It covers the architecture, known issues, and how to work on the AI pipeline without breaking caching or report accuracy.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails building pandas or scikit-learn | Use Python 3.13. Older pins needed older Pythons. |
| `certificate verify failed` on macOS | Run `/Applications/Python\ 3.13/Install\ Certificates.command`, or `pip install --upgrade certifi`. |
| "No reviews found" | The App ID is wrong (often missing `com.`), or that language/country has no reviews. |
| AI Report: "No Anthropic API key found" | Create `.env` from `.env.example` and restart the app. |
| AI Report: "credit balance is too low" | Add credit to your Anthropic account (Plans & Billing in the Console). |
| `LookupError: Resource punkt_tab not found` | The first-run NLTK download was interrupted. Restart the app with internet access. |
