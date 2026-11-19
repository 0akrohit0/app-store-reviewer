"""Claude-powered PM report: themes, counts, deep-dives and recommendations.

Pipeline (no Streamlit here, so it can be tested with a fake client):

1. discover_taxonomy  - Claude reads a sample of reviews and proposes the app's
                        issue / like / feature-request themes.
2. classify_reviews   - every review is tagged with theme ids, in chunks. The
                        taxonomy lives in a cached system prompt, so repeating
                        it per chunk is cheap.
3. compute_stats      - counts, shares, upvotes, trends and quote candidates are
                        computed in Python. Numbers in the report never come
                        from the model.
4. synthesize         - Claude writes the executive summary, priorities and
                        deep-dives from those stats. It cites evidence by review
                        id; quotes are filled in verbatim from the data.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from typing import Callable, Literal, Optional

import anthropic
import pandas as pd
from pydantic import BaseModel, ValidationError

MODEL = "claude-opus-5-5"
# Opus 5.5 can't disable thinking; effort is the cost/quality control
TAXONOMY_EFFORT = "medium"
CLASSIFY_EFFORT = "low"
SYNTHESIS_EFFORT = "high"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

TAXONOMY_SAMPLE_SIZE = 400
CHUNK_SIZE = 100
MAX_PARALLEL_REQUESTS = 4
QUOTES_PER_THEME = 6
CRITICAL_MAX_SCORE = 2
FAVOURABLE_MIN_SCORE = 4

# USD per million tokens for MODEL (cache writes are 1.25x input with the 5-minute TTL)
PRICE_INPUT = 4.00
PRICE_OUTPUT = 20.00
PRICE_CACHE_READ = 0.20
PRICE_CACHE_WRITE = 5.00
# Rough planning numbers for the pre-run estimate
CHARS_PER_TOKEN = 3.5
OUTPUT_TOKENS_PER_REVIEW = 40


class ReportError(Exception):
    """A step failed in a way the user should see (refusal, truncation, bad output)."""


# --- Structured output schemas -------------------------------------------------

class Theme(BaseModel):
    id: str  # short snake_case, e.g. "doc_verification_delay"
    name: str
    kind: Literal["issue", "like", "request"]
    definition: str


class Taxonomy(BaseModel):
    themes: list[Theme]


class ReviewLabel(BaseModel):
    review_id: str
    theme_ids: list[str]


class ChunkLabels(BaseModel):
    labels: list[ReviewLabel]


class Recommendation(BaseModel):
    action: str
    impact: Literal["high", "medium", "low"]
    effort: Literal["high", "medium", "low"]


class Priority(BaseModel):
    theme_id: str
    priority: Literal["P0", "P1", "P2"]
    rationale: str


class DeepDive(BaseModel):
    theme_id: str
    headline: str
    what_is_happening: str
    user_impact: str
    likely_root_causes: list[str]
    recommendations: list[Recommendation]
    metric_to_track: str
    evidence_review_ids: list[str]


class Synthesis(BaseModel):
    headline: str
    executive_summary: str
    key_takeaways: list[str]
    priorities: list[Priority]
    deep_dives: list[DeepDive]
    what_users_like: str
    feature_requests_summary: str
    risks_and_watchouts: list[str]


# --- Report container ----------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage):
        # Classification chunks run in parallel threads
        with self._lock:
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.requests += 1

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')} | {
            'cost_usd': round(self.cost_usd, 4)}

    @property
    def cost_usd(self):
        return (
            self.input_tokens * PRICE_INPUT
            + self.output_tokens * PRICE_OUTPUT
            + self.cache_read_tokens * PRICE_CACHE_READ
            + self.cache_write_tokens * PRICE_CACHE_WRITE
        ) / 1_000_000


@dataclass
class AIReport:
    app_id: str
    context: str
    generated_at: str
    model: str
    overview: dict
    themes: list  # list[dict]: theme + stats + quotes, sorted by count
    synthesis: dict
    monthly: list  # list[dict]: month, reviews, avg_rating
    rating_distribution: dict
    unlabeled_reviews: int
    usage: dict = field(default_factory=dict)

    def theme(self, theme_id):
        return next((t for t in self.themes if t["id"] == theme_id), None)

    def to_json(self):
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))


# --- Helpers -------------------------------------------------------------------

def make_client(api_key=None):
    # With no key, the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    # Without credentials the SDK only fails (with a TypeError) once a request is sent
    if not (client.api_key or client.auth_token or getattr(client, 'credentials', None)):
        raise ReportError(
            "No Anthropic API key found. Add ANTHROPIC_API_KEY to a .env file in the project folder, "
            "your environment, or .streamlit/secrets.toml."
        )
    return client


def prepare_reviews(df):
    """Short ids ("r1", "r2", ...) and the fields the model needs; keeps prompts compact."""
    out = pd.DataFrame({
        'rid': [f"r{i + 1}" for i in range(len(df))],
        'review_id': df['reviewId'].astype(str).to_numpy() if 'reviewId' in df else None,
        'score': df['score'].astype(int).to_numpy(),
        'thumbs': df.get('thumbsUpCount', pd.Series(0, index=df.index)).fillna(0).astype(int).to_numpy(),
        'at': pd.to_datetime(df['at']).to_numpy(),
        'version': df.get('reviewCreatedVersion', pd.Series(None, index=df.index)).fillna('').astype(str).to_numpy(),
        'replied': df.get('replyContent', pd.Series(None, index=df.index)).notna().to_numpy(),
        'content': df['content'].fillna('').astype(str).map(lambda t: ' '.join(t.split())).to_numpy(),
    })
    return out[out['content'] != ''].reset_index(drop=True)


def _format_reviews(reviews):
    # Play Store reviews are capped at 500 characters, so nothing is truncated
    return '\n'.join(
        f"[{r.rid}] {r.score}★ | {r.thumbs} upvotes | {pd.Timestamp(r.at):%Y-%m-%d} | {r.content}"
        for r in reviews.itertuples()
    )


def _call(client, usage, schema, system, user, effort, max_tokens):
    """One structured-output request. Streams to avoid HTTP timeouts on long outputs."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",  # re-run a declined request on Anthropic's recommended fallback model
    ) as stream:
        message = stream.get_final_message()
    usage.add(message.usage)

    if message.stop_reason == "refusal":
        raise ReportError("Claude declined this request (refusal). Try a different set of reviews.")
    if message.stop_reason == "max_tokens":
        raise ReportError("Claude's response was cut off (max_tokens). Try fewer reviews.")
    text = ''.join(b.text for b in message.content if b.type == "text")
    try:
        return schema.model_validate_json(text)
    except ValidationError as e:
        raise ReportError(f"Claude returned output that doesn't match the report schema: {e}") from e


def _cached_system(text):
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _context_line(context):
    return f"\nContext from the product manager: {context.strip()}\n" if context and context.strip() else ""


# --- Step 1: taxonomy ----------------------------------------------------------

TAXONOMY_SYSTEM = """You are a senior product analyst. You read app store reviews and build a theme taxonomy that a product team can act on.

Rules for themes:
- 8-14 "issue" themes, 3-6 "like" themes and 3-8 "request" themes (feature requests).
- Each theme is one specific, actionable topic a team could own (e.g. "Document verification stuck after paying the joining fee"), not a sentiment ("bad app") or a vague area ("problems").
- Themes should be mutually distinct; split a theme if it mixes unrelated causes.
- ids are short snake_case and unique.
- Definitions say what counts and what doesn't, so another analyst could tag reviews consistently."""


def discover_taxonomy(client, reviews, context, usage, seed=0):
    # Most-upvoted reviews carry the most signal; add a random slice for coverage
    by_votes = reviews.sort_values('thumbs', ascending=False).head(TAXONOMY_SAMPLE_SIZE // 2)
    rest = reviews.drop(by_votes.index)
    random_part = rest.sample(min(len(rest), TAXONOMY_SAMPLE_SIZE - len(by_votes)), random_state=seed)
    sample = pd.concat([by_votes, random_part])

    user = (
        f"{_context_line(context)}"
        f"Here are {len(sample)} reviews (of {len(reviews)} total). Build the theme taxonomy.\n\n"
        f"{_format_reviews(sample)}"
    )
    taxonomy = _call(client, usage, Taxonomy, TAXONOMY_SYSTEM, user, TAXONOMY_EFFORT, 16000)

    seen = set()
    taxonomy.themes = [t for t in taxonomy.themes if not (t.id in seen or seen.add(t.id))]
    if not any(t.kind == "issue" for t in taxonomy.themes):
        raise ReportError("Claude didn't find any issue themes in these reviews.")
    return taxonomy


# --- Step 2: classification ----------------------------------------------------

CLASSIFY_SYSTEM = """You tag app store reviews with themes from a fixed taxonomy.

For every review, return its review_id and the ids of ALL themes it clearly mentions (a review can match several, or none). Only use theme ids from the taxonomy. Tag what the review actually says, not what it implies. Return one label per review, for every review you are given.

Taxonomy:
{taxonomy}"""


def classify_reviews(client, reviews, taxonomy, usage, progress=None):
    valid_ids = {t.id for t in taxonomy.themes}
    system = _cached_system(CLASSIFY_SYSTEM.format(
        taxonomy=json.dumps([t.model_dump() for t in taxonomy.themes], indent=1, ensure_ascii=False)
    ))
    chunks = [reviews.iloc[i:i + CHUNK_SIZE] for i in range(0, len(reviews), CHUNK_SIZE)]

    def run(chunk):
        result = _call(client, usage, ChunkLabels, system,
                       f"Tag these {len(chunk)} reviews:\n\n{_format_reviews(chunk)}",
                       CLASSIFY_EFFORT, 16000)
        chunk_ids = set(chunk['rid'])
        return {
            label.review_id: [t for t in dict.fromkeys(label.theme_ids) if t in valid_ids]
            for label in result.labels if label.review_id in chunk_ids
        }

    labels = {}
    # The first chunk runs alone so it writes the cache that the parallel chunks then read
    labels.update(run(chunks[0]))
    if progress:
        progress(1, len(chunks))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as pool:
        for done, chunk_labels in enumerate(pool.map(run, chunks[1:]), start=2):
            labels.update(chunk_labels)
            if progress:
                progress(done, len(chunks))
    return labels


# --- Step 3: statistics (pure Python) ------------------------------------------

def _recent_split(reviews):
    """Split into an earlier and a recent period: the last 30 days if both sides have
    enough reviews, otherwise the two halves by date."""
    latest = reviews['at'].max()
    cutoff = latest - pd.Timedelta(days=30)
    if (reviews['at'] > cutoff).sum() >= 30 and (reviews['at'] <= cutoff).sum() >= 30:
        return cutoff, "last 30 days"
    return reviews['at'].median(), "second half of the period"


def _pick_quotes(theme_reviews, n=QUOTES_PER_THEME):
    # Prefer substantive, upvoted reviews; recent ones break ties
    ranked = theme_reviews.assign(_long=theme_reviews['content'].str.len() >= 60).sort_values(
        ['_long', 'thumbs', 'at'], ascending=False)
    return [
        {'rid': r.rid, 'score': int(r.score), 'thumbs': int(r.thumbs),
         'date': f"{pd.Timestamp(r.at):%Y-%m-%d}", 'text': r.content}
        for r in ranked.head(n).itertuples()
    ]


def compute_stats(reviews, labels, taxonomy):
    reviews = reviews.copy()
    reviews['themes'] = reviews['rid'].map(lambda rid: labels.get(rid, []))
    n = len(reviews)
    critical = reviews['score'] <= CRITICAL_MAX_SCORE
    favourable = reviews['score'] >= FAVOURABLE_MIN_SCORE
    cutoff, recent_label = _recent_split(reviews)
    recent = reviews['at'] > cutoff

    themes = []
    for theme in taxonomy.themes:
        has = reviews['themes'].map(lambda ids, tid=theme.id: tid in ids)
        count = int(has.sum())
        if count == 0:
            continue
        earlier_share = has[~recent].mean() if (~recent).any() else 0.0
        recent_share = has[recent].mean() if recent.any() else 0.0
        themes.append({
            **theme.model_dump(),
            'count': count,
            'share': count / n,
            'share_of_critical': float(has[critical].mean()) if critical.any() else 0.0,
            'share_of_favourable': float(has[favourable].mean()) if favourable.any() else 0.0,
            'upvotes': int(reviews.loc[has, 'thumbs'].sum()),
            'avg_rating': float(reviews.loc[has, 'score'].mean()),
            'earlier_share': float(earlier_share),
            'recent_share': float(recent_share),
            'quotes': _pick_quotes(reviews[has]),
        })
    themes.sort(key=lambda t: t['count'], reverse=True)

    monthly = (
        reviews.groupby(reviews['at'].dt.to_period('M'))['score']
        .agg(['count', 'mean']).reset_index()
    )
    overview = {
        'total_reviews': n,
        'avg_rating': float(reviews['score'].mean()),
        'critical_share': float(critical.mean()),
        'favourable_share': float(favourable.mean()),
        'developer_reply_rate': float(reviews['replied'].mean()),
        'date_from': f"{reviews['at'].min():%Y-%m-%d}",
        'date_to': f"{reviews['at'].max():%Y-%m-%d}",
        'trend_window': recent_label,
        'trend_cutoff': f"{cutoff:%Y-%m-%d}",
    }
    return {
        'overview': overview,
        'themes': themes,
        'monthly': [{'month': str(r['at']), 'reviews': int(r['count']), 'avg_rating': float(r['mean'])}
                    for _, r in monthly.iterrows()],
        'rating_distribution': {int(k): int(v) for k, v in reviews['score'].value_counts().sort_index().items()},
        'unlabeled_reviews': int(sum(1 for rid in reviews['rid'] if rid not in labels)),
    }


# --- Step 4: synthesis ---------------------------------------------------------

SYNTHESIS_SYSTEM = """You are a senior product manager writing a review-insights report for your product team and leadership.

You get statistics computed from every review (these numbers are exact; quote them, don't change them) and a few verbatim example reviews per theme, each with an id.

Write like a strong PM: specific, evidence-based, prioritised, no filler.
- executive_summary: 4-6 sentences. Lead with the most important finding and what it costs the business or users.
- key_takeaways: 3-6 one-line bullets with numbers.
- priorities: rank the issue themes worth acting on. P0 = blocking users or causing churn/trust damage at scale, P1 = significant, P2 = worth tracking. Rationale must cite reach, severity and trend.
- deep_dives: one for each of the 3-6 most important issue themes. Root causes are hypotheses grounded in what reviews say. Recommendations are concrete product or ops changes, each with impact and effort. evidence_review_ids must be ids from that theme's examples.
- what_users_like and feature_requests_summary: short paragraphs grounded in the like/request themes.
- risks_and_watchouts: emerging or worsening problems, brand/legal/trust risks, and caveats about the data.
Use theme_id values exactly as given. Don't invent numbers that aren't in the statistics."""


def synthesize(client, stats, context, app_id, usage):
    payload = {
        'app_id': app_id,
        'overview': stats['overview'],
        'monthly': stats['monthly'],
        'themes': [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k != 'quotes'}
            | {'examples': [{'id': q['rid'], 'stars': q['score'], 'upvotes': q['thumbs'], 'text': q['text']}
                            for q in t['quotes']]}
            for t in stats['themes']
        ],
    }
    user = (
        f"{_context_line(context)}"
        "Shares are fractions of all reviews (share), of 1-2★ reviews (share_of_critical) and of "
        f"4-5★ reviews (share_of_favourable). earlier_share vs recent_share compares before and after "
        f"{stats['overview']['trend_cutoff']} ({stats['overview']['trend_window']}).\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    synthesis = _call(client, usage, Synthesis, SYNTHESIS_SYSTEM, user, SYNTHESIS_EFFORT, 32000)

    # Keep only references the data can back up
    theme_ids = {t['id'] for t in stats['themes']}
    synthesis.priorities = [p for p in synthesis.priorities if p.theme_id in theme_ids]
    synthesis.deep_dives = [d for d in synthesis.deep_dives if d.theme_id in theme_ids]
    for dive in synthesis.deep_dives:
        quote_ids = {q['rid'] for t in stats['themes'] if t['id'] == dive.theme_id for q in t['quotes']}
        dive.evidence_review_ids = [rid for rid in dive.evidence_review_ids if rid in quote_ids]
    return synthesis


# --- Orchestration -------------------------------------------------------------

def estimate_cost(df):
    """Rough pre-run estimate in USD, so the PM knows what a run costs before starting."""
    chars = df['content'].fillna('').str.len().sum() + 60 * len(df)
    review_tokens = chars / CHARS_PER_TOKEN
    sample_tokens = review_tokens * min(1.0, TAXONOMY_SAMPLE_SIZE / max(len(df), 1))
    input_tokens = sample_tokens + review_tokens + 30_000  # + synthesis payload
    output_tokens = len(df) * OUTPUT_TOKENS_PER_REVIEW + 25_000  # labels + thinking + report
    return (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / 1_000_000


def generate_report(df, app_id, context="", client=None,
                    progress: Optional[Callable[[str, float], None]] = None):
    """Run the full pipeline. progress(message, fraction_done) is called between steps."""
    client = client or make_client()
    usage = Usage()
    say = progress or (lambda message, fraction: None)

    reviews = prepare_reviews(df)
    if len(reviews) < 10:
        raise ReportError("Need at least 10 reviews with text to build a report.")

    say("Finding themes in a sample of reviews…", 0.02)
    taxonomy = discover_taxonomy(client, reviews, context, usage)

    def on_chunk(done, total):
        say(f"Tagging reviews with themes ({done}/{total} batches)…", 0.15 + 0.6 * done / total)
    say("Tagging reviews with themes…", 0.15)
    labels = classify_reviews(client, reviews, taxonomy, usage, on_chunk)

    say("Computing statistics…", 0.77)
    stats = compute_stats(reviews, labels, taxonomy)

    say("Writing the report…", 0.8)
    synthesis = synthesize(client, stats, context, app_id, usage)
    say("Done", 1.0)

    return AIReport(
        app_id=app_id,
        context=context,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        model=MODEL,
        overview=stats['overview'],
        themes=stats['themes'],
        synthesis=synthesis.model_dump(),
        monthly=stats['monthly'],
        rating_distribution=stats['rating_distribution'],
        unlabeled_reviews=stats['unlabeled_reviews'],
        usage=usage.to_dict(),
    )
