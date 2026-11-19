"""Offline executive summary: main issues and what users like.

Pure pandas/scikit-learn, no Streamlit, so it can be unit tested.

Reviews are split by star rating (ratings are more reliable than VADER on short
reviews): 1-2 stars are "critical", 4-5 stars are "favourable". A theme is a
1-3 word phrase that is mentioned by several reviews in one group and is
noticeably more common there than in the other group. Themes are ranked by
reach (each matching review counts 1 + log10(1 + upvotes) / 2, so upvotes help
but one viral review can't dominate) times distinctiveness.
"""

from dataclasses import dataclass, field
import math
import re

from sklearn.feature_extraction.text import CountVectorizer

CRITICAL_MAX_SCORE = 2
FAVOURABLE_MIN_SCORE = 4
MAX_THEMES = 5
MIN_LIFT = 1.5
# Groups smaller than this are too small to find recurring themes in
MIN_GROUP_SIZE = 5
# A theme must appear in at least this many reviews (and MIN_MENTION_SHARE of the group)
MIN_MENTIONS = 2
MIN_MENTION_SHARE = 0.03
# Longer phrases are more specific ("customer care" beats "care")
NGRAM_BONUS = 0.5
# A theme is skipped if most of its reviews are already covered by a higher-ranked one
MAX_REVIEW_OVERLAP = 0.5
QUOTE_MIN_CHARS = 30
QUOTE_MAX_CHARS = 220

# Phrases made only of these words carry sentiment but no actual topic
GENERIC_WORDS = {
    'app', 'application', 'good', 'bad', 'nice', 'great', 'best', 'worst',
    'excellent', 'awesome', 'amazing', 'love', 'like', 'useful', 'super',
    'very', 'really', 'also', 'even', 'one', 'get', 'thing', 'please', 'ok',
    'okay', 'well', 'much', 'thanks', 'thank', 'experience', 'star', 'stars',
    'rating', 'review', 'not', 'no', 'nor', 'dont', 'doesnt', 'didnt', 'cant',
    'wont', 'isnt', 'never', 'always', 'time', 'every', 'many', 'would',
    'still', 'now', 'day', 'days', 'minute', 'minutes', 'hour', 'hours',
    'month', 'months', 'year', 'years', 'something', 'nothing', 'anything',
    'everything', 'back', 'worse', 'better', 'lot', 'make', 'made', 'give',
    'go', 'going', 'want', 'need', 'know', 'see', 'use', 'using', 'used',
    'issue', 'problem', 'user', 'people', 'way', 'say', 'come', 'take',
    'ever', 'right', 'try', 'tried', 'proper', 'properly', 'lover',
}
NEGATION_WORDS = {'not', 'no', 'nor'}


@dataclass
class Theme:
    phrase: str
    review_count: int
    share: float  # fraction of the group's reviews that mention the phrase
    upvotes: int
    quote: str
    review_ids: set = field(default_factory=set, repr=False)


@dataclass
class ExecutiveSummary:
    total_reviews: int
    avg_rating: float
    critical_count: int
    favourable_count: int
    issues: list
    likes: list

    @property
    def critical_percent(self):
        return 100 * self.critical_count / self.total_reviews if self.total_reviews else 0.0

    @property
    def favourable_percent(self):
        return 100 * self.favourable_count / self.total_reviews if self.total_reviews else 0.0


def _truncate(text, max_chars=QUOTE_MAX_CHARS):
    text = ' '.join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '…'


def _is_generic(phrase):
    words = phrase.split()
    # A dangling negation ("order not") reads as a fragment
    return words[-1] in NEGATION_WORDS or all(word in GENERIC_WORDS for word in words)


def _review_weight(thumbs):
    return 1 + math.log10(1 + max(thumbs, 0)) / 2


def _sentence_mentions(sentence, phrase):
    # Theme text is lemmatized, so match each phrase word as a word prefix ("ad" ~ "ads")
    words = re.findall(r'\w+', sentence.lower())
    return all(any(w.startswith(token) for w in words) for token in phrase.split())


def _pick_quote(group_df, phrase):
    """Quote the sentence that mentions the theme, from the most upvoted review."""
    best = None
    for content, thumbs in zip(group_df['content'].fillna(''), group_df['thumbsUpCount'], strict=True):
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', str(content)):
            sentence = sentence.strip()
            if len(sentence) < QUOTE_MIN_CHARS or not _sentence_mentions(sentence, phrase):
                continue
            rank = (_review_weight(thumbs), -len(sentence))
            if best is None or rank > best[0]:
                best = (rank, sentence)
    if best:
        return _truncate(best[1])
    # Fall back to the start of the most upvoted review
    return _truncate(group_df.sort_values('thumbsUpCount', ascending=False).iloc[0]['content'])


def extract_themes(group_df, other_df, text_col='theme_text', max_themes=MAX_THEMES):
    """Return the top themes that distinguish group_df from other_df."""
    group_df = group_df[group_df[text_col].fillna('').str.strip() != '']
    if len(group_df) < MIN_GROUP_SIZE:
        return []

    n_group, n_other = len(group_df), len(other_df)
    min_mentions = max(MIN_MENTIONS, math.ceil(MIN_MENTION_SHARE * n_group))
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        binary=True,
        min_df=min_mentions,
        token_pattern=r'(?u)\b[^\W\d_]{2,}\b',  # words only, no numbers
    )
    try:
        group_matrix = vectorizer.fit_transform(group_df[text_col].fillna(''))
    except ValueError:  # empty vocabulary
        return []
    phrases = vectorizer.get_feature_names_out()

    group_counts = group_matrix.sum(axis=0).A1
    if n_other:
        other_counts = vectorizer.transform(other_df[text_col].fillna('')).sum(axis=0).A1
    else:
        other_counts = [0] * len(phrases)

    weights = group_df['thumbsUpCount'].fillna(0).apply(_review_weight).to_numpy()
    reach = group_matrix.T @ weights

    candidates = []
    for i, phrase in enumerate(phrases):
        if _is_generic(phrase):
            continue
        # Smoothed ratio of how often the phrase appears in this group vs the other
        lift = ((group_counts[i] + 0.5) / (n_group + 1)) / ((other_counts[i] + 0.5) / (n_other + 1))
        if lift < MIN_LIFT:
            continue
        ngram_bonus = 1 + NGRAM_BONUS * (len(phrase.split()) - 1)
        candidates.append((reach[i] * math.log(lift) * ngram_bonus, i, phrase))
    candidates.sort(reverse=True)

    review_ids = group_df['reviewId'].to_numpy()
    upvotes = group_df['thumbsUpCount'].fillna(0).to_numpy()
    themes = []
    for _, i, phrase in candidates:
        rows = group_matrix[:, i].nonzero()[0]
        ids = set(review_ids[rows])
        if any(len(ids & t.review_ids) / min(len(ids), len(t.review_ids)) > MAX_REVIEW_OVERLAP for t in themes):
            continue
        themes.append(Theme(
            phrase=phrase,
            review_count=len(rows),
            share=len(rows) / n_group,
            upvotes=int(upvotes[rows].sum()),
            quote=_pick_quote(group_df.iloc[rows], phrase),
            review_ids=ids,
        ))
        if len(themes) == max_themes:
            break
    return themes


def build_executive_summary(df, text_col='theme_text'):
    """Build the executive summary from an analyzed reviews DataFrame.

    Expects columns: reviewId, content, score, thumbsUpCount and text_col
    (normalized text that keeps negations).
    """
    df = df.copy()
    if 'thumbsUpCount' not in df:
        df['thumbsUpCount'] = 0
    df['thumbsUpCount'] = df['thumbsUpCount'].fillna(0)
    critical = df[df['score'] <= CRITICAL_MAX_SCORE]
    favourable = df[df['score'] >= FAVOURABLE_MIN_SCORE]

    return ExecutiveSummary(
        total_reviews=len(df),
        avg_rating=float(df['score'].mean()) if len(df) else 0.0,
        critical_count=len(critical),
        favourable_count=len(favourable),
        issues=extract_themes(critical, favourable, text_col),
        likes=extract_themes(favourable, critical, text_col),
    )


def format_headline(summary):
    """One-paragraph markdown headline for the summary."""
    text = (
        f"Based on **{summary.total_reviews}** reviews (average **{summary.avg_rating:.1f}★**), "
        f"**{summary.critical_percent:.0f}%** are critical (1–2★) and "
        f"**{summary.favourable_percent:.0f}%** are favourable (4–5★)."
    )
    if summary.issues:
        top = summary.issues[0]
        text += f" The biggest complaint is **{top.phrase}**, raised in {top.share:.0%} of critical reviews."
    if summary.likes:
        top = summary.likes[0]
        text += f" Users most appreciate **{top.phrase}**, mentioned in {top.share:.0%} of favourable reviews."
    return text
