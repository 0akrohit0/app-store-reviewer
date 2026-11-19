import pandas as pd

from reviewer.summary import (
    build_executive_summary,
    extract_themes,
    format_headline,
)


def make_df(rows):
    # rows: (theme_text, score, thumbsUpCount)
    return pd.DataFrame([
        {
            'reviewId': f'r{i}',
            'content': f'Review {i}: {text} and some more words to make it substantive',
            'theme_text': text,
            'score': score,
            'thumbsUpCount': thumbs,
        }
        for i, (text, score, thumbs) in enumerate(rows)
    ])


SAMPLE = make_df(
    [('payment not working', 1, 10)] * 6
    + [('customer care slow', 2, 0)] * 4
    + [('easy use interface', 5, 3)] * 6
    + [('fast delivery', 4, 0)] * 4
    + [('average update', 3, 0)] * 3
)


def test_groups_split_by_star_rating():
    summary = build_executive_summary(SAMPLE)
    assert summary.total_reviews == 23
    assert summary.critical_count == 10
    assert summary.favourable_count == 10


def test_issues_and_likes_land_in_the_right_group():
    summary = build_executive_summary(SAMPLE)
    issue_phrases = [t.phrase for t in summary.issues]
    like_phrases = [t.phrase for t in summary.likes]
    assert 'payment not working' in issue_phrases[0]
    assert any('customer care' in p for p in issue_phrases)
    assert any('easy use' in p for p in like_phrases)
    assert not any('payment' in p for p in like_phrases)


def test_negation_is_kept_in_issue_phrase():
    summary = build_executive_summary(SAMPLE)
    assert 'not working' in summary.issues[0].phrase


def test_overlapping_phrases_are_merged():
    summary = build_executive_summary(SAMPLE)
    # "payment", "payment not", "not working"... all cover the same 6 reviews
    payment_themes = [t for t in summary.issues if t.review_ids & {f'r{i}' for i in range(6)}]
    assert len(payment_themes) == 1
    assert payment_themes[0].review_count == 6
    assert payment_themes[0].upvotes == 60


def test_generic_phrases_are_ignored():
    df = make_df([('bad app', 1, 0)] * 5 + [('good app', 5, 0)] * 5)
    summary = build_executive_summary(df)
    assert summary.issues == []
    assert summary.likes == []


def test_empty_and_missing_groups():
    assert extract_themes(SAMPLE.iloc[0:0], SAMPLE) == []
    only_critical = make_df([('login otp failing', 1, 0)] * 5)
    summary = build_executive_summary(only_critical)
    assert summary.likes == []
    assert summary.issues[0].phrase == 'login otp failing'


def test_small_groups_have_no_themes():
    df = make_df([('login otp failing', 1, 0)] * 4 + [('fast delivery', 5, 0)] * 10)
    summary = build_executive_summary(df)
    assert summary.issues == []
    assert summary.likes


def test_one_viral_review_does_not_make_a_theme():
    df = make_df([('payment not working', 1, 0)] * 10 + [('dark mode missing', 1, 100000)])
    phrases = [t.phrase for t in build_executive_summary(df).issues]
    assert not any('dark mode' in p for p in phrases)


def test_quote_is_the_sentence_mentioning_the_theme():
    df = make_df([('payment not working', 1, 0)] * 5)
    df.loc[0, 'thumbsUpCount'] = 50
    df.loc[0, 'content'] = 'The colours are lovely and bright overall. Sadly my payments are not working since Monday.'
    theme = build_executive_summary(df).issues[0]
    assert theme.quote == 'Sadly my payments are not working since Monday.'


def test_headline_mentions_top_themes():
    headline = format_headline(build_executive_summary(SAMPLE))
    assert '23' in headline
    assert 'payment not working' in headline
    assert 'easy use' in headline


def test_quote_is_truncated():
    long_text = 'word ' * 200
    df = make_df([('battery drain', 1, 0)] * 5)
    df['content'] = long_text
    theme = build_executive_summary(df).issues[0]
    assert len(theme.quote) <= 221
    assert theme.quote.endswith('…')
