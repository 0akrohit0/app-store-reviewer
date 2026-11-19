import re
from types import SimpleNamespace

import pandas as pd
import pytest

from reviewer import ai_report
from reviewer.ai_report import (
    AIReport, ChunkLabels, ReportError, Synthesis, Taxonomy, compute_stats, generate_report,
    prepare_reviews,
)
from reviewer.pdf_report import build_pdf

TAXONOMY = {
    'themes': [
        {'id': 'verification_stuck', 'name': 'Document verification stuck', 'kind': 'issue',
         'definition': 'Onboarding documents pending for days.'},
        {'id': 'no_support', 'name': 'Support unreachable', 'kind': 'issue',
         'definition': 'Cannot reach a human.'},
        {'id': 'good_earnings', 'name': 'Good earning potential', 'kind': 'like',
         'definition': 'Praise for income.'},
        {'id': 'go_home', 'name': 'Go-home option', 'kind': 'request',
         'definition': 'Asks for orders towards home.'},
        {'id': 'verification_stuck', 'name': 'Duplicate id', 'kind': 'issue', 'definition': 'dupe'},
    ]
}
KEYWORDS = {'verification_stuck': 'verification', 'no_support': 'support',
            'good_earnings': 'earn', 'go_home': 'home'}


def synthesis_payload(evidence_id):
    return {
        'headline': 'Onboarding is broken', 'executive_summary': 'Drivers pay ₹500 and wait.',
        'key_takeaways': ['40% mention verification'],
        'priorities': [
            {'theme_id': 'verification_stuck', 'priority': 'P0', 'rationale': 'Blocks earning.'},
            {'theme_id': 'made_up_theme', 'priority': 'P1', 'rationale': 'Should be dropped.'},
        ],
        'deep_dives': [{
            'theme_id': 'verification_stuck', 'headline': 'Paid, then stuck',
            'what_is_happening': 'Docs pending.', 'user_impact': 'Cannot earn.',
            'likely_root_causes': ['Manual review backlog'],
            'recommendations': [{'action': 'Auto-refund after 10 days', 'impact': 'high', 'effort': 'medium'}],
            'metric_to_track': 'Median time to approval',
            'evidence_review_ids': [evidence_id, 'r99999'],
        }],
        'what_users_like': 'Earnings when orders flow.',
        'feature_requests_summary': 'Go-home mode.',
        'risks_and_watchouts': ['"Scam" language in reviews'],
    }


class FakeStream:
    def __init__(self, text, stop_reason):
        self.message = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type='text', text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=500, cache_creation_input_tokens=0),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    """Answers each step by the requested schema, tagging reviews by keyword."""

    def __init__(self, stop_reason='end_turn'):
        self.calls = []
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs['output_format']
        prompt = kwargs['messages'][0]['content']
        if schema is Taxonomy:
            return FakeStream(Taxonomy(**TAXONOMY).model_dump_json(), self.stop_reason)
        if schema is ChunkLabels:
            labels = []
            for rid, text in re.findall(r'\[(r\d+)\] .*? \| .*? \| .*? \| (.*)', prompt):
                ids = [tid for tid, kw in KEYWORDS.items() if kw in text.lower()]
                labels.append({'review_id': rid, 'theme_ids': ids + ['not_a_theme']})
            labels.append({'review_id': 'r_outside_chunk', 'theme_ids': ['no_support']})
            return FakeStream(ChunkLabels(labels=labels).model_dump_json(), self.stop_reason)
        if schema is Synthesis:
            evidence = re.search(r'"id": "(r\d+)"', prompt).group(1)
            return FakeStream(Synthesis(**synthesis_payload(evidence)).model_dump_json(), self.stop_reason)
        raise AssertionError(schema)


def make_reviews(n=250):
    texts = [
        ('Document verification pending for 20 days after paying fee', 1, 30),
        ('No support, nobody answers the helpline 😡', 1, 5),
        ('Good app, I earn well when orders come', 5, 2),
        ('Please add a go home option', 4, 1),
        ('Verification stuck and no support at all', 1, 12),
    ]
    rows = []
    for i in range(n):
        text, score, thumbs = texts[i % len(texts)]
        rows.append({'reviewId': f'id{i}', 'content': text, 'score': score, 'thumbsUpCount': thumbs,
                     'at': pd.Timestamp('2026-06-01') + pd.Timedelta(days=i % 150),
                     'reviewCreatedVersion': '5.1', 'replyContent': 'Thanks' if i % 50 == 0 else None})
    return pd.DataFrame(rows)


def test_full_pipeline_counts_come_from_code():
    client = FakeClient()
    report = generate_report(make_reviews(), 'com.example.driver', 'Reviewers are drivers', client=client)

    verification = report.theme('verification_stuck')
    support = report.theme('no_support')
    assert verification['count'] == 100  # texts 0 and 4, 50 each
    assert support['count'] == 100
    assert report.theme('good_earnings')['share_of_favourable'] == pytest.approx(0.5)
    assert report.overview['total_reviews'] == 250
    assert report.overview['developer_reply_rate'] == pytest.approx(5 / 250)
    assert report.unlabeled_reviews == 0
    # duplicate theme ids from the model are dropped
    assert len(report.themes) == 4


def test_model_references_are_validated():
    report = generate_report(make_reviews(), 'com.example.driver', client=FakeClient())
    syn = report.synthesis
    assert [p['theme_id'] for p in syn['priorities']] == ['verification_stuck']
    quote_ids = {q['rid'] for q in report.theme('verification_stuck')['quotes']}
    assert syn['deep_dives'][0]['evidence_review_ids']
    assert set(syn['deep_dives'][0]['evidence_review_ids']) <= quote_ids


def test_requests_use_caching_effort_and_fallbacks():
    client = FakeClient()
    generate_report(make_reviews(250), 'app', client=client)
    # 1 taxonomy + 3 classification chunks + 1 synthesis
    assert len(client.calls) == 5
    classify_calls = [c for c in client.calls if c['output_format'] is ChunkLabels]
    assert all(c['system'][0]['cache_control'] == {'type': 'ephemeral'} for c in classify_calls)
    assert len({c['system'][0]['text'] for c in classify_calls}) == 1  # identical prefix → cache hits
    assert all(c['output_config']['effort'] == 'low' for c in classify_calls)
    assert all(c['model'] == ai_report.MODEL and c['fallbacks'] == 'default' for c in client.calls)


def test_usage_and_cost_are_tracked():
    report = generate_report(make_reviews(), 'app', client=FakeClient())
    assert report.usage['requests'] == 5
    assert report.usage['input_tokens'] == 5000
    assert report.usage['cost_usd'] == pytest.approx((5000 * 4 + 1000 * 20 + 2500 * 0.2) / 1e6)


@pytest.mark.parametrize('stop_reason', ['refusal', 'max_tokens'])
def test_refusal_and_truncation_raise_clear_errors(stop_reason):
    with pytest.raises(ReportError, match=stop_reason):
        generate_report(make_reviews(), 'app', client=FakeClient(stop_reason))


def test_too_few_reviews():
    with pytest.raises(ReportError):
        generate_report(make_reviews(5), 'app', client=FakeClient())


def test_prepare_reviews_drops_empty_and_normalizes_whitespace():
    df = make_reviews(3)
    df.loc[1, 'content'] = '   '
    df.loc[2, 'content'] = 'two\n\nlines'
    prepared = prepare_reviews(df)
    assert list(prepared['content']) == [df.loc[0, 'content'], 'two lines']


def test_trend_uses_last_30_days_when_enough_data():
    reviews = prepare_reviews(make_reviews(300))
    stats = compute_stats(reviews, {}, Taxonomy(**TAXONOMY))
    assert stats['overview']['trend_window'] == 'last 30 days'


def test_report_round_trips_and_renders_pdf():
    report = generate_report(make_reviews(), 'com.example.driver', 'drivers', client=FakeClient())
    restored = AIReport.from_json(report.to_json())
    assert restored.synthesis == report.synthesis
    pdf = build_pdf(restored)
    assert pdf.startswith(b'%PDF') and len(pdf) > 20_000


def test_estimate_cost_scales_with_reviews():
    small, large = ai_report.estimate_cost(make_reviews(100)), ai_report.estimate_cost(make_reviews(1000))
    assert 0 < small < large


def test_missing_credentials_raise_report_error(monkeypatch, tmp_path):
    for var in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_PROFILE', 'ANTHROPIC_CONFIG_DIR'):
        monkeypatch.delenv(var, raising=False)
    # An empty home directory has no `ant auth login` profile
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    with pytest.raises(ReportError, match='No Anthropic API key'):
        ai_report.make_client()


def test_explicit_key_builds_client():
    assert ai_report.make_client('sk-ant-test').api_key == 'sk-ant-test'
