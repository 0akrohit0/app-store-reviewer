"""Render an AIReport to a PDF (reportlab + matplotlib charts). No Streamlit imports."""

from io import BytesIO
import os
import re
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# DejaVu ships with matplotlib and covers ₹, ★ and most Latin/Cyrillic text
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT, FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"
if FONT not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(TTFont(FONT, os.path.join(FONT_DIR, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
_GLYPHS = set(pdfmetrics.getFont(FONT).face.charToGlyph)

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#616e7c")
ACCENT = colors.HexColor("#2457c5")
RULE = colors.HexColor("#d9e2ec")
PRIORITY_COLORS = {"P0": "#c62828", "P1": "#ef6c00", "P2": "#7b8794"}
CHART_BLUE, CHART_RED, CHART_GREEN, CHART_GREY = "#2457c5", "#c62828", "#2e7d32", "#9aa5b1"

MAX_QUOTES_PER_DIVE = 3


def _clean(text):
    """Escape for reportlab markup and drop characters the font can't draw (emoji, some scripts)."""
    text = ''.join(ch for ch in str(text) if ord(ch) in _GLYPHS or ch in '\n\t')
    return escape(re.sub(r'[ \t]+', ' ', text).strip())


def _styles():
    base = getSampleStyleSheet()
    def style(name, parent='BodyText', **kw):
        return ParagraphStyle(name, parent=base[parent], fontName=kw.pop('fontName', FONT), textColor=kw.pop('textColor', INK), **kw)
    return {
        'title': style('T', 'Title', fontName=FONT_BOLD, fontSize=24, leading=30, alignment=0),
        'subtitle': style('ST', fontSize=11, leading=15, textColor=MUTED),
        'h1': style('H1', 'Heading1', fontName=FONT_BOLD, fontSize=16, leading=20, spaceBefore=6, spaceAfter=8, textColor=ACCENT),
        'h2': style('H2', 'Heading2', fontName=FONT_BOLD, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4),
        'h3': style('H3', fontName=FONT_BOLD, fontSize=10, leading=13, spaceBefore=6, spaceAfter=2),
        'body': style('B', fontSize=9.5, leading=13.5, spaceAfter=4),
        'bullet': style('BL', fontSize=9.5, leading=13.5, leftIndent=12, bulletIndent=2, spaceAfter=2),
        'quote': style('Q', fontSize=8.8, leading=12, leftIndent=10, textColor=colors.HexColor("#3e4c59"), spaceAfter=1),
        'quote_meta': style('QM', fontSize=7.5, leading=10, leftIndent=10, textColor=MUTED, spaceAfter=5),
        'small': style('S', fontSize=8, leading=11, textColor=MUTED),
        'cell': style('C', fontSize=8.3, leading=11),
        'cell_bold': style('CB', fontName=FONT_BOLD, fontSize=8.3, leading=11),
        'kpi_value': style('KV', fontName=FONT_BOLD, fontSize=17, leading=21, alignment=TA_CENTER),
        'kpi_label': style('KL', fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=MUTED),
    }


def _pct(x):
    return f"{100 * x:.0f}%"


def _trend(theme):
    delta = theme['recent_share'] - theme['earlier_share']
    if abs(delta) < 0.02:
        return "→ flat"
    return f"{'↑' if delta > 0 else '↓'} {_pct(theme['earlier_share'])} → {_pct(theme['recent_share'])}"


def _chart(fig, width_mm, height_mm):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_mm * mm, height=height_mm * mm)


def _style_axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#cbd2d9')
    ax.spines['bottom'].set_color('#cbd2d9')
    ax.tick_params(colors='#52606d', labelsize=8)


def _rating_chart(report):
    dist = {int(k): v for k, v in report.rating_distribution.items()}
    stars = [1, 2, 3, 4, 5]
    counts = [dist.get(s, 0) for s in stars]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar([f"{s}★" for s in stars], counts,
           color=[CHART_RED, CHART_RED, CHART_GREY, CHART_GREEN, CHART_GREEN], width=0.65)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha='center', va='bottom', fontsize=7, color='#52606d')
    ax.set_title("Rating distribution", fontsize=9, loc='left', color='#1f2933')
    _style_axes(ax)
    return _chart(fig, 82, 53)


def _monthly_chart(report):
    months = [m['month'] for m in report.monthly][-12:]
    ratings = [m['avg_rating'] for m in report.monthly][-12:]
    volume = [m['reviews'] for m in report.monthly][-12:]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(months, volume, color='#e4e7eb', width=0.7)
    ax.set_ylabel("Reviews", fontsize=7, color='#7b8794')
    ax2 = ax.twinx()
    ax2.plot(months, ratings, color=CHART_BLUE, marker='o', markersize=3, linewidth=1.6)
    ax2.set_ylim(1, 5)
    ax2.set_ylabel("Avg rating", fontsize=7, color=CHART_BLUE)
    ax.set_title("Monthly volume and average rating", fontsize=9, loc='left', color='#1f2933')
    _style_axes(ax)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(colors='#52606d', labelsize=7)
    ax.tick_params(axis='x', labelrotation=45, labelsize=6.5)
    return _chart(fig, 82, 53)


def _theme_chart(themes, title, color):
    themes = themes[:10][::-1]
    fig_w, fig_h = 7.0, 0.32 * len(themes) + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    shares = [100 * t['share'] for t in themes]
    ax.barh([t['name'][:60] for t in themes], shares, color=color, height=0.6)
    for i, (s, t) in enumerate(zip(shares, themes, strict=True)):
        ax.text(s + 0.4, i, f"{s:.0f}% ({t['count']})", va='center', fontsize=7, color='#52606d')
    ax.set_xlabel("% of all reviews", fontsize=7, color='#7b8794')
    ax.set_title(title, fontsize=9, loc='left', color='#1f2933')
    _style_axes(ax)
    ax.set_xlim(0, max(shares) * 1.25 if shares else 1)
    return _chart(fig, 170, 170 * fig_h / fig_w)


def _table(rows, widths, header=True):
    table = Table(rows, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
                  ('LINEBELOW', (0, 0), (-1, 0), 0.8, ACCENT)]
    table.setStyle(TableStyle(style))
    return table


def _quote_flowables(quote, s):
    return [
        Paragraph(f"“{_clean(quote['text'])}”", s['quote']),
        Paragraph(f"{quote['score']}★ · {quote['thumbs']} upvotes · {quote['date']}", s['quote_meta']),
    ]


def _on_page(canvas, doc, report):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, f"Review insights · {report.app_id}")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(report):
    """Return the report as PDF bytes."""
    s = _styles()
    syn = report.synthesis
    ov = report.overview
    issues = [t for t in report.themes if t['kind'] == 'issue']
    likes = [t for t in report.themes if t['kind'] == 'like']
    requests = [t for t in report.themes if t['kind'] == 'request']
    story = []

    # --- Cover + executive summary
    story += [
        Paragraph("Review Insights Report", s['title']),
        Paragraph(
            f"{_clean(report.app_id)} · {ov['total_reviews']} reviews from {ov['date_from']} to {ov['date_to']} · "
            f"generated {report.generated_at}", s['subtitle']),
        Spacer(1, 6 * mm),
    ]
    kpis = [
        (str(ov['total_reviews']), "reviews analysed"),
        (f"{ov['avg_rating']:.1f}★", "average rating"),
        (_pct(ov['critical_share']), "critical (1–2★)"),
        (_pct(ov['favourable_share']), "favourable (4–5★)"),
        (_pct(ov['developer_reply_rate']), "developer reply rate"),
    ]
    kpi_table = Table([[Paragraph(v, s['kpi_value']) for v, _ in kpis],
                       [Paragraph(label, s['kpi_label']) for _, label in kpis]],
                      colWidths=[34 * mm] * 5)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f5f7fa")),
        ('LINEAFTER', (0, 0), (-2, -1), 0.6, colors.white),
        ('TOPPADDING', (0, 0), (-1, 0), 8), ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
    ]))
    story += [kpi_table, Spacer(1, 6 * mm)]

    story += [Paragraph("Executive summary", s['h1']),
              Paragraph(f"<b>{_clean(syn['headline'])}</b>", s['body']),
              Paragraph(_clean(syn['executive_summary']), s['body']),
              Paragraph("Key takeaways", s['h2'])]
    story += [Paragraph(_clean(t), s['bullet'], bulletText='•') for t in syn['key_takeaways']]
    story += [Spacer(1, 4 * mm), Table([[_rating_chart(report), _monthly_chart(report)]])]

    # --- Priorities
    story += [PageBreak(), Paragraph("What to work on", s['h1']),
              Paragraph("Issues ranked by the analysis. Reach and trend are computed from every review; "
                        "a review can mention several themes.", s['small']), Spacer(1, 3 * mm)]
    rows = [[Paragraph(h, s['cell_bold']) for h in ("", "Issue", "Reach", "Upvotes", "Trend", "Why")]]
    for p in syn['priorities']:
        theme = report.theme(p['theme_id'])
        if not theme:
            continue
        badge = Paragraph(f"<font color='{PRIORITY_COLORS[p['priority']]}'><b>{p['priority']}</b></font>", s['cell'])
        rows.append([
            badge,
            Paragraph(f"<b>{_clean(theme['name'])}</b>", s['cell']),
            Paragraph(f"{_pct(theme['share'])} ({theme['count']})<br/>{_pct(theme['share_of_critical'])} of 1–2★", s['cell']),
            Paragraph(str(theme['upvotes']), s['cell']),
            Paragraph(_trend(theme), s['cell']),
            Paragraph(_clean(p['rationale']), s['cell']),
        ])
    story += [_table(rows, [10, 40, 24, 16, 24, 60]), Spacer(1, 5 * mm)]
    if issues:
        story.append(_theme_chart(issues, "Issue themes by reach", CHART_RED))

    # --- Deep dives
    story += [PageBreak(), Paragraph("Deep dives", s['h1'])]
    for i, dive in enumerate(syn['deep_dives']):
        theme = report.theme(dive['theme_id'])
        if not theme:
            continue
        priority = next((p['priority'] for p in syn['priorities'] if p['theme_id'] == dive['theme_id']), None)
        block = [
            Paragraph(f"{i + 1}. {_clean(dive['headline'])}" + (f"  <font color='#7b8794'>[{priority}]</font>" if priority else ""), s['h2']),
            Paragraph(
                f"<b>{_clean(theme['name'])}</b> · {theme['count']} reviews ({_pct(theme['share'])}) · "
                f"{theme['upvotes']} upvotes · avg {theme['avg_rating']:.1f}★ · trend {_trend(theme)}", s['small']),
        ]
        block += [Paragraph("What's happening", s['h3']), Paragraph(_clean(dive['what_is_happening']), s['body']),
                  Paragraph("Impact on users and the business", s['h3']), Paragraph(_clean(dive['user_impact']), s['body']),
                  Paragraph("Likely root causes", s['h3'])]
        block += [Paragraph(_clean(c), s['bullet'], bulletText='•') for c in dive['likely_root_causes']]
        story += [KeepTogether(block)]

        rec_rows = [[Paragraph(h, s['cell_bold']) for h in ("Recommendation", "Impact", "Effort")]]
        rec_rows += [[Paragraph(_clean(r['action']), s['cell']), Paragraph(r['impact'], s['cell']), Paragraph(r['effort'], s['cell'])]
                     for r in dive['recommendations']]
        story += [Paragraph("Recommendations", s['h3']), _table(rec_rows, [130, 22, 22]),
                  Paragraph(f"<b>Metric to track:</b> {_clean(dive['metric_to_track'])}", s['body'])]

        quotes = {q['rid']: q for q in theme['quotes']}
        evidence = [quotes[rid] for rid in dive['evidence_review_ids'] if rid in quotes] or theme['quotes']
        story.append(Paragraph("What users say", s['h3']))
        for quote in evidence[:MAX_QUOTES_PER_DIVE]:
            story += _quote_flowables(quote, s)
        story.append(Spacer(1, 4 * mm))

    # --- Likes and requests
    story += [PageBreak(), Paragraph("What users like", s['h1']), Paragraph(_clean(syn['what_users_like']), s['body'])]
    for theme in likes:
        story += [Paragraph(f"{_clean(theme['name'])} · {theme['count']} reviews ({_pct(theme['share'])}), "
                            f"{_pct(theme['share_of_favourable'])} of 4–5★", s['h3'])]
        story += _quote_flowables(theme['quotes'][0], s) if theme['quotes'] else []

    story += [Spacer(1, 4 * mm), Paragraph("Feature requests", s['h1']), Paragraph(_clean(syn['feature_requests_summary']), s['body'])]
    if requests:
        rows = [[Paragraph(h, s['cell_bold']) for h in ("Request", "Reviews", "Upvotes", "Example")]]
        rows += [[Paragraph(f"<b>{_clean(t['name'])}</b>", s['cell']), Paragraph(f"{t['count']} ({_pct(t['share'])})", s['cell']),
                  Paragraph(str(t['upvotes']), s['cell']),
                  Paragraph(f"“{_clean(t['quotes'][0]['text'])}”" if t['quotes'] else "", s['cell'])]
                 for t in requests]
        story.append(_table(rows, [42, 20, 16, 96]))

    story += [Paragraph("Risks and watch-outs", s['h1'])]
    story += [Paragraph(_clean(r), s['bullet'], bulletText='•') for r in syn['risks_and_watchouts']]

    # --- Appendix
    story += [PageBreak(), Paragraph("Appendix: all themes", s['h1'])]
    rows = [[Paragraph(h, s['cell_bold']) for h in ("Theme", "Type", "Reviews", "1–2★", "4–5★", "Trend", "Definition")]]
    rows += [[Paragraph(_clean(t['name']), s['cell']), Paragraph(t['kind'], s['cell']),
              Paragraph(f"{t['count']} ({_pct(t['share'])})", s['cell']), Paragraph(_pct(t['share_of_critical']), s['cell']),
              Paragraph(_pct(t['share_of_favourable']), s['cell']), Paragraph(_trend(t), s['cell']),
              Paragraph(_clean(t['definition']), s['cell'])]
             for t in report.themes]
    story.append(_table(rows, [36, 13, 20, 12, 12, 22, 59]))

    usage = report.usage
    story += [Paragraph("Methodology", s['h2']), Paragraph(_clean(
        f"Reviews were fetched from Google Play. {report.model} proposed the themes from a sample of reviews, then "
        f"tagged all {ov['total_reviews']} reviews with them ({report.unlabeled_reviews} could not be tagged). "
        f"All counts, shares, upvotes and trends were computed in code from those tags; quotes are verbatim. "
        f"Trend compares the share of reviews before and after {ov['trend_cutoff']} ({ov['trend_window']}). "
        f"The sample depends on the fetch settings (sort order, language, country) and is not necessarily "
        f"representative of all users. Analysis cost: about ${usage.get('cost_usd', 0):.2f} "
        f"({usage.get('requests', 0)} API requests)."), s['body'])]
    if report.context:
        story.append(Paragraph(f"<b>Context provided:</b> {_clean(report.context)}", s['body']))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"Review insights: {report.app_id}", author="Play Store Reviews Analyzer")
    doc.build(story, onFirstPage=lambda c, d: _on_page(c, d, report), onLaterPages=lambda c, d: _on_page(c, d, report))
    return buf.getvalue()
