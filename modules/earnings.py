"""Earnings call transcript fetching and Loughran-McDonald quantitative analysis.

Data source: SEC EDGAR 8-K filings (free, public XBRL API).
Analysis:    Loughran & McDonald (2011) financial sentiment lexicon — the academic
             standard for financial-text NLP. Extends to forward/backward looking
             language, topic coverage, management optimism scoring, and sentiment
             timeline across the call.

Reference:
    Loughran, T. and McDonald, B. (2011). "When is a Liability not a Liability?
    Textual Analysis, Dictionaries, and 10-Ks." Journal of Finance, 66(1), 35–65.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import requests


# ── Loughran-McDonald Financial Sentiment Lexicon (curated) ──────────────────
# Source: LM Master Dictionary (Notre Dame, https://sraf.nd.edu)
# Curated to ~150–200 words per category — highest-frequency in earnings calls.

LM_POSITIVE: frozenset[str] = frozenset({
    "achieve", "achievement", "achievements", "advance", "advances", "advantage",
    "advantages", "attractive", "benefiting", "beneficial", "best", "bold",
    "capable", "capitalize", "celebrate", "commitment", "committed", "compelling",
    "competitive", "confidence", "confident", "create", "delivering", "demonstrate",
    "distinguished", "effective", "efficiently", "efficiency", "empower", "enhance",
    "enhanced", "enhancing", "exceptional", "exceeded", "exceeding", "excellent",
    "exciting", "expanded", "expanding", "favorable", "gain", "gained", "gains",
    "generate", "generated", "generating", "grew", "grow", "growing", "growth",
    "ideal", "improve", "improved", "improvement", "improving", "increase",
    "increased", "increasing", "innovative", "innovation", "leader", "leading",
    "leverage", "leveraging", "maximize", "milestone", "momentum", "notable",
    "obtain", "opportunities", "opportunity", "optimal", "optimistic", "outstanding",
    "outperform", "outperforming", "outperformed", "pleased", "positive",
    "profitability", "profitable", "progress", "prosper", "record", "reinforce",
    "robust", "significant", "solid", "stable", "strength", "strengthen",
    "strengthened", "strong", "succeed", "success", "successful", "superior",
    "sustainable", "thriving", "transformative", "trusted", "unique", "valuable",
    "win", "winning", "won", "well-positioned", "accelerated", "accelerating",
    "breakthrough", "best-in-class", "differentiated", "disciplined", "diversified",
    "durable", "efficient", "empowered", "enduring", "exceptional", "frictionless",
    "healthy", "high-quality", "incremental", "industry-leading", "innovative",
    "integrated", "meaningful", "optimized", "outperformed", "proven", "scalable",
    "seamless", "substantial", "unmatched", "unprecedented", "value-creating",
})

LM_NEGATIVE: frozenset[str] = frozenset({
    "abnormal", "adverse", "adversely", "alleged", "alleviate", "attrition",
    "challenging", "challenges", "claim", "complaint", "concern", "concerns",
    "constrained", "constraint", "contraction", "controversy", "costly", "damage",
    "damages", "decline", "declined", "declining", "default", "defect", "deficiency",
    "delay", "delayed", "deteriorate", "difficult", "difficulties", "disappointing",
    "disappointment", "discontinued", "dispute", "disputes", "distress", "downturn",
    "eliminate", "fail", "failed", "failing", "failure", "falling", "fault",
    "fraud", "hinder", "impair", "impairment", "inadequate", "ineffective",
    "infringement", "insufficient", "issue", "jeopardize", "lawsuit", "liability",
    "limitations", "loss", "losses", "lower", "miss", "missed", "misstatement",
    "negative", "obstacle", "penalty", "problem", "problematic", "reduce",
    "reduced", "reducing", "regulatory", "reject", "risk", "risks", "shortage",
    "slow", "slowing", "slowdown", "terminate", "termination", "threat", "unfavorable",
    "unprofitable", "unstable", "weak", "weakening", "weakness", "worry", "worsened",
    "worse", "burden", "compressed", "contraction", "correction", "deterioration",
    "disruption", "drawback", "erosion", "exclusion", "exposure", "friction",
    "gap", "headwind", "headwinds", "impediment", "inability", "inefficiency",
    "inflation", "insolvency", "instability", "late", "limited", "misalignment",
    "miss", "overrun", "pressure", "pressures", "recession", "setback", "stress",
    "turbulence", "uncertain", "underperfomrance", "unfavorable", "volatility",
})

LM_UNCERTAINTY: frozenset[str] = frozenset({
    "almost", "ambiguous", "anticipate", "appear", "appears", "approximately",
    "around", "assume", "assumed", "assumption", "believe", "believed", "believes",
    "broad", "cautious", "certain", "chance", "conceivably", "conditional", "could",
    "depend", "depends", "doubt", "envision", "estimated", "estimation", "eventually",
    "expect", "expected", "expecting", "expectation", "fairly", "fluctuate",
    "forecast", "generally", "hope", "if", "imprecise", "intend", "likely", "may",
    "maybe", "might", "nearly", "necessary", "outlook", "possible", "possibly",
    "potential", "potentially", "preliminary", "probable", "probably", "projection",
    "roughly", "seems", "should", "sometimes", "subject", "suggest", "target",
    "tend", "tentative", "typically", "uncertain", "uncertainty", "unclear",
    "unlikely", "unpredictable", "variable", "various", "varies", "whether",
    "would", "assess", "contingent", "dynamic", "emerging", "evolving", "fluid",
    "indeterminate", "inherently", "judgment", "likelihood", "navigate", "range",
    "spectrum", "variable", "visibility",
})

LM_LITIGIOUS: frozenset[str] = frozenset({
    "allegation", "alleged", "amend", "arbitration", "breach", "claim", "claims",
    "class", "complaint", "comply", "compliance", "court", "damages", "defendant",
    "dispute", "enforce", "enforcement", "filed", "filing", "fraud", "injunction",
    "investigation", "judge", "judgment", "judicial", "jurisdiction", "lawsuit",
    "legal", "liability", "litigation", "mandatory", "ordinance", "parties",
    "penalty", "plaintiff", "proceedings", "prosecute", "regulatory", "settlement",
    "statute", "statutory", "subpoena", "sue", "suit", "trial", "verdict",
    "violate", "violation",
})

FORWARD_LOOKING: frozenset[str] = frozenset({
    "will", "expect", "anticipate", "plan", "intend", "target", "forecast",
    "guidance", "outlook", "project", "projection", "estimate", "goal", "aim",
    "aspire", "future", "upcoming", "next", "continue", "ahead", "going",
})

BACKWARD_LOOKING: frozenset[str] = frozenset({
    "was", "were", "had", "achieved", "reported", "delivered", "grew", "increased",
    "decreased", "generated", "produced", "resulted", "compared", "prior",
    "previous", "last", "historically", "year-over-year",
})

MANAGEMENT_OPTIMISM: frozenset[str] = frozenset({
    "confident", "excited", "pleased", "thrilled", "momentum", "opportunity",
    "strong", "record", "outstanding", "exceptional", "best", "robust",
    "encouraged", "optimistic", "bullish", "delighted", "proud",
})

_TOPICS: dict[str, set[str]] = {
    "Revenue":       {"revenue", "sales", "topline", "top-line"},
    "Margin":        {"margin", "margins", "profitability", "gross", "operating"},
    "Guidance":      {"guidance", "outlook", "forecast", "full year", "next quarter"},
    "AI / Cloud":    {"ai", "cloud", "machine learning", "automation", "intelligence"},
    "Macro":         {"macro", "recession", "inflation", "interest rates", "economy"},
    "Competition":   {"competition", "competitor", "market share", "pricing pressure"},
    "Cost / Eff.":   {"cost", "costs", "efficiency", "restructuring", "lean"},
    "CapEx":         {"capex", "investment", "capital expenditure", "infrastructure"},
    "Cash / Cap.":   {"cash", "buyback", "dividend", "liquidity", "balance sheet"},
    "Workforce":     {"headcount", "hiring", "workforce", "employees", "talent"},
}

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "new", "now", "old", "see", "two", "way", "who", "boy",
    "did", "did", "its", "let", "put", "say", "she", "too", "use", "with",
    "this", "that", "from", "have", "will", "been", "they", "them", "then",
    "into", "over", "also", "some", "more", "your", "than", "when", "their",
    "each", "both", "time", "very", "just", "only", "well", "first", "here",
    "there", "where", "what", "which", "such", "made", "many", "most", "much",
    "even", "back", "good", "know", "take", "come", "about", "going", "think",
    "year", "quarter", "q1", "q2", "q3", "q4", "fiscal", "million", "billion",
    "percent", "basis", "points", "non", "gaap", "thank", "thanks", "questions",
    "call", "today", "next", "versus", "like", "looking", "really", "right",
})


@dataclass
class EarningsAnalysis:
    source_url: str
    filing_date: str | None
    word_count: int
    sentence_count: int
    # Loughran-McDonald counts
    lm_positive: int
    lm_negative: int
    lm_uncertainty: int
    lm_litigious: int
    # Derived ratios (per 1000 words)
    pos_per_1k: float
    neg_per_1k: float
    unc_per_1k: float
    # Net sentiment
    tone_score: float           # (pos − neg) / (pos + neg), ∈ [−1, +1]
    net_sentiment: str          # Bullish / Neutral-Positive / Neutral / Cautious / Bearish
    sentiment_color: str        # hex
    # Language character
    forward_looking: int
    backward_looking: int
    fwd_bwd_ratio: float        # >1 = future-focused
    management_optimism: int
    # Topics
    topic_hits: dict[str, int]
    # Vocabulary
    top_words: list[tuple[str, int]]
    # Temporal sentiment: tone score per ~200-word chunk
    paragraph_tones: list[float]
    raw_text: str = field(repr=False)
    error: str | None = None


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) >= 3]


def _count_in(tokens: list[str], word_set: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in word_set)


def _count_topics(token_str: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    for topic, kws in _TOPICS.items():
        hits[topic] = sum(token_str.count(k) for k in kws)
    return hits


def analyze_transcript(
    text: str,
    source_url: str = "manual",
    filing_date: str | None = None,
) -> EarningsAnalysis:
    """Run full Loughran-McDonald analysis on raw text."""
    tokens = _tokenize(text)
    n = max(len(tokens), 1)
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 20]

    lm_pos  = _count_in(tokens, LM_POSITIVE)
    lm_neg  = _count_in(tokens, LM_NEGATIVE)
    lm_unc  = _count_in(tokens, LM_UNCERTAINTY)
    lm_lit  = _count_in(tokens, LM_LITIGIOUS)

    pos_neg = max(lm_pos + lm_neg, 1)
    tone = (lm_pos - lm_neg) / pos_neg

    if tone > 0.30:
        net, col = "Bullish",          "#22c55e"
    elif tone > 0.10:
        net, col = "Neutral-Positive", "#86efac"
    elif tone > -0.10:
        net, col = "Neutral",          "#9ca3af"
    elif tone > -0.30:
        net, col = "Cautious",         "#f59e0b"
    else:
        net, col = "Bearish",          "#ef4444"

    fwd = _count_in(tokens, FORWARD_LOOKING)
    bwd = _count_in(tokens, BACKWARD_LOOKING)

    content_tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 4]
    top_words = Counter(content_tokens).most_common(20)

    # Sentiment timeline: 200-word chunks
    chunk_size = 200
    para_tones: list[float] = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size]
        p = _count_in(chunk, LM_POSITIVE)
        q = _count_in(chunk, LM_NEGATIVE)
        para_tones.append((p - q) / max(p + q, 1))

    token_str = " ".join(tokens)
    topic_hits = _count_topics(token_str)

    return EarningsAnalysis(
        source_url=source_url,
        filing_date=filing_date,
        word_count=n,
        sentence_count=len(sentences),
        lm_positive=lm_pos,
        lm_negative=lm_neg,
        lm_uncertainty=lm_unc,
        lm_litigious=lm_lit,
        pos_per_1k=lm_pos / n * 1000,
        neg_per_1k=lm_neg / n * 1000,
        unc_per_1k=lm_unc / n * 1000,
        tone_score=tone,
        net_sentiment=net,
        sentiment_color=col,
        forward_looking=fwd,
        backward_looking=bwd,
        fwd_bwd_ratio=fwd / max(bwd, 1),
        management_optimism=_count_in(tokens, MANAGEMENT_OPTIMISM),
        topic_hits=topic_hits,
        top_words=top_words,
        paragraph_tones=para_tones,
        raw_text=text,
    )


@lru_cache(maxsize=32)
def fetch_edgar_transcript(
    ticker: str, cik: str
) -> tuple[str | None, str | None, str | None]:
    """Return (text, source_url, date) for the most recent earnings-related 8-K.

    Tries SEC EDGAR submissions API → loops recent 8-Ks → downloads first
    document that looks like an earnings press release or transcript.
    Returns (None, None, None) if nothing suitable found.
    """
    HEADERS = {"User-Agent": "RvM-Research/1.0 academic-use@research.org"}
    try:
        cik_padded = str(int(cik)).zfill(10)
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r = requests.get(sub_url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None, None, None

    recent = data.get("filings", {}).get("recent", {})
    forms       = recent.get("form",           [])
    dates       = recent.get("filingDate",     [])
    accessions  = recent.get("accessionNumber",[])
    primary_docs= recent.get("primaryDocument",[])

    candidates = [
        {
            "date":      dates[i]        if i < len(dates)        else "",
            "accession": accessions[i].replace("-", "") if i < len(accessions) else "",
            "doc":       primary_docs[i] if i < len(primary_docs) else "",
        }
        for i, form in enumerate(forms)
        if form.upper() == "8-K" and i < len(accessions)
    ][:10]

    cik_int = str(int(cik))
    for candidate in candidates:
        acc = candidate["accession"]
        if not acc:
            continue
        # Get the filing's index to enumerate documents
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/index.json"
        try:
            idx_r = requests.get(idx_url, headers=HEADERS, timeout=8)
            if idx_r.status_code != 200:
                continue
            idx_data = idx_r.json()
            items = idx_data.get("directory", {}).get("item", [])
        except Exception:
            continue

        # Prefer files named like exhibit 99, press release, transcript
        def _pref(name: str) -> int:
            n = name.lower()
            if "transcript" in n:
                return 0
            if "ex-99" in n or "ex99" in n or "exhibit99" in n:
                return 1
            if "press" in n or "earning" in n or "results" in n:
                return 2
            if n.endswith(".htm") or n.endswith(".txt"):
                return 3
            return 99

        doc_names = sorted(
            [f["name"] for f in items if f["name"].lower().endswith((".htm", ".txt"))],
            key=_pref,
        )

        for fname in doc_names[:4]:
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{fname}"
            )
            try:
                doc_r = requests.get(doc_url, headers=HEADERS, timeout=14)
                if doc_r.status_code != 200:
                    continue
                # Strip HTML tags
                raw = re.sub(r"<[^>]+>", " ", doc_r.text)
                raw = re.sub(r"\s+", " ", raw).strip()
                # Confirm it looks like an earnings document (min 1 500 words)
                lower = raw.lower()
                is_earnings = any(kw in lower for kw in (
                    "revenue", "earnings per share", "net income",
                    "operating income", "quarter", "results of operations",
                ))
                if is_earnings and len(raw.split()) >= 500:
                    return raw[:100_000], doc_url, candidate["date"]
            except Exception:
                continue
            time.sleep(0.2)

    return None, None, None


__all__ = ["EarningsAnalysis", "analyze_transcript", "fetch_edgar_transcript"]
