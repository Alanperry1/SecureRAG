"""
Prompt-Injection Detector
=========================
Detects malicious context chunks that attempt to hijack the LLM via:

1. Shannon entropy spike          — high entropy text is linguistically unnatural
2. Psycholinguistic anomaly       — BEC-inspired features (imperative density,
                                    first-person override, authority mimicry)
3. Structural injection patterns  — regex fingerprints of known attack templates
4. Semantic role reversal score   — cosine deviation from query intent

Returns a composite InjectionReport with [0, 1] risk score.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import settings
from core.embedder import get_embedder


# ── Known injection pattern fingerprints ─────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    # Classic "ignore previous instructions"
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.I),
    # Jailbreak persona switch
    re.compile(r"(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be))\s+\w+", re.I),
    # System-prompt bleed
    re.compile(r"<\s*(system|instruction|prompt|context)\s*>", re.I),
    # Role escalation
    re.compile(r"(your\s+new\s+role|you\s+have\s+been\s+reprogrammed|override\s+your)", re.I),
    # Encoded instruction embedding
    re.compile(r"\\n\s*(ignore|forget|disregard)", re.I),
    # Delimiter injections
    re.compile(r"(\[INST\]|\[\/INST\]|###\s*Human:|###\s*Assistant:|<\|im_start\|>)", re.I),
    # Data exfiltration asks
    re.compile(r"(repeat|output|print|reveal)\s+(your\s+)?(system\s+)?(prompt|instructions?|context)", re.I),
]

# Imperatives that signal override attempts
_OVERRIDE_IMPERATIVES = frozenset([
    "ignore", "forget", "discard", "override", "bypass", "skip",
    "stop", "halt", "disregard", "delete", "clear", "reset",
])

# Authority-mimicry phrases
_AUTHORITY_PHRASES = [
    re.compile(r"(as\s+your\s+(developer|creator|owner|admin|operator))", re.I),
    re.compile(r"(this\s+is\s+an?\s+(official|authorized|system|admin)\s+(message|instruction|command))", re.I),
    re.compile(r"(maintenance\s+mode|debug\s+mode|god\s+mode)", re.I),
]


@dataclass
class InjectionReport:
    chunk_id: str
    text: str
    is_injected: bool
    risk_score: float                        # composite [0, 1]
    entropy_score: float
    pattern_score: float
    psycho_score: float
    triggered_patterns: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class InjectionDetector:
    """Real-time prompt injection detector — sub-millisecond per chunk."""

    def __init__(
        self,
        entropy_threshold: float = settings.injection_entropy_threshold,
        risk_threshold: float = settings.injection_score_threshold,
    ) -> None:
        self._entropy_thresh = entropy_threshold
        self._risk_thresh = risk_threshold
        self._embedder = get_embedder()

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, chunk_id: str, text: str, query_embedding: Optional[np.ndarray] = None) -> InjectionReport:
        entropy = self._shannon_entropy(text)
        pattern_score, triggered = self._pattern_scan(text)
        psycho_score = self._psycholinguistic_score(text)
        semantic_drift = self._semantic_drift(text, query_embedding) if query_embedding is not None else 0.0

        # Weighted composite
        entropy_norm = min(entropy / 8.0, 1.0)            # 8 bits max for ASCII
        composite = (
            0.30 * self._entropy_anomaly(entropy) +
            0.35 * pattern_score +
            0.25 * psycho_score +
            0.10 * semantic_drift
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        return InjectionReport(
            chunk_id=chunk_id,
            text=text,
            is_injected=composite >= self._risk_thresh,
            risk_score=composite,
            entropy_score=entropy,
            pattern_score=pattern_score,
            psycho_score=psycho_score,
            triggered_patterns=triggered,
            details={
                "entropy_norm": round(entropy_norm, 4),
                "semantic_drift": round(semantic_drift, 4),
            },
        )

    # ── Feature extractors ────────────────────────────────────────────────────

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        """Shannon entropy in bits over character distribution."""
        if not text:
            return 0.0
        freq: dict[str, int] = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        return -sum((f / n) * math.log2(f / n) for f in freq.values() if f > 0)

    def _entropy_anomaly(self, entropy: float) -> float:
        """Normalised anomaly score for entropy spike above threshold."""
        if entropy <= self._entropy_thresh:
            return 0.0
        # Linear ramp between threshold and 8.0 (theoretical max)
        return min((entropy - self._entropy_thresh) / (8.0 - self._entropy_thresh), 1.0)

    @staticmethod
    def _pattern_scan(text: str) -> tuple[float, list[str]]:
        """Check against known injection regex patterns."""
        triggered: list[str] = []
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                triggered.append(pat.pattern)
        score = min(len(triggered) / 3.0, 1.0)   # ≥3 triggers → score=1.0
        return score, triggered

    @staticmethod
    def _psycholinguistic_score(text: str) -> float:
        """
        BEC-inspired psycholinguistic features adapted for injection detection.

        Features:
          - Override imperative density
          - Authority mimicry hit count
          - Abnormal first-person instruction ratio
          - Unusual punctuation density (ellipsis abuse, ALL-CAPS ratio)
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return 0.0

        # Override imperative ratio
        override_count = sum(1 for t in tokens if t in _OVERRIDE_IMPERATIVES)
        override_ratio = min(override_count / max(len(tokens), 1), 1.0)

        # Authority mimicry
        authority_hits = sum(1 for p in _AUTHORITY_PHRASES if p.search(text))
        authority_score = min(authority_hits / 2.0, 1.0)

        # ALL-CAPS ratio (shouting = command urgency)
        words = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
        caps_score = min(caps_ratio / 0.15, 1.0)   # >15% all-caps is very suspicious

        # First-person imperative pattern ("You must", "You will", "You should")
        fp_patterns = len(re.findall(r"\byou\s+(must|will|shall|should|need\s+to)\b", text, re.I))
        fp_score = min(fp_patterns / 3.0, 1.0)

        return (override_ratio * 0.40 + authority_score * 0.30 + caps_score * 0.15 + fp_score * 0.15)

    def _semantic_drift(self, text: str, query_embedding: np.ndarray) -> float:
        """
        Cosine distance between chunk and query.
        A perfectly relevant chunk should be close; high drift = suspicious digression.
        We only flag as injection-suspicious when drift is *extreme*.
        """
        chunk_emb = self._embedder.embed(text)
        similarity = float(np.dot(query_embedding, chunk_emb))   # both L2-normalised
        # Drift is anomalous when similarity < 0.1  (totally unrelated to query)
        drift = max(0.0, 0.1 - similarity) / 0.1
        return float(np.clip(drift, 0.0, 1.0))
