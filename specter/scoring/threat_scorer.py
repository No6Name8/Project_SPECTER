"""
SPECTER Threat Scorer.

Converts per-signal classifier outputs into an actionable threat score
on a 0–100 scale, broken down by component so analysts can see exactly
what drove the score.

Scoring formula (weighted sum → normalise to 0-100):

    Component            Weight   Description
    ─────────────────────────────────────────────────────────────────
    uncertainty_score    0.25     1 - classification_confidence
    snr_danger           0.30     linear map: -20 dB→1.0, +30 dB→0.0
    unknown_penalty      0.35     1.0 if open-set detector rejected
    burst_suspicion      0.10     short bursts (<50 ms) score higher

Threat levels:
    LOW      score  < 40
    MEDIUM   score 40–69
    CRITICAL score >= 70

All weights and thresholds live in ThreatScorer attributes so they can
be reconfigured at runtime or subclassed without touching the formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ThreatResult:
    """Threat assessment for a single signal observation."""

    score:     float          # 0–100
    level:     str            # "LOW" | "MEDIUM" | "CRITICAL"
    breakdown: dict           # component name → raw component score (0–1)

    def __str__(self) -> str:
        bar = "█" * int(self.score // 5)
        lines = [f"[{self.level}] score={self.score:.1f}/100  {bar}"]
        for name, val in self.breakdown.items():
            lines.append(f"  {name:<22} {val:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ThreatScorer:
    """
    Stateless threat scorer — all configuration lives on the instance so
    weights and thresholds can be tuned without subclassing.

    Usage:
        scorer = ThreatScorer()
        result = scorer.score(
            classification_confidence=0.42,
            snr_db=-14.0,
            is_unknown=True,
            burst_duration_ms=30.0,
        )
        print(result)

        results = scorer.batch_score(signal_list)
    """

    # --- weights (must sum to 1.0) -----------------------------------------
    W_UNCERTAINTY:   float = 0.25
    W_SNR_DANGER:    float = 0.30
    W_UNKNOWN:       float = 0.35
    W_BURST:         float = 0.10

    # --- SNR danger mapping --------------------------------------------------
    SNR_MIN_DB:  float = -20.0   # maps to danger = 1.0
    SNR_MAX_DB:  float =  30.0   # maps to danger = 0.0

    # --- burst suspicion -----------------------------------------------------
    BURST_SHORT_MS:  float =  50.0   # below this → max suspicion
    BURST_LONG_MS:   float = 500.0   # above this → min suspicion (0.0)

    # --- threat level thresholds ---------------------------------------------
    THRESH_MEDIUM:   float = 40.0
    THRESH_CRITICAL: float = 70.0

    # ------------------------------------------------------------------
    def _uncertainty(self, classification_confidence: float) -> float:
        """Low confidence → high uncertainty score."""
        return 1.0 - max(0.0, min(1.0, classification_confidence))

    def _snr_danger(self, snr_db: float) -> float:
        """
        Linearly map SNR to danger in [0, 1].
        Low SNR signals are harder to characterise and therefore more suspicious.
        """
        danger = (self.SNR_MAX_DB - snr_db) / (self.SNR_MAX_DB - self.SNR_MIN_DB)
        return max(0.0, min(1.0, danger))

    def _unknown_penalty(self, is_unknown: bool) -> float:
        """Binary penalty for open-set rejected signals."""
        return 1.0 if is_unknown else 0.0

    def _burst_suspicion(self, burst_duration_ms: float) -> float:
        """
        Short bursts are more suspicious (harder to intercept, evasion indicator).
        Maps linearly from BURST_SHORT_MS (→ 1.0) to BURST_LONG_MS (→ 0.0).
        """
        if burst_duration_ms <= self.BURST_SHORT_MS:
            return 1.0
        if burst_duration_ms >= self.BURST_LONG_MS:
            return 0.0
        suspicion = (self.BURST_LONG_MS - burst_duration_ms) / (
            self.BURST_LONG_MS - self.BURST_SHORT_MS
        )
        return max(0.0, min(1.0, suspicion))

    def _level(self, score: float) -> str:
        if score >= self.THRESH_CRITICAL:
            return "CRITICAL"
        if score >= self.THRESH_MEDIUM:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    def score(
        self,
        classification_confidence: float,
        snr_db: float,
        is_unknown: bool,
        burst_duration_ms: float = 100.0,
    ) -> ThreatResult:
        """
        Score a single signal observation.

        Args:
            classification_confidence : model confidence in [0, 1]
            snr_db                    : signal SNR in dB (e.g. -14.0)
            is_unknown                : True if open-set detector rejected the signal
            burst_duration_ms         : observed burst length in milliseconds

        Returns:
            ThreatResult with score, level, and per-component breakdown.
        """
        c_uncertainty = self._uncertainty(classification_confidence)
        c_snr         = self._snr_danger(snr_db)
        c_unknown     = self._unknown_penalty(is_unknown)
        c_burst       = self._burst_suspicion(burst_duration_ms)

        raw = (
            self.W_UNCERTAINTY * c_uncertainty
            + self.W_SNR_DANGER  * c_snr
            + self.W_UNKNOWN     * c_unknown
            + self.W_BURST       * c_burst
        )

        # raw is in [0, 1] because weights sum to 1 and each component ∈ [0,1]
        final_score = round(raw * 100.0, 2)

        return ThreatResult(
            score=final_score,
            level=self._level(final_score),
            breakdown={
                "uncertainty_score": round(c_uncertainty, 4),
                "snr_danger":        round(c_snr,         4),
                "unknown_penalty":   round(c_unknown,     4),
                "burst_suspicion":   round(c_burst,       4),
            },
        )

    # ------------------------------------------------------------------
    def batch_score(
        self,
        signals: list[dict],
    ) -> list[ThreatResult]:
        """
        Score a list of signals.

        Each element of `signals` is a dict with keys matching score() args:
            classification_confidence (required)
            snr_db                    (required)
            is_unknown                (required)
            burst_duration_ms         (optional, default 100.0)

        Args:
            signals: list of signal dicts

        Returns:
            List of ThreatResult in the same order as the input.

        Example:
            signals = [
                {"classification_confidence": 0.9, "snr_db": 10.0,  "is_unknown": False},
                {"classification_confidence": 0.3, "snr_db": -18.0, "is_unknown": True, "burst_duration_ms": 25.0},
            ]
            results = scorer.batch_score(signals)
        """
        results = []
        for sig in signals:
            results.append(self.score(
                classification_confidence=sig["classification_confidence"],
                snr_db=sig["snr_db"],
                is_unknown=sig["is_unknown"],
                burst_duration_ms=sig.get("burst_duration_ms", 100.0),
            ))
        return results
