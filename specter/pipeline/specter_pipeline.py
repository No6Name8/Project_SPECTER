"""
SPECTER End-to-End Pipeline.

Chains every component in order:

    IQ samples (np.ndarray)
        │
        ▼
    SpecterCNN          → raw logits (batch, 24)
        │
        ▼
    EnergyOpenSetDetector → class_id, confidence, is_unknown
        │
        ▼
    ThreatScorer        → score (0-100), level, breakdown
        │
        ▼  [optional]
    TDOASimulator       → estimated_location, confidence_radius
        │
        ▼
    SPECTERResult

Usage (inference):
    pipeline = SPECTERPipeline("specter.pt", "threshold.json")
    result   = pipeline.run(iq_array, snr_db=-12.0, burst_duration_ms=35.0)

Usage (batch):
    results = pipeline.process_batch(signal_list)

Usage (demo, no trained model needed):
    demo_run()
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

# Resolve sibling packages regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataset          import get_class_names
from models.specter_cnn    import SpecterCNN
from models.open_set       import EnergyOpenSetDetector
from scoring.threat_scorer import ThreatScorer
from geolocation.tdoa_simulator import TDOASimulator


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SPECTERResult:
    """
    Complete output for a single signal observation passed through the pipeline.
    """

    # Classification
    modulation:   str    # detected modulation name, or "UNKNOWN"
    confidence:   float  # model confidence [0, 1]
    is_unknown:   bool   # True → open-set detector rejected the signal

    # Threat assessment
    threat_level:    str    # "LOW" | "MEDIUM" | "CRITICAL"
    threat_score:    float  # 0–100
    score_breakdown: dict   # component name → score (0–1)

    # Geolocation (None when use_tdoa=False or TDOA not triggered)
    estimated_location:     Optional[tuple[float, float]] = None
    location_confidence_m:  Optional[float]               = None

    # Metadata
    snr_db:            float = 0.0
    burst_duration_ms: float = 100.0
    inference_time_ms: float = 0.0

    def __str__(self) -> str:
        loc = (
            f"({self.estimated_location[0]:.1f}, {self.estimated_location[1]:.1f}) m"
            f"  ±{self.location_confidence_m:.0f} m"
            if self.estimated_location else "N/A"
        )
        bar = "█" * int(self.threat_score // 5)
        lines = [
            "─" * 52,
            f"  SPECTER RESULT",
            "─" * 52,
            f"  Modulation  : {self.modulation}",
            f"  Confidence  : {self.confidence:.3f}",
            f"  Unknown     : {self.is_unknown}",
            f"  SNR         : {self.snr_db:.1f} dB",
            f"  Burst       : {self.burst_duration_ms:.1f} ms",
            f"  Threat      : [{self.threat_level}]  {self.threat_score:.1f}/100  {bar}",
            f"  Location    : {loc}",
            f"  Latency     : {self.inference_time_ms:.1f} ms",
            "  Breakdown:",
        ]
        for k, v in self.score_breakdown.items():
            lines.append(f"    {k:<22} {v:.3f}")
        lines.append("─" * 52)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SPECTERPipeline:
    """
    End-to-end SPECTER signal intelligence pipeline.

    Args:
        model_path     : path to a saved SpecterCNN checkpoint (.pt file).
                         The file must have been saved with torch.save(model.state_dict(), ...).
        threshold_path : path to the open-set threshold JSON written by
                         EnergyOpenSetDetector.save_threshold().
        use_tdoa       : if True, run TDOASimulator on every signal flagged
                         MEDIUM or higher.  Defaults to True.
        device         : "cuda", "cpu", or None (auto-detect).
        num_classes    : number of modulation classes (24 for RadioML 2018).
        input_len      : IQ sample length expected by the model (128).
    """

    def __init__(
        self,
        model_path:     str,
        threshold_path: str,
        use_tdoa:       bool = True,
        device:         Optional[str] = None,
        num_classes:    int = 24,
        input_len:      int = 128,
    ):
        # --- device ----------------------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # --- classifier ------------------------------------------------------
        self.model = SpecterCNN(num_classes=num_classes, input_len=input_len)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # --- open-set detector -----------------------------------------------
        self.detector = EnergyOpenSetDetector()
        self.detector.load_threshold(threshold_path)

        # --- threat scorer ---------------------------------------------------
        self.scorer = ThreatScorer()

        # --- TDOA geolocation (simulated) ------------------------------------
        self.use_tdoa = use_tdoa
        self.tdoa = TDOASimulator() if use_tdoa else None

        # --- class name lookup -----------------------------------------------
        self.class_names = get_class_names()

    # ------------------------------------------------------------------
    # Single-signal inference
    # ------------------------------------------------------------------

    def run(
        self,
        iq_samples:        np.ndarray,
        snr_db:            float,
        burst_duration_ms: float = 100.0,
    ) -> SPECTERResult:
        """
        Run the full SPECTER pipeline on a single IQ sample frame.

        Args:
            iq_samples        : np.ndarray of shape (2, input_len) — channels-first IQ.
                                Values should be normalised (float32).
            snr_db            : estimated SNR of the signal in dB.
            burst_duration_ms : observed burst duration in milliseconds.

        Returns:
            SPECTERResult with all pipeline outputs populated.
        """
        t_start = time.perf_counter()

        # --- preprocess: (2, L) → (1, 2, L) tensor --------------------------
        x = torch.from_numpy(iq_samples.astype(np.float32)).unsqueeze(0)
        x = x.to(self.device, non_blocking=True)

        # --- classify --------------------------------------------------------
        with torch.no_grad():
            logits = self.model(x)          # (1, 24)

        # --- open-set check --------------------------------------------------
        open_preds = self.detector.predict(logits)
        pred       = open_preds[0]

        modulation = (
            "UNKNOWN" if pred.is_unknown
            else self.class_names[pred.class_id]
        )

        # Softmax confidence — use the max probability as human-readable score
        confidence = float(torch.softmax(logits, dim=1).max().item())

        # --- threat scoring --------------------------------------------------
        threat = self.scorer.score(
            classification_confidence=confidence,
            snr_db=snr_db,
            is_unknown=pred.is_unknown,
            burst_duration_ms=burst_duration_ms,
        )

        # --- TDOA geolocation (only on non-trivial threats) ------------------
        estimated_location    = None
        location_confidence_m = None

        if self.use_tdoa and self.tdoa and threat.level in ("MEDIUM", "CRITICAL"):
            # In a real system the transmitter position would be unknown —
            # here we simulate a detection at a plausible position and run
            # the estimator on the noisy outputs.
            sim = self.tdoa.run_simulation(
                true_transmitter_pos=(250.0, 300.0),   # placeholder: unknown in real ops
                noise_std_meters=100.0,
                visualize=False,
            )
            estimated_location    = sim["estimated_pos"]
            location_confidence_m = sim["confidence_radius"]

        t_end = time.perf_counter()

        return SPECTERResult(
            modulation=modulation,
            confidence=confidence,
            is_unknown=pred.is_unknown,
            threat_level=threat.level,
            threat_score=threat.score,
            score_breakdown=threat.breakdown,
            estimated_location=estimated_location,
            location_confidence_m=location_confidence_m,
            snr_db=snr_db,
            burst_duration_ms=burst_duration_ms,
            inference_time_ms=(t_end - t_start) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def process_batch(
        self,
        signals: list[dict],
    ) -> list[SPECTERResult]:
        """
        Score a list of signals.

        Each element of `signals` is a dict with keys:
            iq_samples        (required) : np.ndarray (2, input_len)
            snr_db            (required) : float
            burst_duration_ms (optional) : float, default 100.0

        Args:
            signals: list of signal dicts (order is preserved)

        Returns:
            List of SPECTERResult in the same order.

        Example:
            signals = [
                {"iq_samples": arr1, "snr_db": -12.0, "burst_duration_ms": 40.0},
                {"iq_samples": arr2, "snr_db":   4.0},
            ]
            results = pipeline.process_batch(signals)
        """
        results = []
        for sig in signals:
            results.append(self.run(
                iq_samples=sig["iq_samples"],
                snr_db=sig["snr_db"],
                burst_duration_ms=sig.get("burst_duration_ms", 100.0),
            ))
        return results


# ---------------------------------------------------------------------------
# Demo  — no trained model required
# ---------------------------------------------------------------------------

def demo_run() -> None:
    """
    Demonstrate the full SPECTER pipeline using a synthetic IQ signal and
    a mock model/detector so no trained weights are needed.

    Simulates three scenarios:
        1. High-SNR known signal   → LOW threat
        2. Low-SNR unknown signal  → CRITICAL threat  (TDOA triggered)
        3. Short-burst low-SNR     → CRITICAL threat  (TDOA triggered)
    """
    print("\n" + "═" * 52)
    print("  SPECTER  —  Pipeline Demo (synthetic signals)")
    print("═" * 52)

    # --- build a mock pipeline without weights on disk ----------------------
    class _MockModel(nn.Module):
        """Returns deterministic logits for demo purposes."""
        def __init__(self, num_classes=24):
            super().__init__()
            self._dummy = nn.Linear(1, 1)   # keeps torch happy
            self.num_classes = num_classes

        def forward(self, x):
            batch = x.size(0)
            # High confidence on class 3 (BPSK)
            logits = torch.full((batch, self.num_classes), -5.0)
            logits[:, 3] = 5.0
            return logits

    class _MockLowConfModel(nn.Module):
        """Flat logits → maximum uncertainty → open-set rejection."""
        def __init__(self, num_classes=24):
            super().__init__()
            self._dummy = nn.Linear(1, 1)
            self.num_classes = num_classes

        def forward(self, x):
            return torch.zeros(x.size(0), self.num_classes)

    device = torch.device("cpu")
    scorer = ThreatScorer()
    tdoa   = TDOASimulator()

    def _run_demo_signal(
        label:             str,
        model:             nn.Module,
        snr_db:            float,
        burst_duration_ms: float,
        force_unknown:     bool = False,
    ) -> None:
        """Run one scenario and print the result."""
        print(f"\n── Scenario: {label}")

        iq = np.random.randn(2, 128).astype(np.float32) * 0.1

        x = torch.from_numpy(iq).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)

        # Manually build a minimal open-set result for the demo
        from models.open_set import OpenSetPrediction
        probs      = torch.softmax(logits, dim=1)
        class_id   = int(probs.argmax().item())
        confidence = float(probs.max().item())
        is_unknown = force_unknown or (confidence < 0.10)

        modulation = "UNKNOWN" if is_unknown else get_class_names()[class_id]

        threat = scorer.score(
            classification_confidence=confidence,
            snr_db=snr_db,
            is_unknown=is_unknown,
            burst_duration_ms=burst_duration_ms,
        )

        estimated_location    = None
        location_confidence_m = None
        if threat.level in ("MEDIUM", "CRITICAL"):
            sim = tdoa.run_simulation(
                true_transmitter_pos=(250.0, 300.0),
                noise_std_meters=100.0,
                visualize=False,
            )
            estimated_location    = sim["estimated_pos"]
            location_confidence_m = sim["confidence_radius"]

        result = SPECTERResult(
            modulation=modulation,
            confidence=confidence,
            is_unknown=is_unknown,
            threat_level=threat.level,
            threat_score=threat.score,
            score_breakdown=threat.breakdown,
            estimated_location=estimated_location,
            location_confidence_m=location_confidence_m,
            snr_db=snr_db,
            burst_duration_ms=burst_duration_ms,
        )
        print(result)

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    _run_demo_signal(
        label="High-SNR known signal (BPSK @ +20 dB, 200 ms burst)",
        model=_MockModel(),
        snr_db=20.0,
        burst_duration_ms=200.0,
    )

    _run_demo_signal(
        label="Low-SNR unknown signal (-18 dB, 100 ms burst)",
        model=_MockLowConfModel(),
        snr_db=-18.0,
        burst_duration_ms=100.0,
        force_unknown=True,
    )

    _run_demo_signal(
        label="Very short burst, low SNR (-14 dB, 20 ms burst)",
        model=_MockLowConfModel(),
        snr_db=-14.0,
        burst_duration_ms=20.0,
        force_unknown=True,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_run()
