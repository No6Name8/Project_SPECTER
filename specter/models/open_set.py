"""
Open-set recognition for SPECTER.

Two detectors are provided so results are directly comparable:

  EnergyOpenSetDetector   — Energy Score (Liu et al., NeurIPS 2020)
                            E(x) = -log Σ exp(fk(x))
                            Lower energy → known class
                            Higher energy → UNKNOWN

  SoftmaxThresholdDetector — MSP baseline (Hendrycks & Gimpel, ICLR 2017)
                             Rejects when max softmax probability < threshold

Both share the same predict() and AUROC evaluation interface so they can
be swapped without changing downstream pipeline code.

Reference:
  Liu, W., Wang, X., Owens, J., Li, Y. (2020).
  "Energy-based Out-of-distribution Detection."
  NeurIPS 2020.  arXiv:2010.03759
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class OpenSetPrediction:
    """Single-sample open-set prediction."""
    class_id:   int    # predicted class index (−1 when is_unknown=True)
    confidence: float  # score used for the decision (higher = more confident known)
    is_unknown: bool   # True → rejected as out-of-distribution


# ---------------------------------------------------------------------------
# Energy score helper
# ---------------------------------------------------------------------------

def energy_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the Energy Score for a batch of logits.

    E(x) = -log Σ_k exp(f_k(x))
           = -logsumexp(logits)

    Args:
        logits: (batch, num_classes)  raw model output — NO softmax applied
    Returns:
        energy: (batch,)  more negative = more in-distribution
    """
    return -torch.logsumexp(logits, dim=1)


# ---------------------------------------------------------------------------
# Energy-based open-set detector
# ---------------------------------------------------------------------------

class EnergyOpenSetDetector:
    """
    Open-set detector based on the Energy Score.

    Workflow:
        detector = EnergyOpenSetDetector()
        detector.fit(known_loader, model, device)   # sets rejection threshold
        pred = detector.predict(logits)             # per-sample decision
        auroc = eval_auroc(known_logits, unk_logits, detector="energy")
    """

    def __init__(self, percentile: float = 95.0):
        """
        Args:
            percentile: energy percentile on known-class data used as the
                        rejection threshold.  95th percentile means 5% of
                        known samples will be incorrectly flagged as unknown.
        """
        self.percentile = percentile
        self.threshold: float | None = None   # set by fit()

    # ------------------------------------------------------------------
    def fit(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device: torch.device,
    ) -> float:
        """
        Compute the energy distribution on known-class samples and set the
        rejection threshold at self.percentile of that distribution.

        Args:
            loader : DataLoader over known-class samples, yields (x, label, snr)
            model  : SpecterCNN (or any model outputting raw logits)
            device : cuda or cpu

        Returns:
            threshold (float) — stored in self.threshold
        """
        model.eval()
        energies = []

        with torch.no_grad():
            for x, _, _ in loader:
                x      = x.to(device, non_blocking=True)
                logits = model(x)
                e      = energy_score(logits)   # (batch,)
                energies.append(e.cpu())

        all_energies = torch.cat(energies).numpy()  # (N,)

        # Threshold: 95th percentile of known-class energies.
        # Samples with energy ABOVE this are treated as unknown.
        self.threshold = float(np.percentile(all_energies, self.percentile))
        return self.threshold

    # ------------------------------------------------------------------
    def predict(self, logits: torch.Tensor) -> list[OpenSetPrediction]:
        """
        Classify a batch and flag out-of-distribution samples.

        Args:
            logits: (batch, num_classes)  raw model output

        Returns:
            List of OpenSetPrediction, one per sample in the batch.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")

        energies    = energy_score(logits)              # (batch,)
        probs       = torch.softmax(logits, dim=1)
        class_ids   = probs.argmax(dim=1)               # (batch,)
        # Confidence: negate energy so higher = more confident known
        confidences = -energies

        results = []
        for i in range(logits.size(0)):
            e          = energies[i].item()
            is_unknown = e > self.threshold
            results.append(OpenSetPrediction(
                class_id   = -1 if is_unknown else int(class_ids[i].item()),
                confidence = float(confidences[i].item()),
                is_unknown = is_unknown,
            ))
        return results

    # ------------------------------------------------------------------
    def save_threshold(self, path: str) -> None:
        """Persist the fitted threshold to a JSON file."""
        if self.threshold is None:
            raise RuntimeError("Call fit() before save_threshold().")
        with open(path, "w") as f:
            json.dump({"detector": "energy",
                       "percentile": self.percentile,
                       "threshold": self.threshold}, f, indent=2)

    def load_threshold(self, path: str) -> None:
        """Load a previously saved threshold from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.percentile = data["percentile"]
        self.threshold  = data["threshold"]


# ---------------------------------------------------------------------------
# Softmax threshold baseline (MSP — Hendrycks & Gimpel 2017)
# ---------------------------------------------------------------------------

class SoftmaxThresholdDetector:
    """
    Baseline open-set detector using Maximum Softmax Probability (MSP).

    Rejects a sample when max_k softmax(logits)_k < threshold.
    Lower MSP = less confident = more likely unknown.

    Fit sets the threshold at the (100 - percentile)th percentile of
    known-class MSP values so the same FPR budget is used as the energy
    detector, making AUROC curves directly comparable.
    """

    def __init__(self, percentile: float = 95.0):
        """
        Args:
            percentile: MSP percentile on known data below which samples are
                        rejected.  5th percentile → 5% false-positive rate.
        """
        self.percentile = percentile
        self.threshold: float | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device: torch.device,
    ) -> float:
        """
        Set rejection threshold from the known-class MSP distribution.

        Returns:
            threshold (float) — stored in self.threshold
        """
        model.eval()
        msps = []

        with torch.no_grad():
            for x, _, _ in loader:
                x    = x.to(device, non_blocking=True)
                probs = torch.softmax(model(x), dim=1)
                msps.append(probs.max(dim=1).values.cpu())

        all_msps = torch.cat(msps).numpy()

        # Reject below the (100 - percentile)th percentile of known MSPs
        # so the in-distribution acceptance rate matches the energy detector
        self.threshold = float(np.percentile(all_msps, 100.0 - self.percentile))
        return self.threshold

    # ------------------------------------------------------------------
    def predict(self, logits: torch.Tensor) -> list[OpenSetPrediction]:
        """
        Classify a batch and flag low-confidence samples as unknown.

        Args:
            logits: (batch, num_classes)  raw model output

        Returns:
            List of OpenSetPrediction, one per sample.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")

        probs       = torch.softmax(logits, dim=1)
        max_probs   = probs.max(dim=1).values         # (batch,)
        class_ids   = probs.argmax(dim=1)             # (batch,)

        results = []
        for i in range(logits.size(0)):
            msp        = float(max_probs[i].item())
            is_unknown = msp < self.threshold
            results.append(OpenSetPrediction(
                class_id   = -1 if is_unknown else int(class_ids[i].item()),
                confidence = msp,
                is_unknown = is_unknown,
            ))
        return results

    # ------------------------------------------------------------------
    def save_threshold(self, path: str) -> None:
        if self.threshold is None:
            raise RuntimeError("Call fit() before save_threshold().")
        with open(path, "w") as f:
            json.dump({"detector": "softmax_msp",
                       "percentile": self.percentile,
                       "threshold": self.threshold}, f, indent=2)

    def load_threshold(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.percentile = data["percentile"]
        self.threshold  = data["threshold"]


# ---------------------------------------------------------------------------
# AUROC evaluation
# ---------------------------------------------------------------------------

def _collect_energy_scores(
    logits_list: list[torch.Tensor],
) -> np.ndarray:
    """Concatenate a list of logit tensors and return numpy energy scores."""
    all_logits = torch.cat(logits_list, dim=0)
    return energy_score(all_logits).numpy()


def _collect_msp_scores(
    logits_list: list[torch.Tensor],
) -> np.ndarray:
    all_logits = torch.cat(logits_list, dim=0)
    probs = torch.softmax(all_logits, dim=1)
    return probs.max(dim=1).values.numpy()


def eval_auroc(
    known_logits: torch.Tensor,
    unknown_logits: torch.Tensor,
    detector: str = "energy",
) -> float:
    """
    Compute AUROC for separating known-class from unknown-class samples.

    Convention:
        label 1 = in-distribution (known)
        label 0 = out-of-distribution (unknown)

    For the energy detector the score used is the NEGATED energy
    (so higher score = more likely known), making AUROC interpretation
    consistent: AUROC > 0.5 means the detector is better than chance.

    Args:
        known_logits  : (N_known,   num_classes)  logits from known-class samples
        unknown_logits: (N_unknown, num_classes)  logits from unknown-class samples
        detector      : "energy" (default) or "softmax"

    Returns:
        AUROC (float, 0–1).  Perfect separation = 1.0, random = 0.5.
    """
    if detector == "energy":
        # Energy: lower = known, so confidence = -energy (higher = known)
        known_scores   = -energy_score(known_logits).numpy()
        unknown_scores = -energy_score(unknown_logits).numpy()
    elif detector == "softmax":
        # MSP: higher = known
        known_scores   = torch.softmax(known_logits,   dim=1).max(dim=1).values.numpy()
        unknown_scores = torch.softmax(unknown_logits, dim=1).max(dim=1).values.numpy()
    else:
        raise ValueError(f"Unknown detector type '{detector}'. Use 'energy' or 'softmax'.")

    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.ones(len(known_scores),   dtype=int),
        np.zeros(len(unknown_scores), dtype=int),
    ])

    return float(roc_auc_score(labels, scores))


def eval_auroc_from_loaders(
    known_loader: torch.utils.data.DataLoader,
    unknown_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device,
    detector: str = "energy",
) -> float:
    """
    Convenience wrapper: collect logits from DataLoaders then compute AUROC.

    Args:
        known_loader  : DataLoader over known-class test samples
        unknown_loader: DataLoader over held-out (unknown) class samples
        model         : trained SpecterCNN
        device        : cuda or cpu
        detector      : "energy" or "softmax"

    Returns:
        AUROC (float, 0–1)
    """
    model.eval()

    def _gather(loader):
        parts = []
        with torch.no_grad():
            for x, _, _ in loader:
                parts.append(model(x.to(device)).cpu())
        return torch.cat(parts, dim=0)

    known_logits   = _gather(known_loader)
    unknown_logits = _gather(unknown_loader)

    return eval_auroc(known_logits, unknown_logits, detector=detector)
