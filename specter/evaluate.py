"""
specter/evaluate.py — Full benchmark evaluation for the SPECTER competition.

Run from inside the specter/ folder:
    python evaluate.py
    python evaluate.py --model specter   # skip baseline
    python evaluate.py --data_path /abs/path/to/file.hdf5

Produces:
    results/accuracy_vs_snr.csv
    results/accuracy_vs_snr.png
    results/confusion_baseline.png
    results/confusion_specter.png
    results/open_set_auroc.png
    Console: threat calibration table, latency, final summary table
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — run from specter/ directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.dataset        import MODULATION_CLASSES, get_class_names
from models.baseline_cnn import VT_CNN2
from models.specter_cnn  import SpecterCNN
from models.open_set     import EnergyOpenSetDetector, energy_score, eval_auroc
from scoring.threat_scorer import ThreatScorer

NUM_CLASSES = 24
INPUT_LEN   = 1024
HOLDOUT_IDS = [20, 21, 22, 23]   # AM-DSB-SC, FM, GMSK, OQPSK
KNOWN_IDS   = [i for i in range(NUM_CLASSES) if i not in HOLDOUT_IDS]
LOW_SNR     = (-20, -6)
ALL_SNRS    = list(range(-20, 32, 2))
CLASS_NAMES = get_class_names()


# ============================================================
# Lazy HDF5 Dataset (same pattern as train.py — no full load)
# ============================================================

class LazyHDF5Dataset(torch.utils.data.Dataset):
    """
    Lazy-loading dataset. Y and Z are held in RAM; X is read from disk
    one row at a time per __getitem__.

    Extra arg `only_classes` keeps ONLY those class indices (inverse of
    held_out_classes). Used to build the unknown-only loader for AUROC.
    """

    def __init__(
        self,
        hdf5_path:        str,
        snr_range:        tuple | None = None,
        held_out_classes: list  | None = None,
        only_classes:     list  | None = None,
    ):
        self._path = hdf5_path
        self._file = None

        with h5py.File(hdf5_path, "r") as f:
            Y = f["Y"][:]            # (N, 24) — ~375 MB, OK to keep
            Z = f["Z"][:].squeeze()  # (N,)    — ~16 MB

        labels = np.argmax(Y, axis=1).astype(np.int64)
        del Y

        mask = np.ones(len(labels), dtype=bool)

        if snr_range is not None:
            lo, hi = snr_range
            mask &= (Z >= lo) & (Z <= hi)

        if held_out_classes:
            for c in held_out_classes:
                mask &= labels != int(c)

        if only_classes is not None:
            keep = np.zeros(len(labels), dtype=bool)
            for c in only_classes:
                keep |= labels == int(c)
            mask &= keep

        self._indices = np.where(mask)[0].astype(np.int64)
        self._labels  = torch.from_numpy(labels[mask])
        self._snrs    = torch.from_numpy(Z[mask].astype(np.float32))

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self._path, "r")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        self._open()
        x = self._file["X"][int(self._indices[idx])]          # (1024, 2)
        x = torch.from_numpy(x.T.copy().astype(np.float32))  # (2, 1024)
        return x, self._labels[idx], self._snrs[idx]

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def make_loader(ds: torch.utils.data.Dataset, batch_size: int, shuffle: bool = False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Model loading helpers
# ============================================================

def _try_load_threshold(path: str) -> EnergyOpenSetDetector | None:
    """
    Try path as given, then with .json extension.
    Threshold files are JSON regardless of the extension used.
    """
    candidates = [path, os.path.splitext(path)[0] + ".json"]
    for p in candidates:
        if os.path.isfile(p):
            det = EnergyOpenSetDetector()
            det.load_threshold(p)
            return det
    return None


def load_baseline(path: str, device: torch.device) -> VT_CNN2 | None:
    if not os.path.isfile(path):
        print(f"  [SKIP] Baseline model not found: {path}")
        print("         Run  python train.py --model baseline  first.")
        return None
    m = VT_CNN2(num_classes=NUM_CLASSES, input_len=INPUT_LEN)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    print(f"  Loaded baseline  ← {path}")
    return m


def load_specter(path: str, device: torch.device) -> SpecterCNN | None:
    if not os.path.isfile(path):
        print(f"  [SKIP] SPECTER model not found: {path}")
        print("         Run  python train.py --model specter  first.")
        return None
    m = SpecterCNN(num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    print(f"  Loaded specter   ← {path}")
    return m


# ============================================================
# Shared evaluation primitives
# ============================================================

@torch.no_grad()
def collect_predictions(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    device:      torch.device,
    is_baseline: bool,
    desc:        str = "Evaluating",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over a loader and collect:
        preds  (N,)  — predicted class indices
        labels (N,)  — true class indices
        snrs   (N,)  — SNR values in dB
        logits (N, C)— raw model outputs (NLL log-probs for baseline, logits for specter)
    """
    all_preds  = []
    all_labels = []
    all_snrs   = []
    all_logits = []

    for x, labels, snrs in tqdm(loader, desc=f"  {desc}", leave=False, ncols=100):
        x      = x.to(device, non_blocking=True)
        out    = model(x).cpu()
        preds  = out.argmax(dim=1)

        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())
        all_snrs.append(snrs.numpy())
        all_logits.append(out.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_snrs),
        np.concatenate(all_logits),
    )


# ============================================================
# Section 1 — Per-SNR Accuracy
# ============================================================

def per_snr_accuracy(
    preds:  np.ndarray,
    labels: np.ndarray,
    snrs:   np.ndarray,
) -> dict:
    """Map SNR → accuracy from pre-collected arrays."""
    snr_correct = defaultdict(int)
    snr_total   = defaultdict(int)
    for p, l, s in zip(preds, labels, snrs):
        key = int(round(float(s)))
        snr_correct[key] += int(p == l)
        snr_total[key]   += 1
    return {snr: snr_correct[snr] / snr_total[snr] for snr in sorted(snr_total)}


def save_snr_csv(
    baseline_accs: dict | None,
    specter_accs:  dict | None,
    path:          str,
) -> None:
    all_snrs = sorted(set(
        list(baseline_accs.keys() if baseline_accs else []) +
        list(specter_accs.keys()  if specter_accs  else [])
    ))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", "baseline_acc", "specter_acc"])
        for snr in all_snrs:
            b = f"{baseline_accs[snr]:.6f}" if baseline_accs and snr in baseline_accs else ""
            s = f"{specter_accs[snr]:.6f}"  if specter_accs  and snr in specter_accs  else ""
            writer.writerow([snr, b, s])
    print(f"  Saved: {path}")


def plot_snr_accuracy(
    baseline_accs: dict | None,
    specter_accs:  dict | None,
    path:          str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Shaded low-SNR region
    ax.axvspan(LOW_SNR[0], LOW_SNR[1], color="gold", alpha=0.18, label="Low-SNR region")

    snrs_b = sorted(baseline_accs.keys()) if baseline_accs else []
    snrs_s = sorted(specter_accs.keys())  if specter_accs  else []

    if baseline_accs:
        accs_b = [baseline_accs[s] for s in snrs_b]
        ax.plot(snrs_b, accs_b, color="crimson",    linewidth=2.2,
                marker="o", markersize=4, label="Baseline VT-CNN2", zorder=4)

    if specter_accs:
        accs_s = [specter_accs[s] for s in snrs_s]
        ax.plot(snrs_s, accs_s, color="steelblue",  linewidth=2.2,
                marker="s", markersize=4, label="SPECTER-CNN",      zorder=4)

    # Gap annotation in the low-SNR window
    if baseline_accs and specter_accs:
        low_snrs = [s for s in ALL_SNRS if LOW_SNR[0] <= s <= LOW_SNR[1]]
        gaps = []
        for s in low_snrs:
            if s in specter_accs and s in baseline_accs:
                gaps.append((s, specter_accs[s] - baseline_accs[s]))
        if gaps:
            mid_s, mid_gap = gaps[len(gaps) // 2]
            mid_b = baseline_accs.get(mid_s, 0)
            mid_sp = specter_accs.get(mid_s, 0)
            ax.annotate(
                f"Δ {mid_gap:+.3f}",
                xy=(mid_s, (mid_b + mid_sp) / 2),
                xytext=(mid_s + 3, (mid_b + mid_sp) / 2 + 0.04),
                fontsize=9, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
            )
            ax.annotate("", xy=(mid_s, mid_b), xytext=(mid_s, mid_sp),
                        arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5))

    ax.set_xlabel("SNR (dB)",  fontsize=12)
    ax.set_ylabel("Accuracy",  fontsize=12)
    ax.set_title("Modulation Classification Accuracy vs SNR\nSPECTER vs Baseline VT-CNN2",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(LOW_SNR[0] - 1, 31)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(ALL_SNRS)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Section 2 — Confusion Matrices
# ============================================================

def plot_confusion_matrix(
    preds:      np.ndarray,
    labels:     np.ndarray,
    snrs:       np.ndarray,
    title:      str,
    path:       str,
    snr_range:  tuple = LOW_SNR,
) -> None:
    """Plot normalised confusion matrix filtered to snr_range."""
    lo, hi = snr_range
    mask   = (snrs >= lo) & (snrs <= hi)
    p_filt = preds[mask]
    l_filt = labels[mask]

    if len(p_filt) == 0:
        print(f"  [WARN] No samples in SNR {snr_range} for {title}, skipping.")
        return

    present_classes = sorted(np.unique(np.concatenate([l_filt, p_filt])))
    class_labels    = [CLASS_NAMES[c] for c in present_classes]

    # Remap to 0-based indices for sklearn
    remap  = {orig: new for new, orig in enumerate(present_classes)}
    p_remap = np.array([remap[v] for v in p_filt])
    l_remap = np.array([remap[v] for v in l_filt])

    cm = confusion_matrix(l_remap, p_remap, labels=list(range(len(present_classes))))
    # Normalise by row (true label) — gives per-class recall
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n = len(present_classes)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(f"{title}\n(SNR {lo} to {hi} dB, normalised by row)", fontsize=11)

    # Cell text for cells above 0.05 to keep the chart readable
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0.02:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5 if n > 16 else 7,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Section 2b — Per-Class F1 Bar Chart
# ============================================================

def plot_per_class_f1(
    preds:  np.ndarray,
    labels: np.ndarray,
    title:  str,
    path:   str,
) -> None:
    """Horizontal bar chart of per-class F1 scores, saved to path."""
    f1_per_class = f1_score(
        labels, preds,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    # Sort by F1 ascending so weakest classes appear at top (easiest to scan)
    order       = np.argsort(f1_per_class)
    sorted_f1   = f1_per_class[order]
    sorted_names = [CLASS_NAMES[i] for i in order]
    colors      = ["crimson" if f < 0.5 else "steelblue" for f in sorted_f1]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_names, sorted_f1, color=colors, edgecolor="white", linewidth=0.4)

    ax.axvline(x=0.5, color="grey", linestyle="--", linewidth=1.2, alpha=0.6,
               label="F1 = 0.50 threshold")

    for bar, val in zip(bars, sorted_f1):
        ax.text(
            min(val + 0.01, 1.00), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8,
            color="black",
        )

    overall_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title(
        f"{title} — Per-Class F1 Score\n"
        f"Weighted avg F1 = {overall_f1:.3f}  |  "
        f"{int((f1_per_class >= 0.5).sum())}/{NUM_CLASSES} classes ≥ 0.50",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Section 3 — Open-Set AUROC
# ============================================================

def evaluate_open_set(
    specter_model: SpecterCNN,
    detector:      EnergyOpenSetDetector,
    data_path:     str,
    batch_size:    int,
    save_dir:      str,
    device:        torch.device,
) -> float:
    """
    Collect energy scores for known (classes 0-19) and unknown (classes 20-23),
    compute AUROC, plot the ROC curve with the operating threshold marked.
    """
    print("\n  Building known-class loader (classes 0-19)...")
    known_ds = LazyHDF5Dataset(data_path, only_classes=KNOWN_IDS)
    known_loader = make_loader(known_ds, batch_size)

    print("  Building unknown-class loader (classes 20-23)...")
    unk_ds = LazyHDF5Dataset(data_path, only_classes=HOLDOUT_IDS)
    unk_loader = make_loader(unk_ds, batch_size)

    # Gather logits
    def _gather_logits(loader, desc):
        parts = []
        with torch.no_grad():
            for x, _, _ in tqdm(loader, desc=f"  {desc}", leave=False, ncols=100):
                parts.append(specter_model(x.to(device)).cpu())
        return torch.cat(parts, dim=0)

    known_logits = _gather_logits(known_loader,  "Known logits")
    unk_logits   = _gather_logits(unk_loader,    "Unknown logits")

    # Scores: higher = more likely known
    known_scores = (-energy_score(known_logits)).numpy()
    unk_scores   = (-energy_score(unk_logits)).numpy()

    all_scores = np.concatenate([known_scores, unk_scores])
    all_labels = np.concatenate([
        np.ones(len(known_scores),  dtype=int),
        np.zeros(len(unk_scores),   dtype=int),
    ])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    print(f"\n  Energy Open-Set AUROC : {auroc:.4f}")

    # Also compute softmax baseline AUROC for comparison
    softmax_auroc = eval_auroc(known_logits, unk_logits, detector="softmax")
    print(f"  Softmax MSP AUROC     : {softmax_auroc:.4f}")

    # Operating point: find (fpr, tpr) closest to the fitted threshold
    op_score = -detector.threshold   # threshold is in energy space; score = -energy
    op_idx   = np.argmin(np.abs(thresholds - op_score))
    op_fpr   = fpr[op_idx]
    op_tpr   = tpr[op_idx]

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="steelblue", linewidth=2.2,
            label=f"Energy Score  (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1, linestyle="--", label="Random")

    # Operating threshold point
    ax.scatter([op_fpr], [op_tpr], s=120, color="crimson", zorder=6,
               label=f"Operating point (95th pct)\nFPR={op_fpr:.3f}  TPR={op_tpr:.3f}")
    ax.annotate(
        f"  FPR={op_fpr:.3f}\n  TPR={op_tpr:.3f}",
        xy=(op_fpr, op_tpr), xytext=(op_fpr + 0.08, op_tpr - 0.12),
        fontsize=8, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Open-Set Detection ROC\n"
                 f"Unknown classes: {[CLASS_NAMES[i] for i in HOLDOUT_IDS]}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "open_set_auroc.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return auroc


# ============================================================
# Section 4 — Threat Scoring Calibration
# ============================================================

THREAT_SCENARIOS = [
    # (description, confidence, snr_db, is_unknown, burst_ms, expected_level)
    ("Known, high SNR +20dB, high conf",         0.95,  20.0, False, 100.0, "LOW"),
    ("Known, medium SNR 0dB, medium conf",        0.70,   0.0, False, 100.0, "LOW"),
    ("Known, low SNR -10dB, low conf",            0.40, -10.0, False, 100.0, "MEDIUM"),
    ("Known, very low SNR -18dB, very low conf",  0.20, -18.0, False, 100.0, "MEDIUM"),
    ("Unknown, high SNR +20dB",                   0.90,  20.0, True,  100.0, "MEDIUM"),
    ("Unknown, medium SNR 0dB",                   0.60,   0.0, True,  100.0, "CRITICAL"),
    ("Unknown, low SNR -10dB",                    0.35, -10.0, True,  100.0, "CRITICAL"),
    ("Unknown, very low SNR -18dB, short burst",  0.15, -18.0, True,   30.0, "CRITICAL"),
    ("Known, low SNR -14dB, short burst <50ms",   0.45, -14.0, False,  40.0, "MEDIUM"),
    ("Unknown, very low SNR -20dB, short burst",  0.10, -20.0, True,   20.0, "CRITICAL"),
]


def run_threat_calibration() -> int:
    """Run all 10 scenarios, print pass/fail table. Returns number of passes."""
    scorer = ThreatScorer()

    col_w = [44, 10, 10, 6]
    header = (
        f"  {'Scenario':<{col_w[0]}}  {'Expected':<{col_w[1]}}"
        f"  {'Actual':<{col_w[2]}}  {'Score':>6}  Result"
    )
    sep = "  " + "─" * (sum(col_w) + 20)

    print(f"\n  Threat Scoring Calibration  ({len(THREAT_SCENARIOS)} scenarios)")
    print(sep)
    print(header)
    print(sep)

    passes = 0
    for desc, conf, snr, unk, burst, expected in THREAT_SCENARIOS:
        result  = scorer.score(conf, snr, unk, burst)
        ok      = result.level == expected
        passes += int(ok)
        status  = "PASS ✓" if ok else f"FAIL ✗ (got {result.level})"
        print(
            f"  {desc:<{col_w[0]}}  {expected:<{col_w[1]}}"
            f"  {result.level:<{col_w[2]}}  {result.score:>5.1f}  {status}"
        )

    print(sep)
    print(f"  Result: {passes}/{len(THREAT_SCENARIOS)} passed\n")
    return passes


# ============================================================
# Section 5 — Inference Latency
# ============================================================

def measure_latency(
    model:   nn.Module,
    name:    str,
    input_len: int = INPUT_LEN,
    warmup:  int = 10,
) -> dict:
    """
    Measure per-sample latency on CPU at batch sizes 1 and 512.
    CPU is used deliberately — competition judges may not have a GPU.
    """
    cpu = torch.device("cpu")
    model_cpu = model.to(cpu)
    model_cpu.eval()

    results = {}
    for bs in (1, 512):
        dummy = torch.randn(bs, 2, input_len)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_cpu(dummy)

        # Timed runs
        n_runs = 50 if bs == 1 else 20
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model_cpu(dummy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ms_per_sample = elapsed_ms / (n_runs * bs)
        results[bs] = ms_per_sample

    print(f"  {name} latency:")
    print(f"    batch=1   → {results[1]:.3f} ms/sample")
    print(f"    batch=512 → {results[512]:.4f} ms/sample")

    return results


# ============================================================
# Section 6 — Final Summary Table
# ============================================================

def print_summary_table(
    b_overall:   float | None,
    b_low_snr:   float | None,
    b_precision: float | None,
    b_recall:    float | None,
    b_f1:        float | None,
    b_lat_1:     float | None,
    s_overall:   float | None,
    s_low_snr:   float | None,
    s_precision: float | None,
    s_recall:    float | None,
    s_f1:        float | None,
    s_auroc:     float | None,
    s_lat_1:     float | None,
) -> None:

    def _pct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def _ms(v):  return f"{v:.1f} ms"   if v is not None else "N/A"
    def _flt(v): return f"{v:.3f}"      if v is not None else "N/A"

    rows = [
        ("Overall Accuracy",  _pct(b_overall),   _pct(s_overall)),
        ("Low SNR Accuracy",  _pct(b_low_snr),   _pct(s_low_snr)),
        ("Precision",         _flt(b_precision),  _flt(s_precision)),
        ("Recall",            _flt(b_recall),     _flt(s_recall)),
        ("F1 Score",          _flt(b_f1),         _flt(s_f1)),
        ("Open-Set AUROC",    "N/A",              _flt(s_auroc)),
        ("Inference (ms)",    _ms(b_lat_1),       _ms(s_lat_1)),
    ]

    W = [20, 14, 15]
    inner = W[0] + W[1] + W[2] + 7

    print(f"\n  ┌{'─'*inner}┐")
    print(f"  │{'SPECTER BENCHMARK RESULTS':^{inner}}│")
    print(f"  ├{'─'*W[0]}┬{'─'*W[1]}┬{'─'*W[2]}┤")
    print(f"  │ {'Metric':<{W[0]-2}} │ {'Baseline':<{W[1]-2}} │ {'SPECTER':<{W[2]-2}} │")
    print(f"  ├{'─'*W[0]}┼{'─'*W[1]}┼{'─'*W[2]}┤")
    for metric, b_val, s_val in rows:
        print(f"  │ {metric:<{W[0]-2}} │ {b_val:<{W[1]-2}} │ {s_val:<{W[2]-2}} │")
    print(f"  └{'─'*W[0]}┴{'─'*W[1]}┴{'─'*W[2]}┘\n")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPECTER competition benchmark evaluation"
    )
    p.add_argument(
        "--data_path",
        default=os.path.join("data", "raw", "radio_ML",
                             "GOLD_XYZ_OSC.0001_1024.hdf5"),
    )
    p.add_argument("--baseline_path",  default=os.path.join("results", "baseline_best.pt"))
    p.add_argument("--specter_path",   default=os.path.join("results", "specter_best.pt"))
    p.add_argument("--threshold_path", default=os.path.join("results", "open_set_threshold.pkl"))
    p.add_argument("--save_dir",       default="results")
    p.add_argument("--batch_size",     type=int, default=512)
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    data_path      = os.path.abspath(args.data_path)
    baseline_path  = os.path.abspath(args.baseline_path)
    specter_path   = os.path.abspath(args.specter_path)
    threshold_path = os.path.abspath(args.threshold_path)
    save_dir       = os.path.abspath(args.save_dir)
    batch_size     = args.batch_size

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isfile(data_path):
        sys.exit(f"[ERROR] Dataset not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")
    print(f"  Data       : {data_path}")
    print(f"  Save dir   : {save_dir}\n")

    # ── load models (both optional — missing = skip gracefully) ──────────
    baseline = load_baseline(baseline_path, device)
    specter  = load_specter(specter_path,   device)
    detector = _try_load_threshold(threshold_path)

    if detector is None:
        print(f"  [SKIP] Threshold not found near: {threshold_path}")
        print("         Run  python train.py  first to fit the detector.")

    any_model = baseline is not None or specter is not None
    if not any_model:
        print("\n  No models loaded. Run train.py first.\n")
        return

    # ── build full-dataset test loader ───────────────────────────────────
    print("\n  Loading full dataset index (lazy)...")
    full_ds     = LazyHDF5Dataset(data_path)
    full_loader = make_loader(full_ds, batch_size)

    # ── per-model inference ───────────────────────────────────────────────
    b_preds = b_labels = b_snrs = b_logits = None
    s_preds = s_labels = s_snrs = s_logits = None

    if baseline is not None:
        print("\n  Running baseline inference over full dataset...")
        b_preds, b_labels, b_snrs, b_logits = collect_predictions(
            baseline, full_loader, device, is_baseline=True, desc="Baseline")

    if specter is not None:
        print("\n  Running SPECTER inference over full dataset...")
        s_preds, s_labels, s_snrs, s_logits = collect_predictions(
            specter, full_loader, device, is_baseline=False, desc="SPECTER")

    # ────────────────────────────────────────────────────────────────────
    # Section 1: Per-SNR Accuracy
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 1 — Per-SNR Accuracy")
    print(f"{'═'*58}")

    b_snr_accs = per_snr_accuracy(b_preds, b_labels, b_snrs) if baseline is not None else None
    s_snr_accs = per_snr_accuracy(s_preds, s_labels, s_snrs) if specter  is not None else None

    save_snr_csv(b_snr_accs, s_snr_accs, os.path.join(save_dir, "accuracy_vs_snr.csv"))
    plot_snr_accuracy(b_snr_accs, s_snr_accs, os.path.join(save_dir, "accuracy_vs_snr.png"))

    # Console print
    all_eval_snrs = sorted(set(
        list(b_snr_accs.keys() if b_snr_accs else []) +
        list(s_snr_accs.keys() if s_snr_accs else [])
    ))
    print(f"\n  {'SNR':>6}   {'Baseline':>10}   {'SPECTER':>10}   {'Delta':>8}")
    print("  " + "─" * 42)
    for snr in all_eval_snrs:
        b_v = b_snr_accs.get(snr) if b_snr_accs else None
        s_v = s_snr_accs.get(snr) if s_snr_accs else None
        b_s = f"{b_v:.4f}" if b_v is not None else "    N/A"
        s_s = f"{s_v:.4f}" if s_v is not None else "    N/A"
        d_s = f"{s_v-b_v:+.4f}" if (b_v is not None and s_v is not None) else "    N/A"
        print(f"  {snr:>+5d} dB   {b_s:>10}   {s_s:>10}   {d_s:>8}")

    # ────────────────────────────────────────────────────────────────────
    # Section 2: Confusion Matrices (low SNR only)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 2 — Confusion Matrices (Low SNR: -20 to -6 dB)")
    print(f"{'═'*58}")

    if baseline is not None:
        plot_confusion_matrix(
            b_preds, b_labels, b_snrs,
            title="Baseline VT-CNN2",
            path=os.path.join(save_dir, "confusion_baseline.png"),
        )
    if specter is not None:
        plot_confusion_matrix(
            s_preds, s_labels, s_snrs,
            title="SPECTER-CNN",
            path=os.path.join(save_dir, "confusion_specter.png"),
        )

    # ────────────────────────────────────────────────────────────────────
    # Section 3: Open-Set AUROC
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 3 — Open-Set AUROC")
    print(f"{'═'*58}")

    s_auroc = None
    if specter is not None and detector is not None:
        s_auroc = evaluate_open_set(
            specter, detector, data_path, batch_size, save_dir, device)
    elif specter is None:
        print("  [SKIP] SPECTER model not loaded.")
    elif detector is None:
        print("  [SKIP] Open-set threshold not loaded.")

    # ────────────────────────────────────────────────────────────────────
    # Section 4: Threat Scoring Calibration
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 4 — Threat Scoring Calibration")
    print(f"{'═'*58}")
    run_threat_calibration()

    # ────────────────────────────────────────────────────────────────────
    # Section 5: Inference Latency
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 5 — Inference Latency (CPU)")
    print(f"{'═'*58}")

    b_lat = measure_latency(baseline, "Baseline") if baseline is not None else None
    s_lat = measure_latency(specter,  "SPECTER")  if specter  is not None else None

    # ────────────────────────────────────────────────────────────────────
    # Compute summary metrics
    # ────────────────────────────────────────────────────────────────────
    def _overall_acc(preds, labels):
        if preds is None:
            return None
        return float((preds == labels).mean())

    def _low_snr_acc(preds, labels, snrs):
        if preds is None:
            return None
        lo, hi = LOW_SNR
        mask = (snrs >= lo) & (snrs <= hi)
        if mask.sum() == 0:
            return None
        return float((preds[mask] == labels[mask]).mean())

    def _weighted_metrics(preds, labels):
        """Return (f1, precision, recall) weighted, or (None, None, None)."""
        if preds is None:
            return None, None, None
        kw = dict(average="weighted", zero_division=0)
        return (
            float(f1_score(labels, preds, **kw)),
            float(precision_score(labels, preds, **kw)),
            float(recall_score(labels, preds, **kw)),
        )

    b_overall = _overall_acc(b_preds, b_labels)
    b_low_snr = _low_snr_acc(b_preds, b_labels, b_snrs)
    b_f1, b_prec, b_rec = _weighted_metrics(b_preds, b_labels)

    s_overall = _overall_acc(s_preds, s_labels)
    s_low_snr = _low_snr_acc(s_preds, s_labels, s_snrs)
    s_f1, s_prec, s_rec = _weighted_metrics(s_preds, s_labels)

    # ────────────────────────────────────────────────────────────────────
    # Section 2b: Per-Class F1 Charts
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 2b — Per-Class F1 Score Charts")
    print(f"{'═'*58}")

    if baseline is not None and b_preds is not None:
        plot_per_class_f1(
            b_preds, b_labels,
            title="Baseline VT-CNN2",
            path=os.path.join(save_dir, "per_class_f1_baseline.png"),
        )
    if specter is not None and s_preds is not None:
        plot_per_class_f1(
            s_preds, s_labels,
            title="SPECTER-CNN",
            path=os.path.join(save_dir, "per_class_f1.png"),
        )

    # ────────────────────────────────────────────────────────────────────
    # Section 6: Final Summary Table
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Section 6 — Final Competition Summary")
    print(f"{'═'*58}")
    print_summary_table(
        b_overall   = b_overall,
        b_low_snr   = b_low_snr,
        b_precision = b_prec,
        b_recall    = b_rec,
        b_f1        = b_f1,
        b_lat_1     = b_lat[1]  if b_lat else None,
        s_overall   = s_overall,
        s_low_snr   = s_low_snr,
        s_precision = s_prec,
        s_recall    = s_rec,
        s_f1        = s_f1,
        s_auroc     = s_auroc,
        s_lat_1     = s_lat[1]  if s_lat else None,
    )

    print("  Evaluation complete.\n")


if __name__ == "__main__":
    main()
