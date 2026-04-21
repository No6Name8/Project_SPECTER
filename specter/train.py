"""
specter/train.py — Train VT-CNN2 baseline and/or SPECTER-CNN on RadioML 2018.01A.

Usage examples:
    python train.py
    python train.py --model specter --epochs 30 --batch_size 256
    python train.py --model both --low_snr_only
    python train.py --model baseline --holdout_classes 20,21,22,23

The RadioML 2018.01A HDF5 file (21 GB) is accessed with lazy per-sample reads —
only X[i] (one IQ frame) is fetched from disk per __getitem__; the label and SNR
arrays are small enough to load into memory in full at startup.
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.dataset       import get_class_names, MODULATION_CLASSES
from models.baseline_cnn import (
    VT_CNN2,
    accuracy_by_snr as baseline_accuracy_by_snr,
)
from models.specter_cnn import (
    SpecterCNN,
    snr_weighted_loss,
    accuracy_by_snr as specter_accuracy_by_snr,
)
from models.open_set import EnergyOpenSetDetector

# RadioML 2018.01A frame length
INPUT_LEN   = 1024
NUM_CLASSES = 24


# ---------------------------------------------------------------------------
# Lazy HDF5 Dataset
# ---------------------------------------------------------------------------

class LazyHDF5Dataset(Dataset):
    """
    Memory-efficient RadioML 2018.01A dataset.

    Y (labels) and Z (SNR values) are loaded fully into RAM — they are
    ~375 MB and ~16 MB respectively, well within budget.

    X (IQ samples, ~32 GB uncompressed) is accessed lazily: each worker
    opens its own HDF5 file handle on first __getitem__ call, reads a single
    row, and keeps the handle open for the lifetime of the worker process.
    This avoids both the memory cost of loading all data and the overhead of
    opening/closing the file on every access.
    """

    def __init__(
        self,
        hdf5_path:        str,
        snr_range:        tuple | None = None,
        held_out_classes: list  | None = None,
    ):
        self._path = hdf5_path
        self._file = None   # opened lazily per-process in __getitem__

        # Load only Y and Z into memory — X stays on disk
        with h5py.File(hdf5_path, "r") as f:
            print("  Loading label and SNR arrays into memory...", flush=True)
            Y = f["Y"][:]           # (N, 24)  float32 one-hot
            Z = f["Z"][:].squeeze() # (N,)     float32 SNR dB

        labels = np.argmax(Y, axis=1).astype(np.int64)  # (N,)
        del Y   # free ~375 MB

        # Build boolean sample mask
        mask = np.ones(len(labels), dtype=bool)

        if snr_range is not None:
            lo, hi = snr_range
            mask &= (Z >= lo) & (Z <= hi)

        if held_out_classes:
            for c in held_out_classes:
                mask &= labels != int(c)

        # Store only the selected indices, labels, and SNR values
        self._indices = np.where(mask)[0].astype(np.int64)  # disk row indices
        self._labels  = torch.from_numpy(labels[mask])
        self._snrs    = torch.from_numpy(Z[mask].astype(np.float32))

        n_total   = len(labels)
        n_kept    = int(mask.sum())
        print(f"  Dataset: {n_kept:,} / {n_total:,} samples selected", flush=True)

    # ------------------------------------------------------------------
    def _open_file(self):
        """Open the HDF5 file once per worker process."""
        if self._file is None:
            self._file = h5py.File(self._path, "r")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        self._open_file()
        disk_idx = int(self._indices[idx])
        # Read one IQ frame: (1024, 2) → transpose to (2, 1024) channels-first
        x = self._file["X"][disk_idx]                           # (1024, 2)
        x = torch.from_numpy(x.T.copy().astype(np.float32))    # (2, 1024)
        return x, self._labels[idx], self._snrs[idx]

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def make_loaders(
    hdf5_path:        str,
    batch_size:       int,
    snr_range:        tuple | None,
    held_out_classes: list  | None,
    num_workers:      int = 0,
    seed:             int = 42,
) -> dict:
    """Split dataset 80/10/10 and return DataLoaders."""
    ds = LazyHDF5Dataset(hdf5_path, snr_range=snr_range, held_out_classes=held_out_classes)

    n       = len(ds)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=generator)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return {
        "train": DataLoader(train_ds, shuffle=True,  **common),
        "val":   DataLoader(val_ds,   shuffle=False, **common),
        "test":  DataLoader(test_ds,  shuffle=False, **common),
    }


# ---------------------------------------------------------------------------
# Training loops with tqdm + time tracking
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s" if m else f"{s}s"


def train_one_epoch_baseline(
    model:     VT_CNN2,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
) -> dict:
    model.train()
    criterion  = nn.NLLLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0
    t_start    = time.perf_counter()

    bar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}", leave=False,
               unit="batch", ncols=100)

    for x, labels, _ in bar:
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        log_probs = model(x)
        loss      = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        bs          = x.size(0)
        total_loss += loss.item() * bs
        preds       = log_probs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += bs

        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    elapsed = time.perf_counter() - t_start
    return {"loss": total_loss / total, "acc": correct / total, "elapsed_s": elapsed}


def train_one_epoch_specter(
    model:     SpecterCNN,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
) -> dict:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    t_start    = time.perf_counter()

    bar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}", leave=False,
               unit="batch", ncols=100)

    for x, labels, snrs in bar:
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        snrs   = snrs.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss   = snr_weighted_loss(logits, labels, snrs)
        loss.backward()
        optimizer.step()

        bs          = x.size(0)
        total_loss += loss.item() * bs
        preds       = logits.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += bs

        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    elapsed = time.perf_counter() - t_start
    return {"loss": total_loss / total, "acc": correct / total, "elapsed_s": elapsed}


def eval_one_epoch(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_baseline: bool,
    label:  str = "Val",
) -> dict:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    bar = tqdm(loader, desc=f"  {label}", leave=False, unit="batch", ncols=100)

    with torch.no_grad():
        for x, labels, snrs in bar:
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            snrs   = snrs.to(device, non_blocking=True)

            if is_baseline:
                log_probs  = model(x)
                loss       = F.nll_loss(log_probs, labels)
                preds      = log_probs.argmax(dim=1)
            else:
                logits = model(x)
                loss   = snr_weighted_loss(logits, labels, snrs)
                preds  = logits.argmax(dim=1)

            bs          = x.size(0)
            total_loss += loss.item() * bs
            correct    += preds.eq(labels).sum().item()
            total      += bs

            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return {"loss": total_loss / total, "acc": correct / total}


def snr_breakdown(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    is_baseline: bool,
) -> dict:
    """Per-SNR accuracy — inline so we avoid double-importing."""
    model.eval()
    snr_correct  = defaultdict(int)
    snr_total    = defaultdict(int)

    with torch.no_grad():
        for x, labels, snrs in tqdm(loader, desc="  SNR eval", leave=False, ncols=100):
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds  = (
                model(x).argmax(dim=1) if is_baseline
                else model(x).argmax(dim=1)
            )
            hits = preds.eq(labels)
            for hit, snr_val in zip(hits.cpu(), snrs):
                key = int(round(snr_val.item()))
                snr_correct[key] += int(hit.item())
                snr_total[key]   += 1

    return {snr: snr_correct[snr] / snr_total[snr] for snr in sorted(snr_total)}


def print_snr_table(snr_accs: dict, model_name: str) -> None:
    print(f"\n  {'SNR (dB)':<10}  {model_name} Acc")
    print("  " + "-" * 30)
    for snr, acc in snr_accs.items():
        bar = "█" * int(acc * 20)
        print(f"  {snr:>+5d} dB   {acc:.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Model training orchestrator
# ---------------------------------------------------------------------------

def train_model(
    model_tag:        str,       # "baseline" or "specter"
    hdf5_path:        str,
    epochs:           int,
    batch_size:       int,
    save_dir:         str,
    snr_range:        tuple | None,
    held_out_classes: list,
    device:           torch.device,
) -> tuple[nn.Module, dict]:
    """
    Train one model end-to-end.

    Returns:
        (trained_model, test_snr_accuracy_dict)
    """
    is_baseline = model_tag == "baseline"
    label       = "Baseline VT-CNN2" if is_baseline else "SPECTER-CNN"
    ckpt_path   = os.path.join(save_dir, f"{model_tag}_best.pt")

    print(f"\n{'═'*58}")
    print(f"  Training {label}")
    print(f"{'═'*58}")

    # --- loaders -------------------------------------------------------------
    loaders = make_loaders(
        hdf5_path        = hdf5_path,
        batch_size       = batch_size,
        snr_range        = snr_range,
        held_out_classes = held_out_classes,
        num_workers      = 0,   # 0 = safe for HDF5 on Windows
    )

    # --- model ---------------------------------------------------------------
    if is_baseline:
        model = VT_CNN2(num_classes=NUM_CLASSES, input_len=INPUT_LEN).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        model = SpecterCNN(num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {param_count:,}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {ckpt_path}\n")

    best_val_acc = 0.0
    epoch_times  = []

    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()

        # Train
        if is_baseline:
            train_m = train_one_epoch_baseline(
                model, loaders["train"], optimizer, device, epoch, epochs)
        else:
            train_m = train_one_epoch_specter(
                model, loaders["train"], optimizer, device, epoch, epochs)

        # Validate
        val_m = eval_one_epoch(model, loaders["val"], device, is_baseline)

        scheduler.step()

        epoch_elapsed = time.perf_counter() - t_epoch
        epoch_times.append(epoch_elapsed)
        avg_epoch_s   = sum(epoch_times) / len(epoch_times)
        eta_s         = avg_epoch_s * (epochs - epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:>3}/{epochs}"
            f"  train_loss={train_m['loss']:.4f}  train_acc={train_m['acc']:.4f}"
            f"  val_loss={val_m['loss']:.4f}  val_acc={val_m['acc']:.4f}"
            f"  lr={lr_now:.2e}"
            f"  epoch={_fmt_time(epoch_elapsed)}"
            f"  ETA={_fmt_time(eta_s)}"
        )

        # Checkpoint
        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✓ Best checkpoint saved  (val_acc={best_val_acc:.4f})")

        # Per-SNR breakdown every 10 epochs and at the final epoch
        if epoch % 10 == 0 or epoch == epochs:
            accs = snr_breakdown(model, loaders["val"], device, is_baseline)
            print_snr_table(accs, label)

    # Reload best weights for test eval
    print(f"\n  Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m    = eval_one_epoch(model, loaders["test"], device, is_baseline, label="Test")
    test_snrs = snr_breakdown(model, loaders["test"], device, is_baseline)

    print(f"\n  ── {label} Final Results ──")
    print(f"  Best val acc : {best_val_acc:.4f}")
    print(f"  Test acc     : {test_m['acc']:.4f}")
    print(f"  Test loss    : {test_m['loss']:.4f}")

    return model, test_snrs


# ---------------------------------------------------------------------------
# Open-set detector fitting
# ---------------------------------------------------------------------------

def fit_open_set_detector(
    specter_model: SpecterCNN,
    hdf5_path:     str,
    batch_size:    int,
    save_dir:      str,
    device:        torch.device,
) -> None:
    """Fit EnergyOpenSetDetector on the known-class validation split and save."""
    print(f"\n{'─'*58}")
    print("  Fitting Energy Open-Set Detector...")
    print(f"{'─'*58}")

    # Use a clean validation loader — no holdouts — so the detector sees all
    # 24 classes as "known" for calibration
    loaders = make_loaders(
        hdf5_path        = hdf5_path,
        batch_size       = batch_size,
        snr_range        = None,
        held_out_classes = None,
        num_workers      = 0,
    )

    detector = EnergyOpenSetDetector(percentile=95.0)
    threshold = detector.fit(loaders["val"], specter_model, device)
    print(f"  Energy threshold (95th pct): {threshold:.6f}")

    threshold_path = os.path.join(save_dir, "open_set_threshold.json")
    detector.save_threshold(threshold_path)
    print(f"  Threshold saved → {threshold_path}")


# ---------------------------------------------------------------------------
# Benchmark comparison table
# ---------------------------------------------------------------------------

def print_benchmark_table(
    baseline_snrs: dict | None,
    specter_snrs:  dict | None,
) -> None:
    all_snrs = sorted(set(
        list(baseline_snrs.keys() if baseline_snrs else []) +
        list(specter_snrs.keys()  if specter_snrs  else [])
    ))

    print(f"\n{'═'*58}")
    print("  SPECTER vs Baseline — Per-SNR Test Accuracy")
    print(f"{'═'*58}")
    print(f"  {'SNR (dB)':<10}  {'Baseline':<12}  {'SPECTER':<12}  {'Delta':>8}")
    print("  " + "-" * 50)

    for snr in all_snrs:
        b_acc = baseline_snrs.get(snr) if baseline_snrs else None
        s_acc = specter_snrs.get(snr)  if specter_snrs  else None

        b_str = f"{b_acc:.4f}" if b_acc is not None else "  N/A  "
        s_str = f"{s_acc:.4f}" if s_acc is not None else "  N/A  "

        if b_acc is not None and s_acc is not None:
            delta    = s_acc - b_acc
            sign     = "+" if delta >= 0 else ""
            d_str    = f"{sign}{delta:+.4f}"
            marker   = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "  ")
        else:
            d_str  = "  N/A"
            marker = ""

        print(f"  {snr:>+5d} dB   {b_str:<12}  {s_str:<12}  {d_str}{marker}")

    print()

    # Summary averages per region
    for region, lo, hi in [("Low SNR [-20, -6]",  -20, -6),
                            ("Mid SNR [ -4, +8]",   -4,  8),
                            ("High SNR[+10,+30]",   10, 30)]:
        keys = [s for s in all_snrs if lo <= s <= hi]
        if not keys:
            continue
        b_avg = (sum(baseline_snrs[k] for k in keys if k in (baseline_snrs or {})) / len(keys)
                 if baseline_snrs else None)
        s_avg = (sum(specter_snrs[k]  for k in keys if k in (specter_snrs  or {})) / len(keys)
                 if specter_snrs  else None)
        b_s = f"{b_avg:.4f}" if b_avg is not None else "  N/A"
        s_s = f"{s_avg:.4f}" if s_avg is not None else "  N/A"
        print(f"  {region:<22}  baseline={b_s}  specter={s_s}")

    print(f"{'═'*58}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SPECTER signal classification models on RadioML 2018.01A"
    )
    p.add_argument(
        "--data_path",
        default=os.path.join("specter", "data", "raw", "radio_ML",
                             "GOLD_XYZ_OSC.0001_1024.hdf5"),
        help="Path to RadioML 2018.01A HDF5 file",
    )
    p.add_argument(
        "--model",
        choices=["baseline", "specter", "both"],
        default="both",
        help="Which model(s) to train",
    )
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument(
        "--low_snr_only",
        action="store_true",
        help="Train only on -20 dB to -6 dB SNR range",
    )
    p.add_argument(
        "--holdout_classes",
        default="20,21,22,23",
        help="Comma-separated class indices to exclude (open-set validation split). "
             "Default: 20,21,22,23 (AM-DSB-SC, FM, GMSK, OQPSK)",
    )
    p.add_argument(
        "--save_dir",
        default=os.path.join("specter", "results"),
        help="Directory for checkpoints and threshold files",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- validate paths ------------------------------------------------------
    data_path = os.path.abspath(args.data_path)
    save_dir  = os.path.abspath(args.save_dir)

    if not os.path.isfile(data_path):
        sys.exit(f"[ERROR] Dataset not found: {data_path}")

    os.makedirs(save_dir, exist_ok=True)

    # --- device --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"\n  GPU: {gpu_name}  ({gpu_mem} MB VRAM)")
    else:
        print("\n  No CUDA GPU detected — training on CPU (will be slow)")

    # --- config summary ------------------------------------------------------
    snr_range = (-20, -6) if args.low_snr_only else None
    held_out  = [int(c.strip()) for c in args.holdout_classes.split(",") if c.strip()]
    class_names = get_class_names()
    held_names  = [class_names[i] for i in held_out if i < len(class_names)]

    print(f"\n  Config")
    print(f"  ├─ data_path      : {data_path}")
    print(f"  ├─ model          : {args.model}")
    print(f"  ├─ epochs         : {args.epochs}")
    print(f"  ├─ batch_size     : {args.batch_size}")
    print(f"  ├─ snr_range      : {snr_range if snr_range else 'all'}")
    print(f"  ├─ holdout        : {held_out} → {held_names}")
    print(f"  └─ save_dir       : {save_dir}")

    # --- train ---------------------------------------------------------------
    baseline_snrs = None
    specter_model = None
    specter_snrs  = None

    if args.model in ("baseline", "both"):
        _, baseline_snrs = train_model(
            model_tag        = "baseline",
            hdf5_path        = data_path,
            epochs           = args.epochs,
            batch_size       = args.batch_size,
            save_dir         = save_dir,
            snr_range        = snr_range,
            held_out_classes = held_out,
            device           = device,
        )

    if args.model in ("specter", "both"):
        specter_model, specter_snrs = train_model(
            model_tag        = "specter",
            hdf5_path        = data_path,
            epochs           = args.epochs,
            batch_size       = args.batch_size,
            save_dir         = save_dir,
            snr_range        = snr_range,
            held_out_classes = held_out,
            device           = device,
        )

    # --- fit open-set detector (specter only) --------------------------------
    if specter_model is not None:
        fit_open_set_detector(
            specter_model = specter_model,
            hdf5_path     = data_path,
            batch_size    = args.batch_size,
            save_dir      = save_dir,
            device        = device,
        )

    # --- benchmark table -----------------------------------------------------
    if baseline_snrs is not None or specter_snrs is not None:
        print_benchmark_table(baseline_snrs, specter_snrs)

    print("  Training complete.\n")


if __name__ == "__main__":
    main()
