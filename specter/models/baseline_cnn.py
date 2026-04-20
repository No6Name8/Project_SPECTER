"""
VT-CNN2  —  Baseline modulation classifier.

Reproduces the architecture from:
  O'Shea, T., Corgan, J., Clancy, T.C. (2016).
  "Convolutional Radio Modulation Recognition Networks."
  arXiv:1602.04105

Original used Keras Conv2D on (2, 128) IQ frames.  This implementation
maps those to Conv1d: the I and Q rows become the 2-channel 1-D input of
length 128, preserving every filter count, FC width, dropout rate, and
pooling stride from the paper.

Input tensor shape : (batch, 2, 128)   — channels-first IQ
Output tensor shape: (batch, 24)       — log-softmax scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VT_CNN2(nn.Module):
    """
    VT-CNN2 baseline from O'Shea et al. 2016, adapted for 24-class RadioML 2018.

    Architecture (paper §III-B):
        Conv1(256, k=1) → ReLU → MaxPool(2)
        Dropout(0.5)
        Conv2(80,  k=3) → ReLU → MaxPool(2)
        Dropout(0.5)
        Flatten
        FC(256) → ReLU → Dropout(0.5)
        FC(num_classes) → log-softmax
    """

    def __init__(self, num_classes: int = 24, input_len: int = 128):
        super().__init__()

        # --- convolutional stack -------------------------------------------
        # Conv1: 256 filters, kernel 1  (matches paper's 1×1 spatial kernel)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.5)

        # Conv2: 80 filters, kernel 3  (matches paper's 1×3 temporal kernel)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=80, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.5)

        # --- compute flattened size dynamically so input_len can be changed --
        flat_len = self._get_flat_len(input_len)

        # --- fully connected stack ------------------------------------------
        self.fc1  = nn.Linear(flat_len, 256)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2  = nn.Linear(256, num_classes)

    # ------------------------------------------------------------------
    def _get_flat_len(self, input_len: int) -> int:
        """Dry-run a dummy tensor to measure the flattened conv output size."""
        with torch.no_grad():
            x = torch.zeros(1, 2, input_len)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.numel()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 128)  — IQ samples, channels-first
        Returns:
            (batch, num_classes)  — log-softmax scores
        """
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)                  # flatten
        x = self.drop3(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: VT_CNN2,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Run one full training epoch.

    Args:
        model     : VT_CNN2 instance
        loader    : DataLoader yielding (x, label, snr) batches
        optimizer : any torch optimizer
        device    : cuda or cpu

    Returns:
        dict with keys "loss" (float, mean NLL loss) and "acc" (float, 0–1)
    """
    model.train()
    criterion = nn.NLLLoss()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, labels, _ in loader:
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        log_probs = model(x)
        loss      = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds       = log_probs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += x.size(0)

    return {
        "loss": total_loss / total,
        "acc":  correct   / total,
    }


def eval_epoch(
    model: VT_CNN2,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on one DataLoader split (no gradient computation).

    Args:
        model  : VT_CNN2 instance
        loader : DataLoader yielding (x, label, snr) batches
        device : cuda or cpu

    Returns:
        dict with keys "loss" (float) and "acc" (float, 0–1)
    """
    model.eval()
    criterion = nn.NLLLoss()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for x, labels, _ in loader:
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            log_probs  = model(x)
            loss       = criterion(log_probs, labels)

            total_loss += loss.item() * x.size(0)
            preds       = log_probs.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += x.size(0)

    return {
        "loss": total_loss / total,
        "acc":  correct   / total,
    }


# ---------------------------------------------------------------------------
# SNR-band accuracy breakdown
# ---------------------------------------------------------------------------

def accuracy_by_snr(
    model: VT_CNN2,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Compute per-SNR-band classification accuracy, matching the evaluation
    convention from the DeepSig benchmarks.

    Args:
        model  : VT_CNN2 instance
        loader : DataLoader yielding (x, label, snr) batches — SNR must be the
                 raw dB integer value (as stored in RadioML 2018 Z array)
        device : cuda or cpu

    Returns:
        dict mapping SNR value (int, dB) → accuracy (float, 0–1)
        e.g. {-20: 0.083, -18: 0.091, ..., 30: 0.997}
    """
    model.eval()

    # Accumulators: snr → [correct_count, total_count]
    snr_correct = defaultdict(int)
    snr_total   = defaultdict(int)

    with torch.no_grad():
        for x, labels, snrs in loader:
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            log_probs = model(x)
            preds     = log_probs.argmax(dim=1)
            hits      = preds.eq(labels)            # bool tensor (batch,)

            # Accumulate per SNR bin (snrs are float dB values — round to int)
            for hit, snr_val in zip(hits.cpu(), snrs):
                snr_key = int(round(snr_val.item()))
                snr_correct[snr_key] += int(hit.item())
                snr_total[snr_key]   += 1

    return {
        snr: snr_correct[snr] / snr_total[snr]
        for snr in sorted(snr_total.keys())
    }
