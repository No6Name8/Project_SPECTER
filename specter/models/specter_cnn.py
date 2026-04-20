"""
SPECTER-CNN  —  Deep residual classifier optimised for low-SNR AMR.

Architecture overview:
    Stem  : Conv1d(2→64, k=7)  + BN + ReLU
    Stage1: 2 × ResBlock(64→64,   stride=1)
    Stage2: 2 × ResBlock(64→128,  stride=2)
    Stage3: 2 × ResBlock(128→256, stride=2)
    Stage4: 2 × ResBlock(256→512, stride=2)
    Head  : GlobalAvgPool → Dropout → Linear(512→24)   [raw logits, no softmax]

Design choices vs VT-CNN2 baseline:
  - Residual skip connections keep gradients alive in the deeper stack.
  - Batch normalisation after every conv stabilises training at low SNR.
  - Global average pooling removes spatial bias; the model generalises to
    variable-length inputs without retraining the head.
  - Raw logits (no softmax) are required by the open-set rejection layer
    (models/open_set.py) which operates on the penultimate feature space.
  - SNR-aware loss weighting (3× for -20 dB to -6 dB) focuses capacity on
    the hardest regime that the baseline struggles with most.

Input  : (batch, 2, 128)  — channels-first IQ samples
Output : (batch, 24)      — raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Pre-activation residual block (He et al., 2016 "Identity Mappings").

    BN → ReLU → Conv → BN → ReLU → Dropout → Conv → (+skip)

    If in_channels != out_channels or stride != 1 a 1×1 conv projection
    aligns the skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = kernel_size // 2  # same-length output when stride=1

        self.bn1   = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )

        # Projection shortcut — only when shape changes
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.shortcut(x) if self.shortcut else x

        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.drop(F.relu(self.bn2(out))))
        return out + skip


def _make_stage(in_ch, out_ch, num_blocks, stride_first, dropout):
    """Stack num_blocks ResBlocks; first block may downsample via stride."""
    blocks = [ResBlock(in_ch, out_ch, stride=stride_first, dropout=dropout)]
    for _ in range(1, num_blocks):
        blocks.append(ResBlock(out_ch, out_ch, stride=1, dropout=dropout))
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------------------
# SPECTER-CNN
# ---------------------------------------------------------------------------

class SpecterCNN(nn.Module):
    """
    Deep residual IQ classifier.  Outputs raw logits suitable for
    cross-entropy training and open-set softmax thresholding.
    """

    def __init__(
        self,
        num_classes: int = 24,
        base_channels: int = 64,
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_classes      : number of modulation classes (24 for RadioML 2018)
            base_channels    : channels in the first residual stage (doubles each stage)
            blocks_per_stage : ResBlocks per stage (paper default = 2)
            dropout          : dropout probability inside each ResBlock
        """
        super().__init__()

        c = base_channels  # shorthand

        # Stem — expand IQ channels to base width
        self.stem = nn.Sequential(
            nn.Conv1d(2, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )

        # Four residual stages — each (except the first) halves temporal length
        self.stage1 = _make_stage(c,     c,     blocks_per_stage, stride_first=1, dropout=dropout)
        self.stage2 = _make_stage(c,     c * 2, blocks_per_stage, stride_first=2, dropout=dropout)
        self.stage3 = _make_stage(c * 2, c * 4, blocks_per_stage, stride_first=2, dropout=dropout)
        self.stage4 = _make_stage(c * 4, c * 8, blocks_per_stage, stride_first=2, dropout=dropout)

        # Head
        self.head_bn   = nn.BatchNorm1d(c * 8)
        self.head_drop = nn.Dropout(p=0.5)
        self.fc        = nn.Linear(c * 8, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 128)
        Returns:
            logits: (batch, num_classes)  — no softmax applied
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Pre-head activation then global average pool
        x = F.relu(self.head_bn(x))          # (batch, 512, T)
        x = x.mean(dim=-1)                   # (batch, 512)  — global avg pool

        x = self.head_drop(x)
        return self.fc(x)                    # (batch, 24)  raw logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate 512-d feature vector (used by open_set.py)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.head_bn(x))
        return x.mean(dim=-1)               # (batch, 512)


# ---------------------------------------------------------------------------
# SNR-aware loss
# ---------------------------------------------------------------------------

# Samples in this SNR window are the hardest — up-weight them during training
_LOW_SNR_MIN = -20
_LOW_SNR_MAX = -6
_LOW_SNR_WEIGHT = 3.0


def snr_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    snrs: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss with per-sample SNR weighting.

    Samples whose SNR falls in [_LOW_SNR_MIN, _LOW_SNR_MAX] dB receive
    _LOW_SNR_WEIGHT× the loss of a normal sample, focusing gradient signal
    on the regime where the baseline collapses.

    Args:
        logits : (batch, num_classes)  raw model output
        labels : (batch,)              integer class indices
        snrs   : (batch,)              SNR values in dB (float)

    Returns:
        Scalar weighted mean loss.
    """
    # Per-sample cross-entropy (no reduction)
    per_sample = F.cross_entropy(logits, labels, reduction="none")  # (batch,)

    # Build weight vector: 3× for low-SNR samples, 1× otherwise
    low_snr_mask = (snrs >= _LOW_SNR_MIN) & (snrs <= _LOW_SNR_MAX)
    weights = torch.where(
        low_snr_mask.to(logits.device),
        torch.full_like(per_sample, _LOW_SNR_WEIGHT),
        torch.ones_like(per_sample),
    )

    # Weighted mean (normalise by sum of weights so scale stays comparable)
    return (per_sample * weights).sum() / weights.sum()


# ---------------------------------------------------------------------------
# Training helpers  (same interface as baseline_cnn.py)
# ---------------------------------------------------------------------------

def train_epoch(
    model: SpecterCNN,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    One full training epoch with SNR-aware loss weighting.

    Args:
        model     : SpecterCNN instance
        loader    : DataLoader yielding (x, label, snr) batches
        optimizer : any torch optimizer (AdamW recommended)
        device    : cuda or cpu

    Returns:
        dict with "loss" (float, weighted mean) and "acc" (float, 0–1)
    """
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, labels, snrs in loader:
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        snrs   = snrs.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss   = snr_weighted_loss(logits, labels, snrs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += x.size(0)

    return {
        "loss": total_loss / total,
        "acc":  correct   / total,
    }


def eval_epoch(
    model: SpecterCNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate on one DataLoader split (no gradient, standard cross-entropy).

    Args:
        model  : SpecterCNN instance
        loader : DataLoader yielding (x, label, snr) batches
        device : cuda or cpu

    Returns:
        dict with "loss" (float) and "acc" (float, 0–1)
    """
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for x, labels, snrs in loader:
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            snrs   = snrs.to(device, non_blocking=True)

            logits = model(x)
            loss   = snr_weighted_loss(logits, labels, snrs)

            total_loss += loss.item() * x.size(0)
            preds       = logits.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += x.size(0)

    return {
        "loss": total_loss / total,
        "acc":  correct   / total,
    }


# ---------------------------------------------------------------------------
# Per-SNR accuracy  (identical interface to baseline_cnn.accuracy_by_snr)
# ---------------------------------------------------------------------------

def accuracy_by_snr(
    model: SpecterCNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Compute per-SNR-band classification accuracy.

    Args:
        model  : SpecterCNN instance
        loader : DataLoader yielding (x, label, snr) batches
        device : cuda or cpu

    Returns:
        dict mapping SNR value (int dB) → accuracy (float 0–1), sorted ascending.
        e.g. {-20: 0.09, -18: 0.14, ..., 30: 0.999}
    """
    model.eval()

    snr_correct = defaultdict(int)
    snr_total   = defaultdict(int)

    with torch.no_grad():
        for x, labels, snrs in loader:
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(x)
            preds  = logits.argmax(dim=1)
            hits   = preds.eq(labels)

            for hit, snr_val in zip(hits.cpu(), snrs):
                snr_key = int(round(snr_val.item()))
                snr_correct[snr_key] += int(hit.item())
                snr_total[snr_key]   += 1

    return {
        snr: snr_correct[snr] / snr_total[snr]
        for snr in sorted(snr_total.keys())
    }
