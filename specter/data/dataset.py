import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# RadioML 2018.01A contains 24 modulation classes at 26 SNR levels (-20 to +30 dB, step 2)
MODULATION_CLASSES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


def get_class_names():
    """Return the ordered list of 24 modulation class names for RadioML 2018.01A."""
    return MODULATION_CLASSES


def get_snr_values():
    """Return the 26 SNR levels present in RadioML 2018.01A (-20 to +30 dB, step 2)."""
    return list(range(-20, 32, 2))


class RadioMLDataset(Dataset):
    """
    PyTorch Dataset wrapping the RadioML 2018.01A HDF5 file.

    The HDF5 file layout:
        X  — shape (N, 1024, 2)  I/Q sample frames, float32
        Y  — shape (N, 24)       one-hot modulation labels
        Z  — shape (N, 1)        SNR value per sample in dB

    Each sample is returned as:
        x  — torch.Tensor of shape (2, 1024)  (channels-first: I then Q)
        y  — int class index  (0–23)
        snr — float SNR value in dB
    """

    def __init__(self, hdf5_path, snr_range=None, held_out_classes=None):
        """
        Args:
            hdf5_path      : path to GOLD_XYZ_OSC.0001_1024.hdf5
            snr_range      : optional (min_snr, max_snr) tuple, inclusive.
                             e.g. (-20, -6) keeps only low-SNR samples.
                             If None, all SNR levels are used.
            held_out_classes: optional list of class name strings or int indices
                             to EXCLUDE from this dataset (used for open-set splits).
                             e.g. ["FM", "GMSK"] or [21, 22]
        """
        self.hdf5_path = hdf5_path
        self.snr_range = snr_range
        self.held_out_classes = held_out_classes

        # Resolve held-out class indices
        held_out_indices = set()
        if held_out_classes:
            for c in held_out_classes:
                if isinstance(c, str):
                    held_out_indices.add(MODULATION_CLASSES.index(c))
                else:
                    held_out_indices.add(int(c))

        # Load everything into memory for fast __getitem__ access.
        # RadioML 2018.01A is ~25 GB on disk but the filtered subset is manageable.
        with h5py.File(hdf5_path, "r") as f:
            X = f["X"][:]          # (N, 1024, 2) float32
            Y = f["Y"][:]          # (N, 24)      float32 one-hot
            Z = f["Z"][:].squeeze()  # (N,)        float32 SNR

        # Convert one-hot labels to integer class indices
        labels = np.argmax(Y, axis=1)  # (N,)

        # Build boolean mask for valid samples
        mask = np.ones(len(labels), dtype=bool)

        # Filter by SNR range if requested
        if snr_range is not None:
            min_snr, max_snr = snr_range
            mask &= (Z >= min_snr) & (Z <= max_snr)

        # Remove held-out classes if requested
        if held_out_indices:
            for idx in held_out_indices:
                mask &= labels != idx

        # Apply mask — store only the retained samples
        self.X = torch.from_numpy(X[mask])          # (M, 1024, 2)
        self.labels = torch.from_numpy(labels[mask].astype(np.int64))  # (M,)
        self.snrs = torch.from_numpy(Z[mask])        # (M,)

        # Transpose X to channels-first: (M, 2, 1024) — expected by Conv1d models
        self.X = self.X.permute(0, 2, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Return (x, label, snr) for sample at index idx."""
        return self.X[idx], self.labels[idx], self.snrs[idx]


def get_dataloaders(
    hdf5_path,
    batch_size=256,
    train_ratio=0.8,
    val_ratio=0.1,
    snr_range=None,
    held_out_classes=None,
    num_workers=4,
    seed=42,
):
    """
    Build train / val / test DataLoaders from a RadioML 2018.01A HDF5 file.

    Args:
        hdf5_path        : path to HDF5 dataset file
        batch_size       : samples per batch
        train_ratio      : fraction of data for training (default 0.8)
        val_ratio        : fraction of data for validation (default 0.1)
                           test gets the remaining 1 - train_ratio - val_ratio
        snr_range        : (min_snr, max_snr) inclusive filter, or None for all
        held_out_classes : class names/indices to exclude entirely, or None
        num_workers      : DataLoader worker processes
        seed             : random seed for reproducible splits

    Returns:
        dict with keys "train", "val", "test" each holding a DataLoader,
        plus "n_classes" (int) and "class_names" (list of str).
    """
    dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        snr_range=snr_range,
        held_out_classes=held_out_classes,
    )

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Reproducible random split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # Determine active class count (may be <24 when held_out_classes is set)
    held_out_indices = set()
    if held_out_classes:
        for c in held_out_classes:
            if isinstance(c, str):
                held_out_indices.add(MODULATION_CLASSES.index(c))
            else:
                held_out_indices.add(int(c))
    active_classes = [
        name for i, name in enumerate(MODULATION_CLASSES) if i not in held_out_indices
    ]

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
        "val":   DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_ds,  shuffle=False, **loader_kwargs),
        "n_classes":   len(active_classes),
        "class_names": active_classes,
    }


def get_open_set_loaders(
    hdf5_path,
    known_classes,
    batch_size=256,
    snr_range=None,
    num_workers=4,
    seed=42,
):
    """
    Convenience wrapper for open-set evaluation splits.

    Builds:
        closed_set  — DataLoader containing ONLY known_classes samples (train/val/test)
        open_set    — DataLoader containing ONLY the unknown/held-out class samples

    Args:
        hdf5_path     : path to HDF5 dataset file
        known_classes : list of class names or indices that are "known"
                        everything else becomes the open-set unknown split
        batch_size    : samples per batch
        snr_range     : optional SNR filter tuple
        num_workers   : DataLoader workers
        seed          : random seed

    Returns:
        dict with "closed" and "open" DataLoaders
    """
    # Resolve known indices
    known_indices = set()
    for c in known_classes:
        if isinstance(c, str):
            known_indices.add(MODULATION_CLASSES.index(c))
        else:
            known_indices.add(int(c))

    unknown_classes = [
        i for i in range(len(MODULATION_CLASSES)) if i not in known_indices
    ]

    # Closed-set dataset: exclude unknowns
    closed_ds = RadioMLDataset(
        hdf5_path=hdf5_path,
        snr_range=snr_range,
        held_out_classes=unknown_classes,
    )

    # Open-set dataset: only the unknowns
    open_ds = RadioMLDataset(
        hdf5_path=hdf5_path,
        snr_range=snr_range,
        held_out_classes=[
            i for i in range(len(MODULATION_CLASSES)) if i in known_indices
        ],
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )

    return {
        "closed": DataLoader(closed_ds, **loader_kwargs),
        "open":   DataLoader(open_ds,   **loader_kwargs),
    }
