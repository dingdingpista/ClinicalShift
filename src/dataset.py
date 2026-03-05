
# =============================================================
# src/dataset.py
# ClinicalShift — ECG Preprocessing Pipeline
# =============================================================
#
# MATHEMATICAL OPERATIONS:
#
# 1. Min-Max Normalization:
#    x_norm = (x - x_min) / (x_max - x_min + epsilon)
#    Maps all signal values to range [0, 1]
#    epsilon = 1e-8 prevents division by zero
#
# 2. Sliding Window Segmentation:
#    n_windows = floor((L - W) / S) + 1
#    Where:
#      L = signal length (1000 samples)
#      W = window size   (250 samples = 2.5 seconds)
#      S = step size     (125 samples = 50% overlap)
# =============================================================

import numpy as np
import wfdb
import os
import torch
from torch.utils.data import Dataset

# ---- Constants ----
WINDOW_SIZE  = 250    # 2.5 seconds at 100 Hz
STEP_SIZE    = 125    # 50% overlap between windows
LEAD_INDEX   = 1      # Lead II — most standard clinical lead
EPSILON      = 1e-8   # prevents division by zero in normalization


def normalize_signal(signal):
    """
    Min-Max Normalization.

    Maps signal values from raw mV range to [0, 1].

    Formula:
        x_norm = (x - x_min) / (x_max - x_min + epsilon)

    Args:
        signal : numpy array of shape (1000,) — single lead

    Returns:
        normalized numpy array of shape (1000,)
    """
    x_min = signal.min()
    x_max = signal.max()
    return (signal - x_min) / (x_max - x_min + EPSILON)


def sliding_window(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Sliding Window Segmentation.

    Cuts a long signal into overlapping windows.

    Formula:
        n_windows = floor((L - W) / S) + 1

    Args:
        signal      : numpy array of shape (L,)
        window_size : W — length of each window (250)
        step_size   : S — step between windows  (125)

    Returns:
        numpy array of shape (n_windows, window_size)
    """
    L = len(signal)
    n_windows = (L - window_size) // step_size + 1
    windows = []
    for i in range(n_windows):
        start = i * step_size
        end   = start + window_size
        windows.append(signal[start:end])
    return np.array(windows)   # shape: (n_windows, 250)


def load_and_preprocess(row, data_path):
    """
    Full preprocessing pipeline for one ECG recording.

    Steps:
        1. Load raw .dat file using wfdb
        2. Extract Lead II
        3. Normalize to [0, 1]
        4. Apply sliding window segmentation

    Args:
        row       : one row from ptbxl_database.csv
        data_path : root folder of dataset

    Returns:
        numpy array of shape (n_windows, window_size)
        typically (7, 250) per recording
    """
    # Step 1: Load raw signal — shape (1000, 12)
    full_path = os.path.join(data_path, row["filename_lr"])
    signal, _ = wfdb.rdsamp(full_path)

    # Step 2: Extract Lead II — shape (1000,)
    lead = signal[:, LEAD_INDEX]

    # Step 3: Normalize to [0, 1]
    lead_normalized = normalize_signal(lead)

    # Step 4: Sliding window — shape (n_windows, 250)
    windows = sliding_window(lead_normalized)

    return windows


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG windows.

    Loads all recordings from a dataframe,
    preprocesses them, and stores all windows.

    Args:
        dataframe : pandas DataFrame (young or old split)
        data_path : root folder of dataset
        label     : 0 = in-distribution, 1 = out-of-distribution
    """
    def __init__(self, dataframe, data_path, label=0):
        self.windows = []
        self.labels  = []

        print(f"  Loading {len(dataframe)} recordings...")

        for idx in range(len(dataframe)):
            row = dataframe.iloc[idx]
            try:
                windows = load_and_preprocess(row, data_path)
                for w in windows:
                    self.windows.append(w)
                    self.labels.append(label)
            except Exception:
                pass   # skip corrupted files silently

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels  = np.array(self.labels,  dtype=np.long)

        print(f"  ✅ Total windows: {len(self.windows):,}")
        print(f"  ✅ Window shape : {self.windows[0].shape}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Add channel dimension: (250,) → (1, 250)
        # Neural networks expect (batch, channels, length)
        x = torch.tensor(self.windows[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx])
        return x, y
