
# =============================================================
# src/model.py
# ClinicalShift — Contrastive Encoder Architecture
# =============================================================
#
# ARCHITECTURE:
#
#   Input: (batch, 1, 250)
#       down
#   Conv Block 1: Conv1D(1->64, k=7) + BatchNorm + ReLU + MaxPool
#       down  shape: (batch, 64, 125)
#   Conv Block 2: Conv1D(64->128, k=5) + BatchNorm + ReLU + MaxPool
#       down  shape: (batch, 128, 62)
#   Conv Block 3: Conv1D(128->256, k=3) + BatchNorm + ReLU + MaxPool
#       down  shape: (batch, 256, 31)
#   Global Average Pooling
#       down  shape: (batch, 256)
#   Embedding Layer: Linear(256->128) + ReLU + Dropout(0.3)
#       down  shape: (batch, 128)
#   Projection Head: Linear(128->64) + L2 Normalize
#       down  shape: (batch, 64)
#
# WHY GLOBAL AVERAGE POOLING?
#   Instead of flattening (which depends on input size),
#   GAP averages each filter across time giving same output
#   size regardless of input length. More robust.
#
# WHY L2 NORMALIZE THE OUTPUT?
#   Cosine similarity works best when vectors are unit length.
#   Formula: z_norm = z / (||z|| + epsilon)
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class ECGEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ECGEncoder, self).__init__()

        self.conv1 = ConvBlock(1,   64,  kernel_size=7)
        self.conv2 = ConvBlock(64,  128, kernel_size=5)
        self.conv3 = ConvBlock(128, 256, kernel_size=3)

        self.embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.projector = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)
        x = self.embedding(x)
        z = self.projector(x)
        z = F.normalize(z, p=2, dim=1)
        return z

    def get_embedding(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)
        return self.embedding(x)
