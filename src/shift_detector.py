
# =============================================================
# src/shift_detector.py
# ClinicalShift — Mahalanobis Distance Shift Detector
# =============================================================
#
# ALGORITHM:
#
#   FITTING (on training/in-distribution data):
#     1. Extract embeddings for all training samples
#     2. Compute mean vector:
#        mu = (1/N) * sum(z_i)
#     3. Compute covariance matrix:
#        Sigma = (1/N) * sum((z_i - mu)(z_i - mu)^T)
#     4. Compute inverse covariance:
#        Sigma_inv = inverse(Sigma + epsilon*I)
#        (epsilon*I added for numerical stability)
#
#   SCORING (on any test sample):
#     1. Extract embedding z
#     2. Compute Mahalanobis distance:
#        D_M(z) = sqrt((z - mu)^T * Sigma_inv * (z - mu))
#     3. High D_M = out-of-distribution = SHIFT DETECTED
#
#   THRESHOLD:
#     threshold = mean(D_M_train) + 2 * std(D_M_train)
#     Samples above threshold are flagged as shifted.
#     This covers ~95% of in-distribution data (2-sigma rule).
# =============================================================

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class MahalanobisShiftDetector:
    """
    Distribution shift detector using Mahalanobis distance.

    Fits a multivariate Gaussian to the training embeddings,
    then scores test samples by their distance from this
    distribution.

    Args:
        model  : trained ECGEncoder
        device : cuda or cpu
    """

    def __init__(self, model, device='cuda'):
        self.model   = model
        self.device  = device
        self.mu      = None   # mean embedding vector
        self.sigma_inv = None # inverse covariance matrix
        self.threshold = None # detection threshold
        self.train_scores = None

    def _extract_embeddings(self, dataset, batch_size=256):
        """
        Extract embeddings for all samples in a dataset.

        Args:
            dataset    : ECGDataset
            batch_size : samples per batch

        Returns:
            embeddings : numpy array (N, embedding_dim)
        """
        loader = DataLoader(dataset, batch_size=batch_size,
                           shuffle=False)
        all_embeddings = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in tqdm(loader, desc="Extracting embeddings"):
                batch_x = batch_x.to(self.device)
                # Use 128-dim embedding (richer than 64-dim projection)
                emb = self.model.get_embedding(batch_x)
                all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)   # (N, 128)

    def fit(self, train_dataset):
        """
        Fit the detector on in-distribution training data.

        Computes mu and Sigma from training embeddings.

        Args:
            train_dataset : ECGDataset of in-distribution data
        """
        print("Fitting shift detector on training data...")

        # Step 1: Extract all training embeddings
        embeddings = self._extract_embeddings(train_dataset)
        print(f"  Training embeddings shape: {embeddings.shape}")

        # Step 2: Compute mean vector
        # mu = (1/N) * sum(z_i)
        # Shape: (128,)
        self.mu = np.mean(embeddings, axis=0)

        # Step 3: Compute covariance matrix
        # Sigma = (1/N) * sum((z_i - mu)(z_i - mu)^T)
        # Shape: (128, 128)
        centered = embeddings - self.mu
        sigma = np.cov(centered.T)   # (128, 128)

        # Step 4: Add epsilon*I for numerical stability
        # Prevents singular matrix inversion
        epsilon = 1e-6
        sigma += epsilon * np.eye(sigma.shape[0])

        # Step 5: Compute inverse covariance
        self.sigma_inv = np.linalg.inv(sigma)

        # Step 6: Compute training scores for threshold
        self.train_scores = self._mahalanobis_scores(embeddings)

        # Step 7: Set threshold = mean + 2*std (2-sigma rule)
        # Covers ~95% of in-distribution data
        self.threshold = (np.mean(self.train_scores) +
                         2 * np.std(self.train_scores))

        print(f"  Mean train score : {np.mean(self.train_scores):.4f}")
        print(f"  Std  train score : {np.std(self.train_scores):.4f}")
        print(f"  Threshold (2σ)   : {self.threshold:.4f}")
        print("✅ Detector fitted!")

    def _mahalanobis_scores(self, embeddings):
        """
        Compute Mahalanobis distance for a set of embeddings.

        Formula:
            D_M(z) = sqrt((z-mu)^T * Sigma_inv * (z-mu))

        Args:
            embeddings : numpy array (N, 128)

        Returns:
            scores : numpy array (N,) of distances
        """
        centered = embeddings - self.mu   # (N, 128)

        # Vectorized Mahalanobis:
        # scores[i] = sqrt(centered[i] @ Sigma_inv @ centered[i]^T)
        temp   = centered @ self.sigma_inv      # (N, 128)
        scores = np.sqrt(np.sum(temp * centered, axis=1))  # (N,)
        return scores

    def score(self, dataset):
        """
        Compute shift scores for a dataset.

        Args:
            dataset : ECGDataset to score

        Returns:
            scores : numpy array (N,) — higher = more shifted
        """
        embeddings = self._extract_embeddings(dataset)
        return self._mahalanobis_scores(embeddings)

    def predict(self, dataset):
        """
        Predict whether each sample is shifted or not.

        Args:
            dataset : ECGDataset

        Returns:
            predictions : numpy array (N,)
                          0 = in-distribution
                          1 = distribution shift detected
        """
        scores = self.score(dataset)
        return (scores > self.threshold).astype(int)
