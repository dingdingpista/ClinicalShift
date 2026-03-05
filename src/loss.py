
# =============================================================
# src/loss.py
# ClinicalShift — NT-Xent Contrastive Loss
# =============================================================
#
# NT-Xent = Normalized Temperature-scaled Cross Entropy Loss
# Originally proposed in SimCLR (Chen et al., 2020)
# Paper: https://arxiv.org/abs/2002.05709
#
# FORMULA:
#
#   L(i,j) = -log [
#       exp(sim(z_i, z_j) / tau)
#       ----------------------------------------
#       sum_{k=1}^{2N} 1[k!=i] exp(sim(z_i,z_k) / tau)
#   ]
#
# WHERE:
#   sim(u,v) = (u . v) / (||u|| * ||v||)  <- cosine similarity
#   tau      = temperature parameter       <- controls sharpness
#   N        = batch size
#   2N       = batch size x 2 (original + augmented views)
#   1[k!=i]  = exclude self-similarity
#
# INTUITION:
#   - Numerator:   similarity between positive pair (same ECG)
#   - Denominator: similarity against ALL other samples
#   - Loss is LOW when positive pair is more similar than negatives
#   - Loss is HIGH when negatives are confused with positives
#
# TEMPERATURE tau:
#   - Low  tau (0.1): very sharp — hard negatives dominate
#   - High tau (1.0): soft — all negatives treated equally
#   - We use tau=0.5 as a balanced default
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Contrastive Loss for SimCLR-style training.

    Takes two batches of embeddings (from two augmented views
    of the same data) and computes the contrastive loss.

    Args:
        temperature : tau in the formula (default 0.5)
        device      : 'cuda' or 'cpu'
    """

    def __init__(self, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device      = device

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss between two sets of embeddings.

        Args:
            z_i : embeddings from augmented view 1 (N, embedding_dim)
            z_j : embeddings from augmented view 2 (N, embedding_dim)

        Returns:
            loss : scalar tensor
        """
        N = z_i.shape[0]   # batch size

        # ---- Step 1: Concatenate both views ----
        # z = [z_i_1, z_i_2, ..., z_i_N, z_j_1, z_j_2, ..., z_j_N]
        # Shape: (2N, embedding_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # ---- Step 2: Compute all pairwise cosine similarities ----
        # sim[a,b] = cosine_similarity(z_a, z_b)
        # Shape: (2N, 2N)
        # Since embeddings are already L2-normalized (from model.py),
        # cosine similarity = dot product
        sim = torch.mm(z, z.T) / self.temperature

        # ---- Step 3: Remove self-similarity (diagonal) ----
        # A sample should not be compared with itself
        # Set diagonal to very large negative number
        mask_self = torch.eye(2*N, dtype=torch.bool).to(self.device)
        sim = sim.masked_fill(mask_self, float('-inf'))

        # ---- Step 4: Create positive pair labels ----
        # For sample i (0 to N-1), its positive is at position i+N
        # For sample i+N (N to 2N-1), its positive is at position i
        # Labels: [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
        labels = torch.cat([
            torch.arange(N, 2*N),   # positives for first N samples
            torch.arange(0, N)      # positives for last N samples
        ]).to(self.device)

        # ---- Step 5: Cross entropy loss ----
        # This is equivalent to the NT-Xent formula:
        # -log( exp(sim_positive) / sum(exp(sim_all_negatives)) )
        loss = F.cross_entropy(sim, labels)

        return loss
