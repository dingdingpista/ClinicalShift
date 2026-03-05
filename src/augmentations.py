
# =============================================================
# src/augmentations.py
# ClinicalShift — ECG Augmentation Functions
# =============================================================
#
# For contrastive learning we need two DIFFERENT augmented
# views of the same ECG window. The encoder must learn
# representations that are invariant to these augmentations.
#
# Each augmentation is applied RANDOMLY with a given
# probability — so two calls always give different results.
# =============================================================

import numpy as np


def add_gaussian_noise(signal, std=0.05):
    """
    Add random Gaussian noise to the signal.

    Simulates electrode contact noise in clinical recordings.

    Formula:
        x_noisy = x + N(0, std)
        where N(0, std) is Gaussian noise with mean=0

    Args:
        signal : numpy array of shape (250,)
        std    : standard deviation of noise (default 0.05)

    Returns:
        noisy signal of same shape
    """
    noise = np.random.normal(0, std, size=signal.shape)
    return signal + noise


def random_scaling(signal, scale_range=(0.8, 1.2)):
    """
    Multiply signal by a random scaling factor.

    Simulates variation in ECG electrode placement and
    patient body composition affecting signal amplitude.

    Formula:
        x_scaled = x * s,  s ~ Uniform(scale_min, scale_max)

    Args:
        signal      : numpy array of shape (250,)
        scale_range : (min, max) range for scaling factor

    Returns:
        scaled signal of same shape
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def random_time_shift(signal, max_shift=20):
    """
    Shift signal left or right by a random amount.

    Simulates variation in when the recording window starts
    relative to the cardiac cycle.

    Formula:
        x_shifted[i] = x[i + shift]  (with zero padding)
        shift ~ Uniform(-max_shift, max_shift)

    Args:
        signal    : numpy array of shape (250,)
        max_shift : maximum samples to shift (default 20)

    Returns:
        shifted signal of same shape
    """
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)


def random_masking(signal, max_mask=30):
    """
    Zero out a random segment of the signal.

    Simulates signal dropout or electrode disconnection.
    Forces the encoder to be robust to missing data.

    Formula:
        x_masked[start:start+length] = 0
        start  ~ Uniform(0, L - max_mask)
        length ~ Uniform(0, max_mask)

    Args:
        signal   : numpy array of shape (250,)
        max_mask : maximum length of masked segment

    Returns:
        masked signal of same shape
    """
    masked = signal.copy()
    length = np.random.randint(0, max_mask)
    start  = np.random.randint(0, len(signal) - max_mask)
    masked[start : start + length] = 0.0
    return masked


def augment(signal):
    """
    Apply a random combination of augmentations.

    Each augmentation is applied independently with
    probability 0.5 — so each call gives a different result.

    Args:
        signal : numpy array of shape (250,)

    Returns:
        augmented signal of same shape (250,)
    """
    if np.random.random() < 0.5:
        signal = add_gaussian_noise(signal)

    if np.random.random() < 0.5:
        signal = random_scaling(signal)

    if np.random.random() < 0.5:
        signal = random_time_shift(signal)

    if np.random.random() < 0.5:
        signal = random_masking(signal)

    return signal


def create_augmented_pair(signal):
    """
    Create two DIFFERENT augmented views of the same signal.

    This is the core of contrastive learning — the encoder
    must learn that aug1 and aug2 came from the same source.

    Args:
        signal : numpy array of shape (250,)

    Returns:
        aug1, aug2 : two differently augmented versions
    """
    aug1 = augment(signal.copy())
    aug2 = augment(signal.copy())
    return aug1, aug2
