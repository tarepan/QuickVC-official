# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
from torch import Tensor
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device), return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def spectral_convergenge_loss(x_mag: Tensor, y_mag: Tensor) -> Tensor:
    """
    Spectral convergence loss.

    Args:
        x_mag :: (B, Frame, Freq) - Magnitude spectrogram of   predicted signal
        y_mag :: (B, Frame, Freq) - Magnitude spectrogram of groundtruth signal
    Returns:
              ::                  - Spectral convergence loss
    """
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


def log_stft_magnitude_loss(x_mag: Tensor, y_mag: Tensor) -> Tensor:
    """Log STFT magnitude loss.
    Args:
        x_mag :: (B, Frame, Freq) - Magnitude spectrogram of   predicted signal
        y_mag :: (B, Frame, Freq) - Magnitude spectrogram of groundtruth signal
    Returns:
              ::                  - Log STFT magnitude loss
    """
    return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size: int = 1024, shift_size: int = 120, win_length: int = 600, window: str = "hann_window"):
        """Initialize STFT loss module."""
        super().__init__()
        self.fft_size, self.shift_size, self.win_length = fft_size, shift_size, win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        Args:
            x :: (B, T) -   Predicted signal
            y :: (B, T) - Groundtruth signal
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss  = spectral_convergenge_loss(x_mag, y_mag)
        mag_loss =   log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
        fft_sizes:   list[int] = [1024, 2048, 512],
        hop_sizes:   list[int] = [ 120,  240,  50],
        win_lengths: list[int] = [ 600, 1200, 240],
        window:      str       = "hann_window"
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes   - List of FFT sizes.
            hop_sizes   - List of hop sizes.
            win_lengths - List of window lengths.
            window      - Window function type name
        """
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        Args:
            x :: (B, T) -   Predicted signal
            y :: (B, T) - Groundtruth signal
        Returns:
              ::        - Multi resolution spectral convergence loss
              ::        - Multi resolution log STFT magnitude loss
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for stft_loss in self.stft_losses:
            sc_l, mag_l = stft_loss(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
