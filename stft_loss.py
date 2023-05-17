# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
from torch import Tensor
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


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

    def __init__(self, fft_size: int, shift_size: int, win_length: int):
        super().__init__()
        self.stft = Spectrogram(fft_size, win_length, shift_size, power=1.0)

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        Args:
            x :: (B, T) -   Predicted signal
            y :: (B, T) - Groundtruth signal
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.stft(x).transpose(2, 1)
        y_mag = self.stft(y).transpose(2, 1)
        sc_loss  = spectral_convergenge_loss(x_mag, y_mag)
        mag_loss =   log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
        fft_sizes:   list[int], # List of FFT sizes.      e.g. [1024, 2048, 512]
        hop_sizes:   list[int], # List of hop sizes.      e.g. [ 120,  240,  50]
        win_lengths: list[int], # List of window lengths. e.g. [ 600, 1200, 240]
    ):
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl)]

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
