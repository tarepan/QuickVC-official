"""Loss functions."""

import torch
from torch import Tensor
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


def feature_loss(fmap_r: list[list[Tensor]], fmap_g: list[list[Tensor]]) -> Tensor:
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.detach()
      loss += F.l1_loss(rl, gl)

  return loss * 2


def discriminator_loss(disc_real_outputs: list[Tensor], disc_generated_outputs: list[Tensor]) -> tuple[Tensor, list[float], list[float]]:
  # For forward
  loss = 0
  # For logging
  r_losses: list[float] = []
  g_losses: list[float] = []

  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    # Forward
    r_loss = F.mse_loss(torch.ones_like( dr), dr)
    g_loss = F.mse_loss(torch.zeros_like(dg), dg)
    loss += (r_loss + g_loss)
    # For logging
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
  loss = 0
  # For logging
  gen_losses: list[Tensor] = []
  for dg in disc_outputs:
    # Forward
    l = F.mse_loss(torch.ones_like(dg), dg)
    loss += l
    # For logging
    gen_losses.append(l)

  return loss, gen_losses


def kl_loss(z_p: Tensor, logs_q: Tensor, m_p: Tensor, logs_p: Tensor) -> Tensor:
  """
  Args:
    z_p :: (B, Feat, Frame)
    logs_q : [b, h, t_t]
    m_p: [b, h, t_t]
    logs_p: [b, h, t_t]
  """
  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  l = torch.mean(kl)

  return l


# Basic code from TomokiHayashi©2019 under MIT License
# Refactored by   tarepan©2023       under MIT License
def _spectral_convergenge_loss(x_mag: Tensor, y_mag: Tensor) -> Tensor:
    """
    Spectral convergence loss.

    Args:
        x_mag :: (B, Frame, Freq) - Magnitude spectrogram of   predicted signal
        y_mag :: (B, Frame, Freq) - Magnitude spectrogram of groundtruth signal
    Returns:
              ::                  - Spectral convergence loss
    """
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


def _log_stft_magnitude_loss(x_mag: Tensor, y_mag: Tensor) -> Tensor:
    """Log STFT magnitude loss.
    Args:
        x_mag :: (B, Frame, Freq) - Magnitude spectrogram of   predicted signal
        y_mag :: (B, Frame, Freq) - Magnitude spectrogram of groundtruth signal
    Returns:
              ::                  - Log STFT magnitude loss
    """
    return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class _STFTLoss(torch.nn.Module):
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
        sc_loss  = _spectral_convergenge_loss(x_mag, y_mag)
        mag_loss =   _log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class _MultiResolutionSTFTLoss(torch.nn.Module):
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
            self.stft_losses += [_STFTLoss(fs, ss, wl)]

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
#/


def subband_stft_loss(h, y_mb, y_hat_mb):
  sub_stft_loss = _MultiResolutionSTFTLoss(h.train.fft_sizes, h.train.hop_sizes, h.train.win_lengths)
  y_mb =  y_mb.view(-1, y_mb.size(2))
  y_hat_mb = y_hat_mb.view(-1, y_hat_mb.size(2))
  sub_sc_loss, sub_mag_loss = sub_stft_loss(y_hat_mb[:, :y_mb.size(-1)], y_mb)
  return sub_sc_loss+sub_mag_loss
