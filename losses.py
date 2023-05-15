"""Loss functions."""

import torch
from torch import Tensor

from stft_loss import MultiResolutionSTFTLoss


def feature_loss(fmap_r: list[list[Tensor]], fmap_g: list[list[Tensor]]) -> Tensor:
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.detach()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2


def discriminator_loss(disc_real_outputs: list[Tensor], disc_generated_outputs: list[Tensor]) -> tuple[Tensor, list[float], list[float]]:
  loss = 0
  r_losses: list[float] = []
  g_losses: list[float] = []

  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    r_loss = torch.mean((1 - dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
  loss = 0
  gen_losses: list[Tensor] = []
  for dg in disc_outputs:
    l = torch.mean((1 - dg)**2)
    gen_losses.append(l)
    loss += l

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

def subband_stft_loss(h, y_mb, y_hat_mb):
  sub_stft_loss = MultiResolutionSTFTLoss(h.train.fft_sizes, h.train.hop_sizes, h.train.win_lengths)
  y_mb =  y_mb.view(-1, y_mb.size(2))
  y_hat_mb = y_hat_mb.view(-1, y_hat_mb.size(2))
  sub_sc_loss, sub_mag_loss = sub_stft_loss(y_hat_mb[:, :y_mb.size(-1)], y_mb)
  return sub_sc_loss+sub_mag_loss

