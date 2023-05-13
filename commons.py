"""Common functions."""

import torch
from torch import Tensor


#### Networks #######################################################################################################
def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def sequence_mask(length: Tensor, max_length: int) -> Tensor:
  """
  Args:
      length :: ()
      max_length   - 
  Returns:
      :: BoolTensor[()] - Mask
  """
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)
#### /Networks ######################################################################################################


#### Slice ##########################################################################################################
def slice_segments(series: Tensor, indice_start: Tensor, segment_size: int):
  """Slice a specified segment from a series.

  Args:
    series       :: (B, Feat, T) - Slice-target series
    indice_start :: (B,)         - Each series's segment start index
    segment_size                 - Segment length
  """
  # Output container :: (B, Feat, T=seg)
  ret = torch.zeros_like(series[:, :, :segment_size])

  for idx_b in range(series.size(0)):
    idx_start = indice_start[idx_b]
    idx_end = idx_start + segment_size
    ret[idx_b] = series[idx_b, :, idx_start : idx_end]
  return ret


def rand_slice_segments(series: Tensor, x_lengths: Tensor | None, segment_size: int) -> tuple[Tensor, Tensor]:
  """Slice a segment randomly from a series.

  Args:
    series       :: (B, Feat, Frame)     - Slice-target series
    x_lengths    :: (B,)                 - Effective length of a series
    segment_size                         - Segment length
  Returns:
    segment      :: (B, Feat, Frame=seg) - Sliced segments
    indice_start :: (B,)                 - Randomly-selected slice start indice
  """
  # Random segment start indice
  b, _, t = series.size()
  series_lengths = x_lengths if x_lengths else t
  indice_start_max = series_lengths - segment_size + 1
  indice_start = (torch.rand([b]).to(device=series.device) * indice_start_max).to(dtype=torch.long)

  # Slice
  segment = slice_segments(series, indice_start, segment_size)

  return segment, indice_start


def rand_spec_segments(series: Tensor, x_lengths: Tensor | None, segment_size: int) -> tuple[Tensor, Tensor]:
  # Random segment start indice
  b, _, t = series.size()
  series_lengths = x_lengths if x_lengths else t
  indice_start_max = series_lengths - segment_size # +1 (in rand_slice_segments)
  indice_start = (torch.rand([b]).to(device=series.device) * indice_start_max).to(dtype=torch.long)

  # Slice
  segment = slice_segments(series, indice_start, segment_size)

  return segment, indice_start
#### /Slice #########################################################################################################


#### Train ##########################################################################################################
def count_grad_norm(parameters) -> float:
  """Count total gradient norm."""

  norm_type = 2.

  # Select Gradient-active parameters
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))

  total_norm: float = 0.
  for param in parameters:
    # Norm calculation
    param_norm = param.grad.data.norm(norm_type)
    # Counting
    total_norm += param_norm.item() ** norm_type

  total_norm = total_norm ** (1. / norm_type)
  return total_norm
#### /Train #########################################################################################################
