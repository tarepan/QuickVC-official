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


def rand_slice_segments(series: Tensor, segment_size: int) -> tuple[Tensor, Tensor]:
  """Slice a segment randomly from a series.

  Args:
    series       :: (B, Feat, Frame)     - Slice-target series
    segment_size                         - Segment length
  Returns:
    segment      :: (B, Feat, Frame=seg) - Sliced segments
    indice_start :: (B,)                 - Randomly-selected slice start indice
  """
  # Random segment start indice
  b, _, t = series.size()
  indice_start_max = t - segment_size + 1
  indice_start = (torch.rand([b]).to(device=series.device) * indice_start_max).to(dtype=torch.long)

  # Slice
  segment = slice_segments(series, indice_start, segment_size)

  return segment, indice_start
#### /Slice #########################################################################################################
