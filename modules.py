import torch
from torch import Tensor, zeros_like, nn
from torch.nn import functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import init_weights, get_padding


LRELU_SLOPE = 0.1


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: Tensor) -> Tensor:
  """Gated Activation Unit with additive conditioning, GAU(x, cond) = σ(split_1(x+cond)) * tanh(split_2(x+cond))
  
  Args:
    input_a    :: (B, Feat=2h, Frame)
    input_b    :: (B, Feat=2h, Frame)
    n_channels
  Returns:
               :: (B, Feat=h,  Frame)
  """
  n_channels_int = n_channels[0]

  # Additive conditioning :: (B, Feat=2h, Frame) + (B, Feat=2h, Frame) -> (B, Feat=2h, Frame)
  in_act = input_a + input_b

  # GAU :: (B, Feat=h, Frame) * (B, Feat=h, Frame) -> (B, Feat=h, Frame)
  t_act =    torch.tanh(in_act[:, :n_channels_int])
  s_act = torch.sigmoid(in_act[:, n_channels_int:])
  acts = t_act * s_act
  return acts


class WN(torch.nn.Module):
  """WaveNet module, Res[Conv-CondGAU-SegFC-(half_skip)]xN-skipsum."""
  def __init__(self,
    hidden_channels: int, # Feature dimension size of input/hidden/output
    kernel_size:     int, # The size of convolution kernel
    n_layers:        int, # The number of conv layers
    gin_channels:    int, # Feature dimension size of conditioning input (`0` means no conditioning)
  ):
    super().__init__()

    assert kernel_size % 2 == 1, f"kernel should be odd number, but {kernel_size}"

    # Params
    self.hidden_channels, self.n_layers, self.gin_channels = hidden_channels, n_layers, gin_channels

    # Conditioning - SegFC for all layers
    if gin_channels != 0:
      self.cond_layer = weight_norm(Conv1d(gin_channels, 2*hidden_channels * n_layers, 1))

    # Dropout
    self.drop = nn.Dropout(0.)

    self.in_layers       = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    for i in range(n_layers):
      # WaveNet layer, Res[Conv-GAU-SegFC-(half_to_final)]
      ## Conv,  doubled channel for gated activation unit
      self.in_layers.append(      weight_norm(Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, padding="same")))
      ## SegFC, doubled channel for residual/skip connections
      res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
      self.res_skip_layers.append(weight_norm(Conv1d(hidden_channels,   res_skip_channels,           1, padding="same")))

  def forward(self, x: Tensor, x_mask, g: Tensor | None = None):
    """
    Args:
      x      :: (B, Feat=h, Frame) - Input series
      x_mask ::                    -
      g      :: maybe (B, Cond, Frame) - Conditioning series
    Returns:
             :: (B, Feat=h, Frame)
    """
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    # Final output :: (B, Feat=h, Frame) - Output, skip connections connected
    output = zeros_like(x)

    # Conditioning :: (B, cond, Frame) -> (B, 2h*n_layers, Frame) - Conditioning for all layers at once
    if g is not None:
      g = self.cond_layer(g)

    # Layers
    for i in range(self.n_layers):
      # x :: (B, Feat=h, Frame) - layer IO

      # Conv :: (B, Feat=h, Frame) -> (B, Feat=2h, Frame)
      x_in = self.in_layers[i](x)

      # Conditioning of layer :: (B, 2h*n_layers, Frame) -> (B, Feat=2h, Frame)
      if g is not None:
        cond_offset = i * (2 * self.hidden_channels)
        g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels]
      else:
        g_l = zeros_like(x_in)

      # Activation :: (B, Feat=2h, Frame) & (B, Feat=2h, Frame) -> (B, Feat=h, Frame) - Gated activation unit with additive conditioning
      acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)

      # Residual/Skip :: (B, Feat=h, Frame) -> (B, Feat=2h, Frame) | (B, Feat=h, Frame) - First half for Res, second half for skip
      res_skip_acts = self.res_skip_layers[i](acts)
      ## Residual connection :: (B, Feat=h, Frame) + (B, Feat=h, Frame) -> (B, Feat=h, Frame)
      if i < self.n_layers - 1:
        x = (x + res_skip_acts[:, :self.hidden_channels]) * x_mask
      ## Skip connection :: (B, Feat=h, Frame) + (B, Feat=h, Frame) -> (B, Feat=h, Frame)
        output = output + res_skip_acts[:, self.hidden_channels:]
      else:
        # :: (B, Feat=h, Frame) + (B, Feat=h, Frame) -> (B, Feat=h, Frame) - Skip connection only for last layer
        output = output + res_skip_acts

    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)


#### iSTFTNet modules ##########################################################
# Totally same as VITS in QuickVC-official. Refactored by tarepan.

class ResBlock1(torch.nn.Module):
    """Residual block, Res[LReLU-DilConv-LReLU-Conv] x3."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size,                       padding="same")),
            weight_norm(Conv1d(channels, channels, kernel_size,                       padding="same")),
            weight_norm(Conv1d(channels, channels, kernel_size,                       padding="same"))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x,  LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
#### /iSTFTNet modules #########################################################


#### Flow modules ##############################################################
class Flip(nn.Module):
  """Flow flip."""
  def forward(self, x: Tensor, *args, reverse: bool = False, **kwargs):
    """:: (B, Feat, T) -> (B, Feat, T)"""
    x = torch.flip(x, [1])
    return x
    # if not reverse:
    #   logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)


class ResidualCouplingLayer(nn.Module):
  """Flow layer."""
  def __init__(self,
      channels:        int, # Feature dimension size of input/output
      hidden_channels: int, # Feature dimension size of hidden layers
      kernel_size:     int, # WaveNet module's convolution kernel size
      n_layers:        int, # WaveNet module's the number of convolution layers
      gin_channels:    int, # Feature dimension size of conditioning input (`0` means no conditioning)
  ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()

    # Params
    self.half_channels = channels // 2

    # PreNet  :: (B, Feat=i/2, T) -> (B, Feat=h,   T) - SegFC
    # MainNet :: (B, Feat=h,   T) -> (B, Feat=h,   T) - WaveNet
    # PostNet :: (B, Feat=h,   T) -> (B, Feat=i/2, T) - SegFC
    self.pre  = Conv1d(self.half_channels, hidden_channels,    1)
    self.enc  = WN(hidden_channels, kernel_size, n_layers, gin_channels)
    self.post = Conv1d(hidden_channels,    self.half_channels, 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None, reverse: bool = False):
    """
    Args:
      x       :: (B, Feat=i, T) - Input
      x_mask
      g                         - Conditioning input
      reverse                   - Whether to 'reverse' flow or not
    Returns:
              :: (B, Feat=i, T)
    """
    # Split
    x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)

    # SegFC-WN-SegFC :: (B, Feat=i/2, T) -> (B, Feat=i/2, T)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    m = self.post(h) * x_mask

    # Normal distribution :: (B, Feat=i/2, T) & (B, Feat=i/2, T) -> (B, Feat=i/2, T) - (?) sampling-like magic. You should study Flow.
    if not reverse:
      x1 = m + x1 * x_mask
      # logdet = torch.sum(logs, [1,2])
    else:
      x1 = (x1 - m) * x_mask

    # Cat :: (B, Feat=i/2, T) & (B, Feat=i/2, T) -> (B, Feat=i, T)
    x = torch.cat([x0, x1], 1)

    return x
#### /Flow modules #############################################################
