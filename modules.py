import torch
from torch import nn, zeros_like
from torch.nn import functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import init_weights, get_padding


LRELU_SLOPE = 0.1


class WN(torch.nn.Module):
  """WaveNet module, used for PriorEncoder (ContentEncoder and Flow) and PosteriorEncoder."""
  def __init__(self,
    hidden_channels: int,       # Feature dimension size of hidden layers
    kernel_size:     int,       # The size of convolution kernel
    n_layers:        int,       # The number of conv layers
    gin_channels:    int   = 0, # Feature dimension size of conditioning input (`0` means no conditioning)
    p_dropout:       float = 0, # Dropout probability
  ):
    super(WN, self).__init__()

    assert kernel_size % 2 == 1, f"kernel should be odd number, but {kernel_size}"

    # Params
    self.hidden_channels, self.n_layers, self.gin_channels = hidden_channels, n_layers, gin_channels

    # Conditioning - SegFC for all layers
    if gin_channels != 0:
      self.cond_layer = weight_norm(Conv1d(gin_channels, 2*hidden_channels*n_layers, 1))

    # Dropout
    self.drop = nn.Dropout(p_dropout)

    self.in_layers       = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    for i in range(n_layers):
      # WaveNet layer, Res[Conv-GAU-SegFC-(half_to_final)]
      ## Conv,  doubled channel for gated activation unit
      self.in_layers.append(weight_norm(Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, padding="same")))
      ## SegFC, doubled channel for residual/skip connections
      res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
      self.res_skip_layers.append(weight_norm(Conv1d(hidden_channels, res_skip_channels, 1, padding="same")))

    # Remnants
    self.kernel_size, self.dilation_rate, self.p_dropout = kernel_size, 1, p_dropout

  def forward(self, x, x_mask, g=None):
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
      acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
      acts = self.drop(acts)

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
class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            #print(xt.size())
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            #print(xt.size())
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
#### /iSTFTNet modules #########################################################


#### Flow modules ##############################################################
class Flip(nn.Module):
  """Flow flip."""
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x


class ResidualCouplingLayer(nn.Module):
  """Flow layer."""
  def __init__(self,
      channels:        int, # Feature dimension size of input
      hidden_channels: int, # Feature dimension size of hidden layers
      kernel_size,          # WaveNet module's convolution kernel size
      dilation_rate,        # WaveNet module's dilation factor per layer
      n_layers,             # WaveNet module's the number of convolution layers
      p_dropout=0,          # WaveNet module's dropout probability
      gin_channels=0,       # WaveNet module's feature dimension size of conditioning input (`0` means no conditioning)
      mean_only=False       #
  ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    assert dilation_rate == 1, f"Support for 'dilation_rate>1' is dropped, but now {dilation_rate}"
    super().__init__()

    # Params
    self.mean_only = mean_only
    self.half_channels = channels // 2

    # PreNet - SegFC, adjust feature dimension size
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    # MainNet - WaveNet
    self.enc = WN(hidden_channels, kernel_size, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    # PostNet - SegFC, adjust feature dimension size to normal distribution parameters
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

    # Remnants
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers = channels, hidden_channels, kernel_size, dilation_rate, n_layers

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)

    # SegFC-WN-SegFC
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask

    # Normal distribution
    ## Parameters
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)
    ## ? (sampling-like magic. You should study Flow)
    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x
#### /Flow modules #############################################################
