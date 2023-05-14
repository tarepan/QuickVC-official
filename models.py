import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import init_weights, get_padding
import modules
from modules import ResBlock1
from pqmf import PQMF
from stft import TorchSTFT


class ResidualCouplingBlock(nn.Module):
  """Flow, chain of 'change of variable' parameterized by SegFC-WaveNet-SegFC."""
  def __init__(self,
      input_channels:  int, # Feature dimension size of input
      output_channels: int, # Feature dimension size of output
      hidden_channels: int, # Feature dimension size of hidden layers
      kernel_size:     int, # WaveNet module's convolution kernel size
      n_layers:        int, # WaveNet module's the number of convolution layers
      n_flows:         int, # The number of Flow layers
      gin_channels:    int, # Feature dimension size of conditioning input (`0` means no conditioning)
    ):
    super().__init__()

    # Params
    assert input_channels == output_channels, f"I/O should match their feature dimension, but {input_channels} != {output_channels}"
    channels = input_channels # Feature dimension size of input/output

    self.flows = nn.ModuleList()
    for _ in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, n_layers, gin_channels))
      self.flows.append(modules.Flip())

  def forward(self, x: Tensor, g: Tensor | None = None, reverse: bool = False):
    """
    Args:
        x      :: (B, Feat, Frame) - Input
        g      :: (B, Feat, Frame) - Condioning input
        reverse                    - Whether to 'reverse' flow or not
    Returns:
               :: (B, Feat, Frame)
    """
    flows = self.flows if not reverse else reversed(self.flows)
    for flow in flows:
      x = flow(x, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  """Normal distribution parameterized by SegFC-WaveNet-SegFC.
  """

  def __init__(self,
      in_channels:     int, # Feature dimension size of input
      out_channels:    int, # Feature dimension size of output
      hidden_channels: int, # Feature dimension size of hidden layer
      kernel_size:     int, # WaveNet module's convolution kernel size
      n_layers:        int, # WaveNet module's the number of layers
      gin_channels:    int, # Feature dimension size of conditionings
    ):
    super().__init__()

    self.out_channels = out_channels

    # PreNet  :: (B, Feat=i, Frame) -> (B, Feat=h,  Frame) - SegFC
    # MainNet :: (B, Feat=h, Frame) -> (B, Feat=h,  Frame) - WaveNet
    # PostNet :: (B, Feat=h, Frame) -> (B, Feat=2o, Frame) - SegFC
    self.pre  = Conv1d(in_channels,     hidden_channels,  1)
    self.enc  = modules.WN(hidden_channels, kernel_size, n_layers, gin_channels)
    self.proj = Conv1d(hidden_channels, 2 * out_channels, 1)

  def forward(self, x: Tensor, g: Tensor | None = None):
    """
    Args:
      x         :: (B, Feat=i, Frame) - Input series
      x_lengths :: (B,)               - Effective lengths of each input series
      g                               - Conditioning input
    Returns:
      z         :: (B, Feat=o, Frame) - Sampled series
      m
      logs

    """

    # :: (B, Feat=i, Frame) -> (B, Feat=2o, Frame)
    x = self.pre(x)
    x = self.enc(x, g=g)
    stats = self.proj(x)

    # Normal distribution :: (B, Feat=2o, Frame) -> (B, Feat=o, Frame) - z ~ N(z|m,s)
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = m + torch.randn_like(m) * torch.exp(logs)
    return z, m, logs


class iSTFT_Generator(torch.nn.Module):
    def __init__(self,
                 initial_channel:          int, # Feature dimension size of latent input
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel: int, # Feature dimension size of first hidden layer
                 upsample_kernel_sizes,
                 n_fft:                    int, # n_fft    of Synthesis iSTFT
                 hop_istft:                int, # Hop size of Synthesis iSTFT
                 subbands:                 int, # The number of subbands
                 gin_channels:             int, # Feature dimension size of conditioning input
        ):
        super().__init__()

        # Params
        self.n_freq = n_fft // 2 + 1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # PreNet for latent :: (B, Feat=i,   Frame) -> (B, Feat=h, Frame) - Conv
        # PreNet for cond   :: (B, Feat=gin, Frame) -> (B, Feat=h, Frame) - SegFC
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, padding="same"))
        self.cond     =             Conv1d(gin_channels,    upsample_initial_channel, 1)

        # MainNet - UpMRF
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
        self.ups.apply(init_weights)
        self.resblocks = nn.ModuleList()
        ch = 0 # For typing
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))

        # PostNet :: (B, Feat, Frame) -> (B, Feat=2*freq, Frame)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.conv_post = weight_norm(Conv1d(ch, 2 * self.n_freq, 7, padding="same"))
        self.conv_post.apply(init_weights)

        # iSTFT
        self.stft = TorchSTFT(n_fft, hop_istft, n_fft)

    def forward(self, x: Tensor, g: Tensor):
        """Forward
        
        Args:
            x   :: (B, Feat, Frame) - Latent series
            g   :: (B, Feat, Frame) - Conditioning series
        Returns:
            out :: (B, 1, T)        - Generated waveform
        """

        # PreNet :: (B, Feat=i, Frame) & (B, Feat=gin, Frame) -> (B, Feat=h, Frame)
        x = self.conv_pre(x) + self.cond(g)

        # MainNet
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        # PostNet :: (B, Feat, Frame) -> (B, Feat=2freq, Frame)
        x = self.reflection_pad(x)
        x = self.conv_post(x)

        # iSTFT
        ## Split :: (B, Feat=2freq, Frame) -> 2x (B, Freq=freq, Frame)
        spec, phase = torch.split(x, [self.n_freq]*2, dim=1)
        ## Run :: 2x (B, Freq, Frame) -> (B, 1, T)
        spec  =           torch.exp(spec)
        phase = math.pi * torch.sin(phase)
        out = self.stft.inverse(spec, phase)

        return out, None

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Multiband_iSTFT_Generator(torch.nn.Module):
    def __init__(self,
                 initial_channel:          int, # Feature dimension size of latent input
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel: int, # Feature dimension size of first hidden layer
                 upsample_kernel_sizes,
                 n_fft:                    int, # n_fft    of Synthesis iSTFT
                 hop_istft:                int, # Hop size of Synthesis iSTFT
                 subbands:                 int, # The number of subbands
                 gin_channels:             int, # Feature dimension size of conditioning input
        ):
        super().__init__()

        # Params
        self.subbands = subbands
        self.n_freq = n_fft // 2 + 1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # PreNet for latent :: (B, Feat=i,   Frame) -> (B, Feat=h, Frame) - Conv
        # PreNet for cond   :: (B, Feat=gin, Frame) -> (B, Feat=h, Frame) - SegFC
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, padding="same"))
        self.cond     =             Conv1d(gin_channels,    upsample_initial_channel, 1)

        # MainNet - UpMRF
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u+1-i)//2,output_padding=1-i)))
        self.ups.apply(init_weights)
        self.resblocks = nn.ModuleList()
        ch = 0 # For typing
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))

        # PostNet :: (B, Feat, Frame) -> (B, Feat=band*2*freq, Frame)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands * 2 * self.n_freq, 7, padding="same"))
        self.subband_conv_post.apply(init_weights)

        # iSTFT
        self.stft = TorchSTFT(n_fft, hop_istft, n_fft)

        # Band synthesis
        self.pqmf = PQMF()

    def forward(self, x: Tensor, g: Tensor):
        """Forward
        
        Args:
            x        :: (B, Feat, Frame) - Latent series
            g        :: (B, Feat, Frame) - Conditioning series
        Returns:
            y_g_hat  :: (B,    1, T)     - Generated waveform
            y_mb_hat :: (B, Band, T')    - Generated subband waveforms
        """

        # PreNet :: (B, Feat=i, Frame) & (B, Feat=gin, Frame) -> (B, Feat=h, Frame)
        x = self.conv_pre(x) + self.cond(g)

        # MainNet
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        # PostNet :: (B, Feat, Frame) -> (B, Band, Feat=2freq, Frame)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1]//self.subbands, x.shape[-1]))

        # iSTFT
        ## Split :: (B, Band, Feat=2freq, Frm) -> 2x (B, Band, Freq=freq, Frm)
        spec, phase = torch.split(x, [self.n_freq]*2, dim=2)
        ## Band batching :: (B=b, Band=band, Freq, Frm) -> (B=b*band, Freq, Frm)
        spec  = torch.reshape(spec,  ( spec.shape[0] * self.subbands, self.n_freq,  spec.shape[-1]))
        phase = torch.reshape(phase, (phase.shape[0] * self.subbands, self.n_freq, phase.shape[-1]))
        ## Run :: (B, Freq, Frm) -> (B, 1, T')
        spec  =           torch.exp(spec)
        phase = math.pi * torch.sin(phase)
        y_mb_hat = self.stft.inverse(spec, phase)
        ## Band un-batching :: (B, 1, T') -> (B, Band, 1, T') -> (B, Band, T')
        y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1])).squeeze(-2)

        # Band synthesis :: (B, Band, T') -> (B, 1, T)
        y_g_hat = self.pqmf.synthesis(y_mb_hat)

        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(self,
                 initial_channel:          int, # Feature dimension size of latent input
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel: int, # Feature dimension size of first hidden layer
                 upsample_kernel_sizes,
                 n_fft:                    int, # n_fft    of Synthesis iSTFT
                 hop_istft:                int, # Hop size of Synthesis iSTFT
                 subbands:                 int, # The number of subbands
                 gin_channels:             int, # Feature dimension size of conditioning input
        ):
        super().__init__()

        # Params
        self.subbands = subbands
        self.n_freq = n_fft // 2 + 1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # PreNet for latent :: (B, Feat=i,   Frame) -> (B, Feat=h, Frame) - Conv
        # PreNet for cond   :: (B, Feat=gin, Frame) -> (B, Feat=h, Frame) - SegFC
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, padding="same"))
        self.cond     =             Conv1d(gin_channels,    upsample_initial_channel, 1)

        # MainNet - UpMRF
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u+1-i)//2,output_padding=1-i)))#这里k和u不是成倍数的关系，对最终结果很有可能是有影响的，会有checkerboard artifacts的现象
        self.ups.apply(init_weights)
        self.resblocks = nn.ModuleList()
        ch = 0 # For typing
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))

        # PostNet
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands * 2 * self.n_freq, 7, padding="same"))
        self.subband_conv_post.apply(init_weights)

        # iSTFT
        self.stft = TorchSTFT(n_fft, hop_istft, n_fft)

        # Band synthesis
        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = weight_norm(Conv1d(4, 1, 63, bias=False, padding=get_padding(63, 1)))
        self.multistream_conv_post.apply(init_weights)

    def forward(self, x: Tensor, g: Tensor):
        """Forward
        
        Args:
            x        :: (B, Feat, Frame) - Latent series
            g        :: (B, Feat, Frame) - Conditioning series
        Returns:
            y_g_hat  :: (B,    1, T)     - Generated waveform
            y_mb_hat :: (B, Band, T')    - Generated subband waveforms
        """

        # PreNet
        x = self.conv_pre(x) + self.cond(g)

        # MainNet
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        # PostNet :: (B, Feat, Frame) -> (B, Band, Feat=2freq, Frame)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1] // self.subbands, x.shape[-1]))

        # iSTFT
        ## Split :: (B, Band, Feat=2freq, Frm) -> 2x (B, Band, Freq=freq, Frm)
        spec, phase = torch.split(x, [self.n_freq]*2, dim=2)
        ## Band batching :: (B=b, Band=band, Freq, Frm) -> (B=b*band, Freq, Frm)
        spec  = torch.reshape(spec,  ( spec.shape[0] * self.subbands, self.n_freq,  spec.shape[-1]))
        phase = torch.reshape(phase, (phase.shape[0] * self.subbands, self.n_freq, phase.shape[-1]))
        ## Run :: (B, Freq, Frm) -> (B, 1, T')
        spec  =           torch.exp(spec)
        phase = math.pi * torch.sin(phase)
        y_mb_hat = self.stft.inverse(spec, phase)

        # Band synthesis :: (B=b*band, 1, T') -> (B=b, Band=band, T') -> (B, 1, T)? - Learnable-filter synthesis
        y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1])).squeeze(-2)
        y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter * self.subbands, stride=self.subbands)
        y_g_hat = self.multistream_conv_post(y_mb_hat)

        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()

        # Params
        self.period = period

        self.convs = nn.ModuleList([
            weight_norm(Conv2d(   1,   32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(  32,  128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d( 128,  512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d( 512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(1024, 1024, (kernel_size, 1),           1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(   1,   16, 15, 1,             padding= 7)),
            weight_norm(Conv1d(  16,   64, 41, 4, groups=  4, padding=20)),
            weight_norm(Conv1d(  64,  256, 41, 4, groups= 16, padding=20)),
            weight_norm(Conv1d( 256, 1024, 41, 4, groups= 64, padding=20)),
            weight_norm(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            weight_norm(Conv1d(1024, 1024,  5, 1,             padding= 2)),
        ])
        self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: Tensor):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Params
        periods = [2, 3, 5, 7, 11]

        discs =         [DiscriminatorS()]
        discs = discs + [DiscriminatorP(i) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y: Tensor, y_hat: Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
        
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed


class SynthesizerTrn(nn.Module):
  """QuickVC Generator for training (w/ posterior encoder, w/o wave-to-unit encoder)"""
  def __init__(self, 
    spec_channels:   int,        # Feature dimension size of linear spectrogram
    segment_size:    int,        # Decoder training segment size [frame]
    inter_channels:  int,        # Feature dimension size of latent z (both Zsi and Zsd)
    hidden_channels: int,        # Feature dimension size of WaveNet layers
    resblock_kernel_sizes:    list[int],          # Decoder
    resblock_dilation_sizes:  list[list[int]],    # Decoder
    upsample_rates:           list[int],          # Decoder
    upsample_initial_channel: int,                # Decoder
    upsample_kernel_sizes:    list[int],          # Decoder
    gen_istft_n_fft:          int,                # Decoder
    gen_istft_hop_size:       int,                # Decoder
    istft_vits:               bool       = False, # Decoder, Whether to use plain iSTFTNet Decoder
    ms_istft_vits:            bool       = False, # Decoder, Whether to use MS-iSTFTNet    Decoder
    mb_istft_vits:            bool       = False, # Decoder, Whether to use MB-iSTFTNet    Decoder
    subbands:                 bool | int = False, # Decoder, (maybe) The number of subbands
    gin_channels:   int = 0,     # Feature dimension size of conditioning series
    **kwargs,                    # (Not used, for backward-compatibility)   # pyright: ignore [reportUnknownParameterType, reportMissingParameterType]
  ):
    super().__init__()

    # For Backward-compatibility
    print(f"Loaded but not used: {kwargs}")
    if kwargs.get("resblock"):
        assert kwargs["resblock"] == '1', "ResBlock2 support is droped."

    # Params
    self.segment_size = segment_size
    unit_channels: int = 256 # 768 # Feature dimension size of unit series

    # PosteriorEncoder / PriorEncoder (ContentEncoder/Flow) / SpeakerEncoder
    self.enc_q =      PosteriorEncoder(spec_channels,  inter_channels, hidden_channels, 5, 16,    gin_channels)
    self.enc_p =      PosteriorEncoder(unit_channels,  inter_channels, hidden_channels, 5, 16,    0)
    self.flow  = ResidualCouplingBlock(inter_channels, inter_channels, hidden_channels, 5,  4, 4, gin_channels)
    self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)

    # Decoder
    Dec = Multiband_iSTFT_Generator if mb_istft_vits else (Multistream_iSTFT_Generator if ms_istft_vits else (iSTFT_Generator if istft_vits else None))
    if Dec is None: raise RuntimeError(f"Not-supported decoder flag: {mb_istft_vits}/{ms_istft_vits}/{istft_vits}")
    print(f"Decoder type: {Dec.__name__}")
    self.dec = Dec(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels)

  def forward(self, unit: Tensor, spec: Tensor, mel: Tensor):
    """
    Args:
      unit - Unit series
      spec - Linear-frequency spectrogram
      mel  - Mel-frequency    spectrogram
    Returns:
      o
      o_mb
      ids_slice
      (
        z
        z_p
        m_p
        logs_p
        m_q
        logs_q
    """

    # Mel-to-Emb
    g = self.enc_spk(mel.transpose(1,2)).unsqueeze(-1)
    # Unit-to-Zsi
    _, m_p, logs_p = self.enc_p(unit)
    # Spec-to-Zsd-to-Zsi
    z, m_q, logs_q = self.enc_q(spec, g=g)
    z_p = self.flow(z, g=g)
    # Zsd-to-Wave
    z_slice, ids_slice = commons.rand_slice_segments(z, self.segment_size)
    o, o_mb = self.dec(z_slice, g=g)

    return o, o_mb, ids_slice, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, unit: Tensor, mel: Tensor) -> Tensor:
    """
    Args:
      unit - Unit series
      mel  - Mel-spectrogram
    Returns:
           :: () - Infered Waveform
    """

    # Speaker embedding
    g = self.enc_spk.embed_utterance(mel.transpose(1,2)).unsqueeze(-1)

    # Enc-Dec
    z_p, _, _ = self.enc_p(unit)
    z = self.flow(z_p, g=g, reverse=True)
    o, _ = self.dec(z, g=g)

    return o
