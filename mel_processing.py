import torch
from torch import Tensor
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


def spectral_normalize_torch(magnitudes: Tensor) -> Tensor:
    return torch.log(torch.clamp(magnitudes, min=1e-5))


mel_basis = {}
hann_window = {}


def wave_to_spec(
        y: Tensor,            # :: (B, T)           - Audio waveforms
        n_fft: int,
        hop_size: int,
        win_size: int,
        center: bool = False,
    ) -> Tensor:              # :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram."""

    # Validation
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    # Window - Cache if needed
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    """
                          hop=6
    pad=3                 ''''''
    ...-----------------------------
    |__________|     |__________|
       ^^^^^^           nfft=12
       1frame
        -> centered
    """
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # Complex Spectrogram :: (B, T) -> (B, Freq, Frame, RealComplex=2)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    return spec


def spec_to_mel(spec: Tensor, n_fft: int, num_mels: int, sampling_rate: int, fmin, fmax) -> Tensor:
    """
    Args:
        spec :: (B, Feat, Frame) - Linear-frequency spectrograms
    """
    # MelBasis - Cache if needed
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

   # Mel-frequency Log-amplitude spectrogram :: (B, Freq=num_mels, Frame)
    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)

    return melspec


def wave_to_mel(
        y:             Tensor,       # :: (B, T)           - Waveforms
        n_fft:         int,
        num_mels:      int,
        sampling_rate: int,
        hop_size:      int,
        win_size:      int,
        fmin              ,
        fmax              ,
        center:        bool = False,
    ) -> Tensor:                     # :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
    """Convert waveform into Mel-frequency Log-amplitude spectrogram."""

    # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
    spec = wave_to_spec(y, n_fft, hop_size, win_size, center)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
    melspec = spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

    return melspec
