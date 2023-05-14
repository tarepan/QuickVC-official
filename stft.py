"""Class wrapper of PyTorch STFT/iSTFT."""

import torch
from torch import Tensor
import numpy as np
from scipy.signal import get_window


class TorchSTFT(torch.nn.Module):
    def __init__(self,
                 filter_length: int,
                 hop_length:    int,
                 win_length:    int,
                 window:        str = 'hann'
    ):
        super().__init__()
        self.filter_length, self.hop_length, self.win_length = filter_length, hop_length, win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data: Tensor) -> tuple[Tensor, Tensor]:
        """STFT"""
        forward_transform = torch.stft(input_data, self.filter_length, self.hop_length, self.win_length, window=self.window, return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        """iSTFT :: 2x (B, Freq, Frame) -> (B, 1, T)"""
        inverse_transform = torch.istft(magnitude * torch.exp(phase * 1j), self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data: Tensor) -> Tensor:
        """Reconstruction, x^ = iSTFT(STFT(x))"""
        return self.inverse(*self.transform(input_data))
