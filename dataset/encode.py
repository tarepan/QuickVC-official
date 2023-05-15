"""Preprocessing"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm


def encode_dataset(args):
    """Preprocess"""
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", f"hubert_soft").cuda().eval()

    print(f"Encoding dataset at {args.in_dir}")
    # All audio files under the `in_dir`
    if not args.suffix_16k:
        # default
        paths = list(args.in_dir.rglob(f"*{args.extension}"))
    else:
        paths = Path(args.in_dir).glob("**/*.16k.wav")
    for in_path in tqdm(paths):
        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        # Wave-to-Unit
        with torch.inference_mode():
            units = hubert.units(wav)

        # Save - saved in same relative path with .npy suffix
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument("model",                      help="available models (HuBERT-Soft or HuBERT-Discrete)", choices=["soft", "soft"])
    parser.add_argument("in_dir",  metavar="in-dir",  help="path to the dataset directory.",                 type=Path)
    parser.add_argument("out_dir", metavar="out-dir", help="path to the output directory.",                  type=Path)
    parser.add_argument("--extension",                help="extension of the audio files.",   default=".wav", type=str)
    parser.add_argument("--suffix_16k",               help="Convert only '.16k.wav' files.",  action="store_true")
    args = parser.parse_args()
    encode_dataset(args)
