"""
Cloned from https://github.com/OlaWod/FreeVC/tree/main @2023-05-11, under MIT licence
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import librosa


def process(in_dir: Path):
    target_sr = 16000
    for p in in_dir.glob("**/*.wav"):
        # Load
        wav, source_sr = librosa.load(p, mono=True)

        # Silent trimming
        wav, _ = librosa.effects.trim(wav, top_db=20)

        # Scaling
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak

        # Resampling
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=target_sr)

        # Save as 16kHz/s16 .wav file
        wavfile.write(str(p.with_suffix(".16k.wav")), target_sr, (wav * np.iinfo(np.int16).max).astype(np.int16))


if __name__ == "__main__":
    # Before run this downsampling, you should get contents of corpus
    # e.g.
    # from speechcorpusy import load_preset
    # load_preset("JSUT", "/content/gdrive/MyDrive/ML_data", download=False).get_contents()

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",  type=str, default="./tmp", help="path to source dir")
    args = parser.parse_args()

    process(args.in_dir)
