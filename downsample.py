"""
Cloned from https://github.com/OlaWod/FreeVC/tree/main @2023-05-11, under MIT licence
"""

import os
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.io import wavfile
import librosa
from tqdm import tqdm


def process(wav_name: str):
    target_sr = 16000
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = wav_name[:4]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '_mic2.flac' in wav_path:

        # Load
        wav, source_sr = librosa.load(wav_path)

        # Silent trimming
        wav, _ = librosa.effects.trim(wav, top_db=20)

        # Scaling
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak

        # Resampling
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=target_sr)

        # Save as 16kHz/s16 .wav file
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)
        save_name = wav_name.replace("_mic2.flac", ".wav")
        save_path1 = os.path.join(args.out_dir, speaker, save_name)
        wavfile.write(save_path1, target_sr, (wav * np.iinfo(np.int16).max).astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir",  type=str, default="./dataset/vctk/wav48_silence_trimmed/", help="path to source dir")
    parser.add_argument("--out_dir", type=str, default="./dataset/vctk-16k/vctk-16k",           help="path to target dir")
    args = parser.parse_args()

    pool = Pool(processes=cpu_count()-2)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            for _ in tqdm(pool.imap_unordered(process, os.listdir(spk_dir))):
                pass
