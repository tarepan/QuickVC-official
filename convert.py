"""Inference"""

import os
import argparse
import logging
import time

import torch
from torch import Tensor
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import wave_to_mel


if __name__ == "__main__":
    logging.getLogger('numba').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile",  type=str, default="logs/quickvc/config.json", help="path to json config file")
    parser.add_argument("--ptfile",  type=str, default="logs/quickvc/quickvc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="convert.txt",              help="path to txt file")
    parser.add_argument("--outdir",  type=str, default="output/quickvc",           help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    ##                          n_freq from n_fft         ,             frame-scale segment size
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
    net_g.eval()
    total = sum([param.nelement() for param in net_g.parameters()])

    print("Number of parameter: %.2fM" % (total/1e6))
    print("Loading checkpoint...")
    utils.load_checkpoint(args.ptfile, net_g, None)

    print("Loading hubert_soft checkpoint")
    hubert_soft = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
    print("Loaded soft hubert.")

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")

    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line

            # Preprocess
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).unsqueeze(0).cuda()

            print(wav_src.size())

            # Infer
            ## TgtWave-to-Mel for speaker embedding
            mel_tgt = wave_to_mel(
                wav_tgt,
                hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            ## SrcWave-to-Unit for content encoding
            unit: Tensor = hubert_soft.units(wav_src).transpose(2,1)
            ## SrcUnit/TgtMel-to-Wave
            audio = net_g.infer(unit, mel_tgt)

            # Save
            audio = audio[0][0].data.cpu().float().numpy()
            filename = f"{time.strftime('%m-%d_%H-%M', time.localtime())}_{title}.wav" if args.use_timestamp else f"{title}.wav"
            write(os.path.join(args.outdir, filename), hps.data.sampling_rate, audio)
