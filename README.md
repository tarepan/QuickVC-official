<div align="center">

# QuickVC : HuBERT-VITS-MSiSTFTNet Voice Conversion <!-- omit in toc -->
[![OpenInColab]][notebook]
[![paper_badge]][paper]

</div>

Clone of the official ***QuickVC*** implementation.  
[Official demo](https://quickvc.github.io/quickvc-demo).  

<img src="qvcfinalwhite.png" width="100%">

## [Pretrained Model](https://drive.google.com/drive/folders/1DF6RgIHHkn2aoyyUMt4_hPitKSc2YR9d?usp=share_link)
Put pretrained model into logs/quickvc

## Inference with pretrained model
```python
python convert.py
```
You can change convert.txt to select the target and source

## Preprocess
1. Hubert-Soft
```python
cd dataset
python encode.py soft dataset/VCTK-16K dataset/VCTK-16K
```
2. Spectrogram resize data augumentation, please refer to [FreeVC](https://github.com/OlaWod/FreeVC).

## Train

```python
python train.py
```

If you want to change the config and model name, change:
```python
parser.add_argument('-c', '--config', type=str, default="./configs/quickvc.json",help='JSON file for configuration')
parser.add_argument('-m', '--model', type=str,default="quickvc",help='Model name')
```                   
in utils.py

## Info from official repository
- Naturalness has Language dependency (c.f. SoftVC) [issue#4](https://github.com/quickvc/QuickVC-VoiceConversion/issues/4)
- Training time: 1~2week on RTX3090 x1 [issue#6](https://github.com/quickvc/QuickVC-VoiceConversion/issues/6)

## References
### Original paper <!-- omit in toc -->
[![paper_badge]][paper]  
<!-- Generated with the tool -> https://arxiv2bibtex.org/?q=2302.08296&format=bibtex -->
```bibtex
@misc{2302.08296,
Author = {Houjian Guo and Chaoran Liu and Carlos Toshinori Ishi and Hiroshi Ishiguro},
Title = {QuickVC: Any-to-many Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion},
Year = {2023},
Eprint = {arXiv:2302.08296},
}
```

### Acknowlegements <!-- omit in toc -->
- [MS-ISTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS): Decoder
- [Soft-VC](https://github.com/bshall/hubert): PriorEncoder's Hubert-soft
- [FreeVC](https://github.com/OlaWod/FreeVC): data augumentation

[paper]: https://arxiv.org/abs/2302.08296
[paper_badge]: http://img.shields.io/badge/paper-arxiv.2302.08296-B31B1B.svg
[notebook]: https://colab.research.google.com/github/tarepan/QuickVC-official/blob/main/quickvc.ipynb
[OpenInColab]: https://colab.research.google.com/assets/colab-badge.svg
