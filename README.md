<div align="center">

# QuickVC : HuBERT-VITS-MSiSTFTNet Voice Conversion <!-- omit in toc -->
[![OpenInColab]][notebook]
[![paper_badge]][paper]

</div>

Clone of the official ***QuickVC*** implementation.  
[Official demo](https://quickvc.github.io/quickvc-demo).  

<img src="qvcfinalwhite.png" width="100%">

## Demo
| Source | Target | ***QuickVC-nosr*** | ***QuickVC-sr*** | Diff-VCTK | BNE-PPG-VC | VQMIVC |
--- | --- | --- | --- | --- | --- | ---
| [4179-25937-0002](https://drive.google.com/file/d/1v_q89V0gxjelJdpE9g0IBgADsa9xGuQ1/view?usp=sharing) | [p226_006](https://drive.google.com/file/d/11bSrhDxNmMPDi6yynQfkpiNbXHPnH70E/view?usp=sharing) | [QVCnosr4179toP226](https://drive.google.com/file/d/1CLpaEjBpY-00vDmu51fqkS0tVA61DzKw/view?usp=share_link) | [QVCsr4179toP226](https://drive.google.com/file/d/1cQwSSU5kT3l3Nuz-787-IMwFqXsPAnh7/view?usp=sharing)| [Diff4179toP226](https://drive.google.com/file/d/17htvoXVOD0wBOJqQvGsDJFotdlfZNvPP/view?usp=share_link)| [PPG4179toP226](https://drive.google.com/file/d/1lKD2ljIcTVGeQIpQ5qvsyGZMGhdxDmmo/view?usp=share_link) | [VQMIVC4179toP226](https://drive.google.com/file/d/1pc3rzlPRZ5dPqoPfkzElyI_L2Wot55M0/view?usp=share_link)|
| [4179-25937-0002](https://drive.google.com/file/d/1v_q89V0gxjelJdpE9g0IBgADsa9xGuQ1/view?usp=sharing) | [p312_019](https://drive.google.com/file/d/1iCiYay2J8gU2J4OroDJO5XbkowjCAKpB/view?usp=sharing) | [QVCnosr4179toP312](https://drive.google.com/file/d/1FqrrOZ19z-ZczbpkTBPICqbTbNIqy5K-/view?usp=share_link) | [QVCsr4179toP312](https://drive.google.com/file/d/1u97udZNzA_ciMLzBSKkCW2B03ljsONI1/view?usp=sharing) | [Diff4179toP312](https://drive.google.com/file/d/1JH300rj4QIAEO0uXVCJs7dHGAd-WzTJA/view?usp=share_link)| [PPG4179toP312](https://drive.google.com/file/d/1nFSSs2hVaSmPrMIUlKJjQJ23p9jtyI95/view?usp=share_link) | [VQMIVC4179toP312](https://drive.google.com/file/d/1ykyUdSv-_QhE2Ahy2k4GcgBqSuLHuHcf/view?usp=share_link)|
| [LJ032-0032](https://drive.google.com/file/d/1Flf3hToXHLWwcxZaq9qI6T6bR6voPDkS/view?usp=sharing) | [p246_011](https://drive.google.com/file/d/1Py7n9P52IGJhLFnmEQ32jIbabny5W7mx/view?usp=sharing) | [QVCnosrLJtoP246](https://drive.google.com/file/d/1k8vdnotJXgi92Q2pmJSTDHLtBJNYiZ3d/view?usp=share_link)| [QVCsrLJtoP246](https://drive.google.com/file/d/1qROuQUeAd0Flumduzr-aCgz9l8SwqsIl/view?usp=sharing)| [DiffLJtoP246](https://drive.google.com/file/d/16Kwyo0jFnNIhO8gsXnY0PenQAS2SM8Kp/view?usp=share_link)| [PPGLJtoP246](https://drive.google.com/file/d/1c5qs8sm2hherKEMV6Rit6UIukkZTaW_j/view?usp=share_link) | [VQMIVCLJtoP246](https://drive.google.com/file/d/1LPGjS8xcCCTXmOVLIEANUX5bsv4T5O7w/view?usp=share_link)|
| [LJ032-0032](https://drive.google.com/file/d/1Flf3hToXHLWwcxZaq9qI6T6bR6voPDkS/view?usp=sharing) | [p229_006](https://drive.google.com/file/d/1cYFT_oy29N5STyZNt5ADRQvaRN5Rphea/view?usp=sharing) | [QVCnosrLJtoP229](https://drive.google.com/file/d/1HrYIBPxpGr_8S6J4CpQUMbSIZw0tKvzd/view?usp=share_link) |  [QVCsrLJtoP229](https://drive.google.com/file/d/1Sf-DrtW-PTv_V7HuC-niMTTI-I0qJi1G/view?usp=sharing) | [DiffLJtoP229](https://drive.google.com/file/d/1mRjG2XoC9FgqwFRSYwpPxb_DMULQkfwS/view?usp=share_link)| [PPGLJtoP229](https://drive.google.com/file/d/1v6HrSMODsOpIDHORixO6GFTdvKdMplnw/view?usp=share_link) | [VQMIVCLJtoP229](https://drive.google.com/file/d/1FTIHZQkBs-AdTnacqsQ7u2O1Bgm4xsN5/view?usp=share_link)|


## Inference with pretrained model
Sorry, but I will open source the code after the paper is accepted.
## References
If you have any question about the decoder, refer to [MS-ISTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS).

If you have any question about the Hubert-soft, refer to [Soft-VC](https://github.com/bshall/hubert).

If you have any question about the data augumentation, refer to [FreeVC](https://github.com/OlaWod/FreeVC).

[paper]: https://arxiv.org/abs/2302.08296
[paper_badge]: http://img.shields.io/badge/paper-arxiv.2302.08296-B31B1B.svg
[notebook]: https://colab.research.google.com/github/tarepan/QuickVC-official/blob/main/quickvc.ipynb
[OpenInColab]: https://colab.research.google.com/assets/colab-badge.svg
