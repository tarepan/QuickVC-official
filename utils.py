from dataclasses import dataclass
import os
import glob
import sys
import argparse
import logging
import json

import torch


MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging


#### HParams #########################################################################
@dataclass
class TrainParams:
  log_interval:  int
  eval_interval: int
  seed:          int
  epochs:        int
  learning_rate: float
  betas:         tuple[float, float]
  eps:           float
  batch_size:    int
  fp16_run:      bool
  lr_decay:      float
  segment_size:  int
  c_mel:         float
  c_kl:          float
  max_speclen:   int
  fft_sizes:     tuple[int, int, int]
  hop_sizes:     tuple[int, int, int]
  win_lengths:   tuple[int, int, int]
  window:        str

@dataclass
class DataParams:
    training_files:   str
    validation_files: str
    sampling_rate:    int
    filter_length:    int
    hop_length:       int
    win_length:       int
    n_mel_channels:   int
    mel_fmin:         float | None
    mel_fmax:         float | None

@dataclass
class ModelParams:
    ms_istft_vits:      bool
    mb_istft_vits:      bool
    istft_vits:         bool
    subbands:           int
    gen_istft_n_fft:    int
    gen_istft_hop_size: int
    inter_channels:     int
    hidden_channels:    int
    resblock_kernel_sizes:    list[int]
    resblock_dilation_sizes:  list[list[int]]
    upsample_rates:           list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes:    list[int]
    gin_channels:       int

@dataclass
class QuickVCParams:
    model_dir: str
    train: TrainParams
    data:  DataParams
    model: ModelParams


def get_hparams() -> QuickVCParams:
  parser = argparse.ArgumentParser()
  parser.add_argument('-c',  '--config',     type=str, default="./configs/quickvc.json", help='JSON file for configuration')
  parser.add_argument('-m',  '--model',      type=str, default="quickvc",                help='Model name')
  parser.add_argument('-mr', '--modelroot',  type=str, default="./logs",                 help='Path of model root directory')
  
  args = parser.parse_args()
  model_dir: str = os.path.join(args.modelroot, args.model)

  # Load
  config_path = args.config
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)
  hparams = HParams(**config)
  hparams.model_dir = model_dir

  # Save
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "w") as f:
    f.write(data)

  return hparams


def get_hparams_from_file(config_path: str) -> QuickVCParams:
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
#### /HParams ########################################################################


#### Check pointing ##################################################################
def load_checkpoint(checkpoint_path: str, model, optimizer=None):
  """Load 4 states."""
  assert os.path.isfile(checkpoint_path)

  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']

  # Optimizer
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])

  # Model
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)

  logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")

  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate: float, iteration: int, checkpoint_path: str):
  """Save 4 states."""
  logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth"):
  """Query latest checkpoint."""
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x
#### /Check pointing #################################################################


#### Logging #########################################################################
def get_logger(model_dir: str, filename: str = "train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data
#### /Logging ########################################################################