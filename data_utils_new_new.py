"""Dataset"""

import os
import random

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from scipy.io.wavfile import read

import commons
from mel_processing import spectrogram_torch


def load_filepaths(filename: str) -> list[list[str]]:
  """
  Args:
    filename - The file containing file lists

  single line example: './dataset/vctk-16k/vctk-16k/p262/p262_350.wav'
  """
  split = "|"
  with open(filename, encoding='utf-8') as f:
    filepaths = [line.strip().split(split) for line in f]
  return filepaths


def load_wav_to_torch(full_path: str):
  # SciPy read
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """Dataset, Multi speaker version.

        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.

        Required (preprocessed) files:
            audio - 16kHz/s16 audio file
            c     - Unit series

    """
    def __init__(self, audiopaths, hparams):
        self.max_wav_value: float = hparams.data.max_wav_value  # Maximum amplitude in the audio format
        self.sampling_rate: int   = hparams.data.sampling_rate  # For validation
        self.filter_length: int   = hparams.data.filter_length  # Spectrogram n_fft
        self.win_length:    int   = hparams.data.win_length     # Spectrogram window length
        self.hop_length:    int   = hparams.data.hop_length     # Frame hop length

        self.audiopaths = load_filepaths(audiopaths)

        random.seed(1243)
        random.shuffle(self.audiopaths)
        self._calculate_frame_lengths()

        # Remnants
        self.use_sr, self.use_spk, self.spec_len = False, False, False

    def _calculate_frame_lengths(self):
        """Store spec lengths for Bucketing."""
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        lengths: list[int] = []
        for audiopath in self.audiopaths:
            lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
        self.lengths = lengths

    def get_audio(self, filename: str):
        """
        Args:
            filename               - File path
        Returns:
            c          :: FloatTensor[] - Unit series,            from    f'{filename.parent.name}/{filename.stem}.npy'
            spec       :: FloatTensor[] - Spectrogram             from/to f'{filename.parent.name}/{filename.stem}.spec.pt'
            audio_norm :: FloatTensor[] - Audio in range [-1, 1], from    f'{filename.parent.name}/{filename.stem}.wav'
        """
        # audio_norm :: () - Audio in range [-1, 1]
        audio, sampling_rate = load_wav_to_torch(filename)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        assert sampling_rate == self.sampling_rate, f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"

        # spec :: () - Linear spectrogram
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        # c :: () - Unit series
        c = np.load(filename.replace(".wav", ".npy"))
        c = torch.FloatTensor(c.transpose(1,0))
        
        return c, spec, audio_norm

    def __getitem__(self, index: int):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)


class TextAudioSpeakerCollate():
    """Collate function, Zero-pads model inputs and targets
    """
    def __init__(self, hps):
        self.hps = hps

        # Remnants
        self.use_spk, self.use_sr = False, False

    def __call__(self, batch: tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
        """Collate's training batch from normalized text, audio and speaker identities
        
        Args:
            batch - (c, spec, audio_norm)
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len  = max([x[2].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths  = torch.LongTensor(len(batch))

        # Padding based on longest series
        c_padded    = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded  = torch.FloatTensor(len(batch), 1,                   max_wav_len)
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        spec_seglen = spec_lengths[-1] if spec_lengths[-1] < self.hps.train.max_speclen + 1 else self.hps.train.max_speclen + 1
        wav_seglen = spec_seglen * self.hps.data.hop_length 
        spec_padded, ids_slice = commons.rand_spec_segments(spec_padded, spec_lengths, spec_seglen)
        wav_padded = commons.slice_segments(wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)

        c_padded = commons.slice_segments(c_padded, ids_slice, spec_seglen)[:,:,:-1]

        spec_padded = spec_padded[:,:,:-1]
        wav_padded = wav_padded[:,:,:-self.hps.data.hop_length]

        return c_padded, spec_padded, wav_padded


class DistributedBucketSampler(DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self,
        dataset:      TextAudioSpeakerLoader, # Dataset
        batch_size:   int,                    # Batch size
        boundaries:   list[int],              # Bucket boundaries [frame], e.g. [32,40,50,60,...]
        shuffle:      bool       = True,      # Whether to shuffle samples
    ):
        super().__init__(dataset, shuffle=shuffle)

        self.lengths = dataset.lengths
        self.batch_size, self.boundaries = batch_size, boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.num_samples = sum(self.num_samples_per_bucket)

    def _create_buckets(self) -> tuple[list[list[int]], list[int]]:
        """Create 'buicket's, which include only samples within a range."""

        buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) - 1)]

        # Assign all samples to corresponding bucket
        for i, length in enumerate(self.lengths):
            # Bisect search
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                # Find the bucket, assign sample index
                buckets[idx_bucket].append(i)
            else:
                # Out of all buckets, removed
                pass

        # Filter null buckets and corresponding boundaries
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)

        num_samples_per_bucket: list[int] = []
        for i, bucket in enumerate(buckets):
            len_bucket = len(bucket)
            # DDP batch size
            rem = (self.batch_size - (len_bucket % self.batch_size)) % self.batch_size
            num_samples_per_bucket.append(len_bucket + rem)

        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # Shuffled sample indices in a bucket
        indices: list[list[int]] = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i, bucket in enumerate(self.buckets):
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                # samples from bucket#i
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j+1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]

        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples

        return iter(self.batches)

    def _bisect(self, x: int, lo: int = 0, hi: int | None = None) -> int:
        """Bisect search of corresponding bucket's index."""
        if hi is None:
            # Uppermost boundary index
            hi = len(self.boundaries) - 1

        if hi > lo:
            # Middle boundary index
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                # Matched, return index
                return mid
            elif x <= self.boundaries[mid]:
                # Bisect search in lower half
                return self._bisect(x, lo, mid)
            else:
                # Bisect search in higher half
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        """The number of batches from this instance."""
        return self.num_samples // self.batch_size


if __name__ == "__main__":
    """Test"""
    import utils
    from torch.utils.data import DataLoader

    hps = utils.get_hparams()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size,
        [32,70,100,200,300,400,500,600,700,800,900,1000], shuffle=True)
    collate_fn = TextAudioSpeakerCollate(hps)
    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)

    for batch_idx, (c, spec, y) in enumerate(train_loader):
        print(c.size(), spec.size(), y.size())
        #print(batch_idx, c, spec, y)
