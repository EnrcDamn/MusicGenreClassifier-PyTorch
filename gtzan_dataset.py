import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

class GTZANDataset(Dataset):

    def __init__(self,
                annotations_file,
                audio_dir,
                transformation,
                target_sample_rate,
                num_samples,
                device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
    
    def __len__(self):
        # len(dataset)
        return len(self.annotations)
    
    def __getitem__(self, index):
        # mylist[0] -> mylist.__getitem__(0)
        # get signal and fold index for an audio file
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # apply processing and transformation to the signal
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # signal = self._spectrogram(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _mix_down_if_necessary(self, signal):
        # signal -> tensor (num_channels, samples) -> (2, 32000) -> (1, 32000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sample_rate
            ).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _cut_if_necessary(self, signal):
        # signal -> tensor (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]   # slicing
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) # (append_left, append_right)
            #  [1, 1, 1] (0, 2) -> [1, 1, 1, 0, 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"{self.annotations.iloc[index, -2]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 1])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, -1]

    def _spectrogram(self, signal):
        stft = torch.stft(signal, n_fft=1024, hop_length=512)
        out_abs = torch.sqrt(stft**2)
        spec = 2 * torch.log(torch.clamp(out_abs, 1e-10, float("inf")))
        return spec

if __name__ == "__main__":
    ANNOTATIONS_FILE = "Data/features_30_sec_final.csv"
    AUDIO_DIR = "Data/genres_original"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 # -> 1 second of audio
    plot = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=20,
        log_mels=True
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # objects inside transforms module are callable!
    # ms = mel_spectrogram(signal)

    gtzan = GTZANDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    print(f"There are {len(gtzan)} samples in the dataset")

    if plot:
        signal, label = gtzan[0]
        signal = signal.cpu()
        print(signal.shape)
        
        plt.figure(figsize=(16, 8), facecolor="white")
        plt.imshow(signal[0,:,:], origin='lower')
        plt.autoscale(False)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.axis('auto')
        plt.show()
