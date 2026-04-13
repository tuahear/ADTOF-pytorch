"""
Clean audio processing implementation for ADTOF without madmom dependency.

Audio I/O and STFT use soundfile + scipy + numpy directly instead of librosa.
The librosa stack has lazy module imports and numba JIT that cost ~20s on the
first call per process, which dominates cold-start latency in production
serverless deployments. soundfile and scipy are pre-compiled C/C++ and have
near-zero first-call overhead.

Output of compute_stft() matches `librosa.stft(..., center=True,
pad_mode='constant', window=hanning(n_fft))` to floating-point precision so
existing model weights do not need retraining.
"""

from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 44100,
        fps: int = 100,
        frame_size: int = 2048,
        bands_per_octave: int = 12,
        fmin: float = 20.0,
        fmax: float = 20000.0,
        n_channels: int = 1,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.fps = fps
        self.frame_size = frame_size
        self.bands_per_octave = bands_per_octave
        self.fmin = fmin
        self.fmax = fmax
        self.n_channels = n_channels
        self.normalize = normalize
        self.hop_length = int(np.round(sample_rate / fps))
        self.n_fft = frame_size
        self._setup_filterbank()

    def _setup_filterbank(self) -> None:
        target_frequencies = self._log_frequencies(self.bands_per_octave, self.fmin, self.fmax)
        fft_freqs = np.fft.fftfreq(self.n_fft, 1 / self.sample_rate)[: self.n_fft // 2]
        bins = self._frequencies_to_bins(target_frequencies, fft_freqs, unique_bins=True)
        self.filterbank = self._create_madmom_filterbank(bins, len(fft_freqs)).astype(np.float32, copy=False)
        self.n_bins = self.filterbank.shape[0]

    def _log_frequencies(self, bands_per_octave: int, fmin: float, fmax: float) -> np.ndarray:
        freqs = []
        factor = 2.0 ** (1.0 / bands_per_octave)
        f = fmin
        while f <= fmax * (1.0 + 1e-12):
            freqs.append(f)
            f *= factor
        return np.array(freqs, dtype=float)

    def _frequencies_to_bins(self, frequencies: np.ndarray, fft_freqs: np.ndarray, unique_bins: bool = True) -> np.ndarray:
        bins = np.empty(len(frequencies), dtype=int)
        for i, f in enumerate(frequencies):
            bins[i] = int(np.argmin(np.abs(fft_freqs - f)))
        if unique_bins:
            unique_bins_array = []
            last_bin = -1
            for bin_idx in bins:
                if bin_idx > last_bin:
                    unique_bins_array.append(int(bin_idx))
                    last_bin = int(bin_idx)
            bins = np.array(unique_bins_array, dtype=int)
        return bins

    def _create_madmom_filterbank(self, bins: np.ndarray, n_fft_bins: int) -> np.ndarray:
        n_filters = len(bins) - 2
        filterbank = np.zeros((n_filters, n_fft_bins), dtype=np.float32)
        for i in range(n_filters):
            left_bin = int(bins[i])
            center_bin = int(bins[i + 1])
            right_bin = int(bins[i + 2])
            if right_bin - left_bin < 2:
                if 0 <= left_bin < n_fft_bins:
                    filterbank[i, left_bin] = 1.0
                continue
            if center_bin > left_bin:
                for b in range(left_bin, center_bin):
                    filterbank[i, b] = (b - left_bin) / float(center_bin - left_bin)
            if 0 <= center_bin < n_fft_bins:
                filterbank[i, center_bin] = 1.0
            if right_bin > center_bin + 0:
                for b in range(center_bin + 1, min(right_bin, n_fft_bins)):
                    filterbank[i, b] = (right_bin - b) / float(right_bin - center_bin)
        filter_sums = np.sum(filterbank, axis=1, keepdims=True)
        filter_sums[filter_sums == 0] = 1
        filterbank = filterbank / filter_sums
        return filterbank

    def load_audio(self, audio_path: str) -> np.ndarray:
        audio, native_sr = sf.read(audio_path, dtype='float32', always_2d=False)
        if audio.ndim > 1:
            # soundfile returns shape (n_samples, n_channels) for multichannel
            if self.n_channels == 1:
                audio = audio.mean(axis=1)
            elif self.n_channels == 2:
                audio = audio.T  # (n_channels, n_samples)
        else:
            if self.n_channels == 2:
                audio = np.stack([audio, audio], axis=0)
        if native_sr != self.sample_rate:
            # scipy.signal.resample_poly is pure C, no JIT cost.
            if audio.ndim == 1:
                audio = resample_poly(audio, self.sample_rate, native_sr).astype(np.float32, copy=False)
            else:
                audio = np.stack(
                    [resample_poly(audio[i], self.sample_rate, native_sr) for i in range(audio.shape[0])],
                    axis=0,
                ).astype(np.float32, copy=False)
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32, copy=False)

    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        # Numpy-native STFT matching librosa.stft(center=True, pad_mode='constant',
        # window=hanning(n_fft)). Avoids librosa's numba JIT.
        n_fft = self.n_fft
        hop = self.hop_length
        # librosa uses np.hanning which is the symmetric Hann window.
        window = np.hanning(n_fft).astype(np.float32)
        # center=True: zero-pad reflect of length n_fft//2 on both sides
        padded = np.pad(audio, n_fft // 2, mode='constant')
        n_frames = 1 + (padded.shape[0] - n_fft) // hop
        # as_strided framing: zero-copy view, no JIT.
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=(n_frames, n_fft),
            strides=(padded.strides[0] * hop, padded.strides[0]),
            writeable=False,
        )
        # Apply window and run real FFT. rfft returns n_fft//2 + 1 bins.
        spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
        # Match librosa.stft output shape: (n_freqs, n_frames). Slice [:n_fft//2, :]
        # to drop the Nyquist bin (matches existing downstream slicing).
        magnitude = np.abs(spectrum.T)[: n_fft // 2, :].astype(np.float32, copy=False)
        return magnitude

    def apply_filterbank(self, spectrogram: np.ndarray) -> np.ndarray:
        filtered = (self.filterbank @ spectrogram.astype(np.float32, copy=False)).astype(np.float32, copy=False)
        filtered = np.log10(1.0 + filtered).astype(np.float32, copy=False)
        return filtered

    def process_audio(self, audio_path: str) -> np.ndarray:
        audio = self.load_audio(audio_path)
        if self.n_channels == 1:
            stft = self.compute_stft(audio)
            filtered = self.apply_filterbank(stft)
            result = filtered.T.astype(np.float32, copy=False)
            result = result[:, :, np.newaxis]
        else:
            results = []
            for ch in range(self.n_channels):
                stft = self.compute_stft(audio[ch])
                filtered = self.apply_filterbank(stft)
                results.append(filtered.T)
            result = np.stack(results, axis=2).astype(np.float32, copy=False)
        return result

    def get_n_bins(self) -> int:
        return self.n_bins


def create_adtof_processor(**kwargs) -> AudioProcessor:
    defaults = {
        'sample_rate': 44100,
        'fps': 100,
        'frame_size': 2048,
        'bands_per_octave': 12,
        'fmin': 20.0,
        'fmax': 20000.0,
        'n_channels': 1,
        'normalize': False,
    }
    defaults.update(kwargs)
    return AudioProcessor(**defaults)


def process_audio_file(audio_path: str, **kwargs) -> Tuple[np.ndarray, int]:
    processor = create_adtof_processor(**kwargs)
    spectrogram = processor.process_audio(audio_path)
    return spectrogram, processor.get_n_bins()
