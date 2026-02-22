"""Sample base class — shared audio loading, cropping, panning, and playback."""

from __future__ import annotations
import numpy as np
import soundfile as sf
from scipy import signal
from dawpy.renderable import Renderable


class Sample(Renderable):
    """Abstract base for pre-recorded audio samples.

    Handles all the mechanics shared between FreeSample and BeatLockedSample:
    - Loading and decoding the audio file
    - Resampling to project sample rate
    - Offset and crop with bounds validation
    - Stereo panning
    - Playback via play()

    Subclasses must implement render(bpm, sample_rate) -> (2, num_samples).
    """

    def __init__(
        self,
        filepath: str,
        sample_rate: int = 44100,
        pan: float = 0.0,
        offset_seconds: float = 0.0,
        crop_seconds: float | None = None,
        label: str | None = None,
    ) -> None:
        """Load and prepare an audio file.

        Args:
            filepath:       Path to a .wav, .ogg, .flac, or .mp3 file.
            sample_rate:    Project sample rate. File is resampled to this
                            rate at load time.
            pan:            Stereo pan (-1.0 left … 0.0 centre … 1.0 right).
            offset_seconds: Skip this many seconds from the start of the file.
                            Must be >= 0 and < the file duration.
            crop_seconds:   Keep only this many seconds from the offset point.
                            None = read to the end of the file.
                            offset_seconds + crop_seconds must not exceed the
                            total file duration.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file cannot be read, has more than 2
                               channels, pan is out of range, or the
                               offset/crop window falls outside the file's
                               timeline.
        """
        if not -1.0 <= pan <= 1.0:
            raise ValueError(f"pan must be between -1.0 and 1.0, got {pan}")
        if offset_seconds < 0.0:
            raise ValueError(f"offset_seconds must be >= 0, got {offset_seconds}")
        if crop_seconds is not None and crop_seconds <= 0.0:
            raise ValueError(
                f"crop_seconds must be > 0 when specified, got {crop_seconds}"
            )

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            raw_audio, file_sr = sf.read(filepath, dtype="float32", always_2d=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        except Exception as exc:
            raise ValueError(f"Could not read audio file '{filepath}': {exc}") from exc

        if raw_audio.shape[1] > 2:
            raise ValueError(
                f"Expected mono or stereo audio, got {raw_audio.shape[1]} channels "
                f"in '{filepath}'"
            )

        # ── Normalise to (2, num_samples) ─────────────────────────────────────
        audio = raw_audio.T  # (channels, num_samples)
        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])  # mono → stereo

        # ── Resample to project rate ──────────────────────────────────────────
        if file_sr != sample_rate:
            target_len = int(audio.shape[1] * sample_rate / file_sr)
            audio = np.asarray(signal.resample(audio, target_len, axis=1)).astype(
                np.float32
            )

        # ── Validate and apply offset / crop ──────────────────────────────────
        total_samples = audio.shape[1]
        total_seconds = total_samples / sample_rate

        if offset_seconds >= total_seconds:
            raise ValueError(
                f"offset_seconds ({offset_seconds:.3f}s) is at or beyond the end "
                f"of '{filepath}' ({total_seconds:.3f}s)"
            )

        offset_samples = int(offset_seconds * sample_rate)

        if crop_seconds is not None:
            end_seconds = offset_seconds + crop_seconds
            if end_seconds > total_seconds:
                available = total_seconds - offset_seconds
                raise ValueError(
                    f"Crop window [{offset_seconds:.3f}s … {end_seconds:.3f}s] "
                    f"extends beyond the end of '{filepath}' ({total_seconds:.3f}s). "
                    f"Maximum crop_seconds from this offset: {available:.3f}s."
                )
            end_samples = int(end_seconds * sample_rate)
        else:
            end_samples = total_samples

        audio = audio[:, offset_samples:end_samples]

        # ── Apply pan ─────────────────────────────────────────────────────────
        left_gain = float(np.clip(1.0 - pan, 0.0, 1.0))
        right_gain = float(np.clip(1.0 + pan, 0.0, 1.0))
        audio[0] *= left_gain
        audio[1] *= right_gain

        import os as _os

        # ── Store ─────────────────────────────────────────────────────────────
        self._audio = audio
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.pan = pan
        self.offset_seconds = offset_seconds
        self.crop_seconds = crop_seconds
        self.label = (
            label
            if label is not None
            else _os.path.splitext(_os.path.basename(filepath))[0]
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _resample_audio(self, audio: np.ndarray, target_sample_rate: int) -> np.ndarray:
        """Resample (2, N) audio to a different sample rate."""
        if target_sample_rate == self.sample_rate:
            return audio
        target_len = int(audio.shape[1] * target_sample_rate / self.sample_rate)
        return np.asarray(signal.resample(audio, target_len, axis=1)).astype(np.float32)

    # ── Playback ──────────────────────────────────────────────────────────────

    # ── Subclass contract ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        raise NotImplementedError
