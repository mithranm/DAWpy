"""BeatLockedSample — time-stretched to fill an exact number of beats."""

from __future__ import annotations
import numpy as np
from scipy import signal
from dawpy.sample._base import Sample


class BeatLockedSample(Sample):
    """A sample that is time-stretched to fill exactly N beats at render time.

    The clip is resampled so its duration matches `beats / bpm * 60` seconds,
    regardless of the original file length. A 4-beat loop will always fill
    4 beats whether the project is at 80 or 160 BPM.

    Use for drum loops, groove loops, or any material that must stay locked
    to the project grid.

    Length queries
    --------------
    Because duration depends on BPM, length methods require bpm:

        loop = BeatLockedSample("groove.wav", beats=4)
        print(loop.beats)                        # 4
        print(loop.length_seconds(bpm=128))      # 1.875s  (4 / 128 * 60)
        print(loop.length_samples(bpm=128))      # 82687   (at 44100 Hz)

    Example::

        loop  = BeatLockedSample("drums.wav",  beats=4)
        perc  = BeatLockedSample("perc.flac",  beats=8, pan=0.3)
        intro = BeatLockedSample("intro.wav",  beats=16, offset_seconds=2.0)

        arr = Arrangement().add(loop, repeat=8).add(perc)
        arr.play(bpm=128)
    """

    def __init__(
        self,
        filepath: str,
        beats: float,
        sample_rate: int = 44100,
        pan: float = 0.0,
        offset_seconds: float = 0.0,
        crop_seconds: float | None = None,
    ) -> None:
        """Load a beat-locked audio sample.

        Args:
            filepath:       Path to .wav, .ogg, .flac, or .mp3.
            beats:          Number of beats this clip should fill at render time.
                            Must be > 0.
            sample_rate:    Project sample rate (default 44100).
            pan:            Stereo pan (-1.0 left … 0.0 centre … 1.0 right).
            offset_seconds: Seconds to skip from the start of the file.
            crop_seconds:   Seconds to keep from the offset. None = to end.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        On invalid parameters or out-of-range crop window.
        """
        if beats <= 0:
            raise ValueError(f"beats must be > 0, got {beats}")

        super().__init__(
            filepath=filepath,
            sample_rate=sample_rate,
            pan=pan,
            offset_seconds=offset_seconds,
            crop_seconds=crop_seconds,
        )
        self.beats = beats

    # ── Length (BPM-dependent) ────────────────────────────────────────────────

    def length_seconds(self, bpm: int) -> float:
        """Duration of the time-stretched clip in seconds at the given BPM.

        Args:
            bpm: Tempo in beats per minute.

        Returns:
            Duration in seconds: beats / bpm * 60
        """
        return self.beats / bpm * 60

    def length_samples(self, bpm: int, sample_rate: int | None = None) -> int:
        """Duration of the time-stretched clip in samples at the given BPM.

        Args:
            bpm:         Tempo in beats per minute.
            sample_rate: Output sample rate. Defaults to self.sample_rate.

        Returns:
            Number of samples: int(length_seconds * sample_rate)
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate
        return int(self.length_seconds(bpm) * sr)

    def duration_seconds(self, bpm: int) -> float:
        """Duration in seconds at the given BPM."""
        return self.length_seconds(bpm)

    # ── Renderable ────────────────────────────────────────────────────────────

    def render(self, bpm: int, sample_rate: int) -> np.ndarray:
        """Render to (2, num_samples) float32, time-stretched to fit beats.

        Args:
            bpm:         Tempo in beats per minute.
            sample_rate: Output sample rate in Hz.

        Returns:
            Stereo float32 array, shape (2, target_samples).
        """
        audio = self._resample_audio(self._audio, sample_rate)
        target_samples = self.length_samples(bpm, sample_rate)
        return np.asarray(signal.resample(audio, target_samples, axis=1)).astype(
            np.float32
        )

    def __repr__(self) -> str:
        raw_secs = self._audio.shape[1] / self.sample_rate
        crop = (
            f", crop=[{self.offset_seconds:.2f}s+{self.crop_seconds:.2f}s]"
            if self.crop_seconds
            else ""
        )
        return (
            f"BeatLockedSample('{self.filepath}', "
            f"beats={self.beats}, raw={raw_secs:.2f}s{crop} @ {self.sample_rate}Hz)"
        )
