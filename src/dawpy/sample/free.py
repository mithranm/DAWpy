"""FreeSample — plays at original speed, BPM-independent."""

from __future__ import annotations
import numpy as np
from dawpy.sample._base import Sample


class FreeSample(Sample):
    """A sample that always plays at its original recorded speed.

    BPM has no effect on a FreeSample. Duration is fixed in wall-clock time
    regardless of project tempo.

    Use for one-shots, vocals, FX hits — anything where the exact timing
    and pitch of the original recording must be preserved.

    Clip length
    -----------
    Because duration is BPM-independent, clip length can be queried directly:

        kick = FreeSample("kick.wav")
        print(kick.length_seconds)   # e.g. 0.42  (raw clip length)
        print(kick.length_samples)   # e.g. 18522  (at 44100 Hz)

    Aligning to a time slot
    -----------------------
    Passing ``pad_to_seconds`` or ``pad_to_beats`` makes the FreeSample fill
    a declared time slot — the clip plays at original speed and the remainder
    is silence.  If the clip is longer than the pad target, it plays in full
    (no truncation).

        kick = FreeSample("kick.wav", pad_to_seconds=0.5)
        # → always contributes 0.5 s to the arrangement, even if clip is 0.35 s

        fx = FreeSample("riser.ogg", pad_to_beats=8)
        # → fills exactly 8 beats at render-time BPM regardless of clip length

    Example::

        kick   = FreeSample("kick.wav",   pad_to_seconds=0.5)
        riser  = FreeSample("riser.ogg",  offset_seconds=2.0, crop_seconds=4.0)
        vocal  = FreeSample("verse.flac", pan=-0.2, pad_to_beats=16)

        arr = Arrangement().add(kick).add(vocal)
        arr.play(bpm=128)
    """

    def __init__(
        self,
        filepath: str,
        sample_rate: int = 44100,
        pan: float = 0.0,
        offset_seconds: float = 0.0,
        crop_seconds: float | None = None,
        pad_to_seconds: float | None = None,
        pad_to_beats: float | None = None,
    ) -> None:
        """Load a free (BPM-independent) audio sample.

        Args:
            filepath:       Path to .wav, .ogg, .flac, or .mp3.
            sample_rate:    Project sample rate (default 44100).
            pan:            Stereo pan (-1.0 left … 0.0 centre … 1.0 right).
            offset_seconds: Seconds to skip from the start of the file.
            crop_seconds:   Seconds to keep from the offset. None = to end.
            pad_to_seconds: If set, render() pads the output with silence so
                            its total length equals this many seconds.
                            Mutually exclusive with pad_to_beats.
                            No effect if the clip is already longer.
            pad_to_beats:   If set, render() pads the output with silence so
                            its total length equals pad_to_beats / bpm * 60.
                            Mutually exclusive with pad_to_seconds.
                            No effect if the clip is already longer.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        On invalid parameters or out-of-range crop window.
        """
        if pad_to_seconds is not None and pad_to_beats is not None:
            raise ValueError(
                "FreeSample accepts either pad_to_seconds= or pad_to_beats=, not both."
            )
        if pad_to_seconds is not None and pad_to_seconds <= 0:
            raise ValueError(f"pad_to_seconds must be > 0, got {pad_to_seconds}")
        if pad_to_beats is not None and pad_to_beats <= 0:
            raise ValueError(f"pad_to_beats must be > 0, got {pad_to_beats}")

        super().__init__(
            filepath=filepath,
            sample_rate=sample_rate,
            pan=pan,
            offset_seconds=offset_seconds,
            crop_seconds=crop_seconds,
        )
        self.pad_to_seconds = pad_to_seconds
        self.pad_to_beats = pad_to_beats

    # ── Length (BPM-independent) ──────────────────────────────────────────────

    @property
    def length_seconds(self) -> float:
        """Raw clip duration in seconds (after offset/crop, before padding)."""
        return self._audio.shape[1] / self.sample_rate

    @property
    def length_samples(self) -> int:
        """Raw clip duration in samples (after offset/crop, before padding)."""
        return self._audio.shape[1]

    def duration_seconds(self, bpm: int) -> float:
        """Declared duration in seconds — the time slot this sample fills.

        Returns the pad target if one is set and larger than the raw clip length;
        otherwise returns the raw clip length.  BPM is only used when pad_to_beats
        is set.

        Args:
            bpm: Tempo in beats per minute. Used only when pad_to_beats is set.

        Returns:
            Duration in seconds.
        """
        clip_secs = self.length_seconds
        if self.pad_to_seconds is not None:
            return max(clip_secs, self.pad_to_seconds)
        if self.pad_to_beats is not None:
            return max(clip_secs, self.pad_to_beats / bpm * 60)
        return clip_secs

    # ── Renderable ────────────────────────────────────────────────────────────

    def render(self, bpm: int, sample_rate: int) -> np.ndarray:
        """Render to (2, num_samples) float32. BPM is ignored except for padding.

        If pad_to_seconds or pad_to_beats is set and the pad target is longer
        than the clip, the output is zero-padded to fill the target duration.
        If the clip is already longer than the pad target, no truncation occurs.

        Args:
            bpm:         Tempo in beats per minute. Only used when pad_to_beats
                         is set to calculate the target duration.
            sample_rate: Output sample rate in Hz.

        Returns:
            Stereo float32 array, shape (2, num_samples).
        """
        audio = self._resample_audio(self._audio, sample_rate)

        # Calculate pad target in samples (if any)
        target_samples: int | None = None
        if self.pad_to_seconds is not None:
            target_samples = int(self.pad_to_seconds * sample_rate)
        elif self.pad_to_beats is not None:
            target_samples = int(self.pad_to_beats / bpm * 60 * sample_rate)

        if target_samples is not None and audio.shape[1] < target_samples:
            pad_width = target_samples - audio.shape[1]
            audio = np.concatenate(
                [audio, np.zeros((2, pad_width), dtype=np.float32)], axis=1
            )

        return audio.astype(np.float32)

    def __repr__(self) -> str:
        crop = (
            f", crop=[{self.offset_seconds:.2f}s+{self.crop_seconds:.2f}s]"
            if self.crop_seconds
            else ""
        )
        pad = ""
        if self.pad_to_seconds is not None:
            pad = f", pad_to_seconds={self.pad_to_seconds}"
        elif self.pad_to_beats is not None:
            pad = f", pad_to_beats={self.pad_to_beats}"
        return (
            f"FreeSample('{self.filepath}', "
            f"{self.length_seconds:.2f}s{crop}{pad} @ {self.sample_rate}Hz)"
        )
