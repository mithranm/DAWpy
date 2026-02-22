"""Project — top-level mixer that places Arrangements on a shared timeline.

Each Arrangement added to a Project can be:

  - Placed at any point in the project timeline (``offset``).
  - Read starting from an internal position (``clip_start``), so the
    opening ``clip_start`` seconds of the arrangement are skipped.
  - Truncated early (``clip_end``), so only audio up to ``clip_end``
    seconds into the arrangement is used ("early stop").

Example::

    melody = Arrangement().add(...)
    bass   = Arrangement().add(...)

    project = Project(bpm=128)
    project.add(melody)                    # starts at t=0, plays in full
    project.add(bass, offset=4.0)         # bass enters 4 s into the mix
    project.add(melody, clip_start=2.0)   # skip the first 2 s of melody
    project.add(melody, clip_end=8.0)     # use only the first 8 s

    project.play()
    project.save("song.wav")
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from dawpy.composition import Arrangement
from dawpy.renderable import Renderable


@dataclass
class _Slot:
    """Internal record of one Arrangement placed on the project timeline."""

    arrangement: Arrangement
    offset: float = 0.0  # seconds into project timeline
    clip_start: float = 0.0  # seconds into arrangement to start reading
    clip_end: float | None = None  # seconds into arrangement to stop reading


class Project:
    """Top-level mixer: places Arrangements on a shared timeline.

    Project owns the tempo (bpm) and sample rate for the whole song.
    Each Arrangement is added with an optional timeline offset,
    clip start (trim from the front), and clip end (early stop).
    """

    def __init__(self, bpm: int = 120, sample_rate: int = 44100) -> None:
        """Create a Project.

        Args:
            bpm:         Tempo in beats per minute (default 120).
            sample_rate: Output sample rate in Hz (default 44100).

        Raises:
            ValueError: If bpm or sample_rate is not a positive integer.
        """
        if not isinstance(bpm, int) or bpm <= 0:
            raise ValueError(f"bpm must be a positive integer, got {bpm!r}")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive integer, got {sample_rate!r}"
            )

        self.bpm = bpm
        self.sample_rate = sample_rate
        self._slots: list[_Slot] = []

    # ── Build ────────────────────────────────────────────────────────────────

    def add(
        self,
        arrangement: Arrangement,
        *,
        offset: float = 0.0,
        clip_start: float = 0.0,
        clip_end: float | None = None,
    ) -> "Project":
        """Add an arrangement to the project timeline.

        Args:
            arrangement: The Arrangement to include.
            offset:      When (in seconds) this arrangement begins in the
                         project timeline. Default 0.0.
            clip_start:  Seconds into the arrangement to start reading from.
                         The opening ``clip_start`` seconds are skipped.
                         Default 0.0.
            clip_end:    Seconds into the arrangement to stop reading.
                         ``None`` plays to the natural end (default).
                         Must be greater than ``clip_start`` when set.

        Returns:
            self, for chaining.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset!r}")
        if clip_start < 0:
            raise ValueError(f"clip_start must be >= 0, got {clip_start!r}")
        if clip_end is not None and clip_end <= clip_start:
            raise ValueError(
                f"clip_end ({clip_end!r}) must be greater than "
                f"clip_start ({clip_start!r})"
            )
        self._slots.append(_Slot(arrangement, offset, clip_start, clip_end))
        return self

    # ── Duration ───────────────────────────────────────────────────────────

    def duration_seconds(self) -> float:
        """Total project duration: the latest end time across all arrangements.

        Returns:
            Duration in seconds, or 0.0 if no arrangements have been added.
        """
        if not self._slots:
            return 0.0
        end_times = []
        for slot in self._slots:
            arr_dur = slot.arrangement.duration_seconds(self.bpm)
            clip_end = slot.clip_end if slot.clip_end is not None else arr_dur
            clip_len = max(0.0, min(clip_end, arr_dur) - slot.clip_start)
            end_times.append(slot.offset + clip_len)
        return max(end_times)

    # ── Render ───────────────────────────────────────────────────────────────

    @property
    def rendered(self) -> np.ndarray:
        """Render and mix all arrangements into a (num_samples, 2) float32 array.

        Each arrangement is fully rendered, then clipped (clip_start / clip_end)
        and summed into the mix at its timeline offset.  The final mix is
        peak-normalised if any sample exceeds 1.0.

        Returns:
            Stereo float32 array, shape (num_samples, 2).

        Raises:
            ValueError: If no arrangements have been added, or the project has
                        zero duration after applying all clips.
        """
        if not self._slots:
            raise ValueError(
                "Cannot render an empty project. Add arrangements with add()."
            )
        total_samples = int(self.duration_seconds() * self.sample_rate)
        if total_samples == 0:
            raise ValueError("Project has zero duration after applying all clips.")

        mix = np.zeros((total_samples, 2), dtype=np.float32)

        for slot in self._slots:
            # Full render: (2, N) → transpose to (N, 2) for sample-index slicing
            audio = slot.arrangement.render(self.bpm, self.sample_rate).T.astype(
                np.float32
            )

            # Clip: skip clip_start, stop at clip_end
            s = int(slot.clip_start * self.sample_rate)
            e = (
                int(slot.clip_end * self.sample_rate)
                if slot.clip_end is not None
                else audio.shape[0]
            )
            clip = audio[s:e]
            if clip.shape[0] == 0:
                continue

            # Sum into mix at timeline offset
            off = int(slot.offset * self.sample_rate)
            end = off + clip.shape[0]
            if end > total_samples:
                clip = clip[: total_samples - off]
                end = total_samples
            mix[off:end] += clip

        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak

        return mix

    # ── Playback / export ─────────────────────────────────────────────────

    def play(self) -> None:
        """Render and play the project. Blocks until done; Ctrl+C stops cleanly."""
        Renderable._play_audio(self.rendered.T, self.sample_rate)

    def save(self, path: str) -> None:
        """Render and save the project to an audio file.

        Format is inferred from the file extension (.wav, .flac, .ogg, …).
        Parent directories are created automatically.

        Args:
            path: Output file path (e.g. ``"renders/song.wav"``).
        """
        Renderable._save_audio(self.rendered.T, self.sample_rate, path)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Project(bpm={self.bpm}, sample_rate={self.sample_rate}, "
            f"arrangements={len(self._slots)}, "
            f"duration={self.duration_seconds():.2f}s)"
        )
