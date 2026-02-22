"""Silence — a zero-amplitude Renderable for gaps and padding.

Two modes, exactly like the Sample types:

    Silence(seconds=1.5)   Fixed wall-clock duration. BPM-independent.
    Silence(beats=4)       Beat-locked duration: 4 beats at render-time BPM.

Silence is composable: add it to any Arrangement to create a pause between
sounds, pad a section to a bar boundary, or hold space before an entry.

Example::

    kick  = FreeSample("kick.wav")       # 0.35s
    pause = Silence(seconds=0.15)        # pad to exactly 0.5s slots
    loop  = BeatLockedSample("groove.wav", beats=4)
    gap   = Silence(beats=4)             # one-bar gap between loops

    arr = (
        Arrangement()
        .add(kick).add(pause)            # kick in a 0.5s slot
        .add(loop)
        .add(gap)                        # one bar of silence
        .add(loop)
    )
"""

from __future__ import annotations
import numpy as np
from dawpy.renderable import Renderable


class Silence(Renderable):
    """A zero-amplitude audio segment of fixed length.

    Exactly one of ``seconds`` or ``beats`` must be provided.

    ``seconds`` mode
    ----------------
    Duration is fixed in wall-clock time.  BPM has no effect.
    Good for padding one-shots to a time slot, or creating fixed-length gaps.

    ``beats`` mode
    ----------------
    Duration is beat-locked: N beats at the BPM supplied to render().
    Good for bar-boundary padding or rhythmic gaps that follow project tempo.

    Args:
        seconds: Fixed duration in wall-clock seconds. Mutually exclusive
                 with ``beats``.
        beats:   Beat-locked duration. Mutually exclusive with ``seconds``.
        pan:     Accepted for API consistency but has no audible effect on
                 silence (zeros stay zero). Range -1.0–1.0.

    Raises:
        ValueError: If neither or both of ``seconds`` / ``beats`` are given,
                    or if the provided value is <= 0.

    Example::

        # 0.5 s of silence
        gap = Silence(seconds=0.5)
        print(gap.duration_seconds(bpm=128))  # 0.5

        # One-bar gap at 128 BPM (4/4 → 4 beats)
        bar = Silence(beats=4)
        print(bar.duration_seconds(bpm=128))  # 1.875
    """

    def __init__(
        self,
        seconds: float | None = None,
        beats: float | None = None,
        pan: float = 0.0,
        label: str | None = None,
    ) -> None:
        if seconds is None and beats is None:
            raise ValueError(
                "Silence requires either seconds= or beats=. "
                "E.g. Silence(seconds=0.5) or Silence(beats=4)."
            )
        if seconds is not None and beats is not None:
            raise ValueError("Silence accepts either seconds= or beats=, not both.")
        if seconds is not None and seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        if beats is not None and beats <= 0:
            raise ValueError(f"beats must be > 0, got {beats}")
        if not -1.0 <= pan <= 1.0:
            raise ValueError(f"pan must be -1.0–1.0, got {pan}")

        self.seconds = seconds  # None in beats mode
        self.beats = beats  # None in seconds mode
        self.label = label
        self.pan = pan

    # ── Duration ─────────────────────────────────────────────────────────────

    def duration_seconds(self, bpm: int) -> float:
        """Duration in seconds.

        Args:
            bpm: Tempo in beats per minute. Ignored in seconds mode.

        Returns:
            Duration in seconds.
        """
        if self.seconds is not None:
            return self.seconds
        return self.beats / bpm * 60  # type: ignore[operator]

    # ── Renderable ────────────────────────────────────────────────────────────

    def render(self, bpm: int, sample_rate: int) -> np.ndarray:
        """Render to a (2, num_samples) float32 zero array.

        Args:
            bpm:         Tempo in beats per minute. Used only in beats mode.
            sample_rate: Output sample rate in Hz.

        Returns:
            Stereo float32 zero array, shape (2, num_samples).
        """
        num_samples = int(self.duration_seconds(bpm) * sample_rate)
        return np.zeros((2, num_samples), dtype=np.float32)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.seconds is not None:
            return f"Silence(seconds={self.seconds})"
        return f"Silence(beats={self.beats})"
