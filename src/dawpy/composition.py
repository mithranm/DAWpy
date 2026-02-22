"""Arrangement — the score that describes what to play.

An Arrangement is pure composition: it holds a sequence of Renderables
(Tones, AudioSamples, DrumMachines, nested Arrangements) and knows nothing
about BPM or sample rate.  Those are supplied at render or playback time.

Anything that implements render(bpm, sample_rate) -> ndarray can be added.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from dawpy.renderable import Renderable
from dawpy.tone import Tone


@dataclass
class TimelineEntry:
    """Position and duration of a single item within an Arrangement.

    Attributes:
        index:           Zero-based position of the item in the arrangement.
        item:            The Renderable itself.
        start_seconds:   Wall-clock start time in seconds.
        start_beats:     Start time measured in beats.
        start_bar:       Start time measured in bars (1-indexed, based on
                         beats_per_bar passed to timeline()).
        duration_seconds: How long this item lasts in seconds.
        end_seconds:     Wall-clock end time in seconds.
        end_beats:       End time measured in beats.
    """

    index: int
    item: Renderable
    start_seconds: float
    start_beats: float
    start_bar: float
    duration_seconds: float
    end_seconds: float
    end_beats: float

    def __repr__(self) -> str:
        return (
            f"[{self.index}] {type(self.item).__name__} "
            f"@ {self.start_seconds:.3f}s / beat {self.start_beats:.2f} / bar {self.start_bar:.2f} "
            f"→ {self.end_seconds:.3f}s ({self.duration_seconds:.3f}s)"
        )


class Arrangement(Renderable):
    """A reusable score: an ordered sequence of Renderables.

    An Arrangement describes *what* to play without committing to any
    particular tempo or sample rate.  The same Arrangement can be rendered
    at 90 BPM or 160 BPM by passing different values to play() or save(),
    or by adding it to a Project.

    It can also preview itself during composition via play().

    Accepted item types
    -------------------
    - Tone              — a synthesised note (duration in beats)
    - FreeSample        — a pre-recorded clip at original speed (BPM-independent)
    - BeatLockedSample  — a pre-recorded clip stretched to fit N beats
    - Arrangement       — a nested sub-arrangement (composes recursively)
    - Any Renderable    — anything that implements render(bpm, sample_rate)

    Example::

        from dawpy.sample import FreeSample, BeatLockedSample

        kick  = FreeSample("kick.wav")
        loop  = BeatLockedSample("groove.wav", beats=4)
        tone  = Tone(frequency=440.0, duration=1.0, volume=-10.0)

        verse = Arrangement().add([tone]).add(loop)
        song  = Arrangement().add(verse, repeat=2).add(kick)

        song.play(bpm=128)          # preview at 128 BPM
        song.save("song.wav", bpm=128)  # export
    """

    def __init__(self, items=None):
        """Initialize Arrangement, optionally with initial items.

        Args:
            items: Optional seed content — a single Renderable, a list of
                Tones, or another Arrangement.  Equivalent to calling
                add(items) immediately after construction.
        """
        self.items: List[Renderable] = []
        if items is not None:
            self.add(items)

    def add(self, items, repeat: int = 1) -> "Arrangement":
        """Add one or more Renderables to this arrangement.

        Args:
            items:  A single Renderable (Tone, AudioSample, Arrangement, …),
                    or a list of Tones.
            repeat: How many times to add these items (default 1).

        Returns:
            Self, for chaining.
        """
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")

        if isinstance(items, list):
            # Flatten a list of Renderables directly into items
            if not all(isinstance(i, Renderable) for i in items):
                raise TypeError(
                    "List passed to add() must contain only Renderable items "
                    "(Tone, Silence, Sample, Arrangement, …)."
                )
            for _ in range(repeat):
                self.items.extend(items)
        elif isinstance(items, Renderable):
            # Single Renderable — Tone, Arrangement, AudioSample, DrumMachine…
            for _ in range(repeat):
                self.items.append(items)
        else:
            raise TypeError(
                f"Expected a Renderable or List[Tone], got {type(items).__name__}. "
                f"Make sure the item implements render(bpm, sample_rate)."
            )

        return self

    def repeat(self, times: int) -> "Arrangement":
        """Repeat entire arrangement N times. Returns self for chaining."""
        if times < 1:
            raise ValueError(f"times must be >= 1, got {times}")
        original = self.items.copy()
        for _ in range(times - 1):
            self.items.extend(original)
        return self

    def render(self, bpm: int, sample_rate: int) -> np.ndarray:
        """Render the full arrangement to a stereo array.

        Each item's render(bpm, sample_rate) is called in sequence and the
        results are concatenated along the sample axis.

        Args:
            bpm:         Tempo in beats per minute.
            sample_rate: Output sample rate in Hz.

        Returns:
            Stereo float32 array of shape (2, total_samples).
        """
        if not self.items:
            raise ValueError("Cannot render an empty Arrangement.")

        chunks = [item.render(bpm, sample_rate) for item in self.items]
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def duration_seconds(self, bpm: int) -> float:
        """Total duration in seconds at the given BPM."""
        total = 0.0
        for item in self.items:
            total += item.duration_seconds(bpm)
        return total

    def duration_beats(self, bpm: int) -> float:
        """Total duration in beats at the given BPM."""
        return self.duration_seconds(bpm) / 60 * bpm

    def duration_bars(self, bpm: int, beats_per_bar: int = 4) -> float:
        """Total duration in bars at the given BPM."""
        return self.duration_beats(bpm) / beats_per_bar

    def timeline(self, bpm: int, beats_per_bar: int = 4) -> list[TimelineEntry]:
        """Return the start position and duration of every item in the arrangement.

        Because the arrangement is sequential, each item's start time is simply
        the cumulative sum of all previous items' durations.

        Useful for aligning a second arrangement — e.g. start a
        vocal arrangement exactly when the eighth bar begins::

            entries = song.timeline(bpm=128, beats_per_bar=4)
            bar8_offset = entries[8].start_seconds
            project.add(vocal_arr, offset=bar8_offset)

        Args:
            bpm:           Tempo in beats per minute.
            beats_per_bar: Number of beats per bar (default 4 = 4/4 time).

        Returns:
            List of TimelineEntry, one per item, in order.
        """
        entries: list[TimelineEntry] = []
        cursor_seconds = 0.0
        seconds_per_beat = 60.0 / bpm
        for i, item in enumerate(self.items):
            dur = item.duration_seconds(bpm)
            start_beats = cursor_seconds / seconds_per_beat
            end_beats = (cursor_seconds + dur) / seconds_per_beat
            entries.append(
                TimelineEntry(
                    index=i,
                    item=item,
                    start_seconds=cursor_seconds,
                    start_beats=start_beats,
                    start_bar=start_beats / beats_per_bar + 1,
                    duration_seconds=dur,
                    end_seconds=cursor_seconds + dur,
                    end_beats=end_beats,
                )
            )
            cursor_seconds += dur
        return entries

    def start_of(self, index: int, bpm: int, beats_per_bar: int = 4) -> TimelineEntry:
        """Return the TimelineEntry for the item at the given index.

        Convenience shortcut for ``arrangement.timeline(bpm)[index]``.

        Args:
            index:         Zero-based item index.
            bpm:           Tempo in beats per minute.
            beats_per_bar: Beats per bar (default 4).

        Raises:
            IndexError: If index is out of range.

        Returns:
            TimelineEntry with full position and duration information.
        """
        entries = self.timeline(bpm, beats_per_bar)
        if index < 0 or index >= len(entries):
            raise IndexError(
                f"index {index} is out of range for arrangement with {len(entries)} items."
            )
        return entries[index]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return f"Arrangement({len(self.items)} items)"
