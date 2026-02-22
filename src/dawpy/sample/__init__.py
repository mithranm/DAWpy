"""Sample submodule — pre-recorded audio for use in Arrangements.

Two concrete types:

    FreeSample(filepath, ...)
        Plays at original speed. BPM-independent.
        Exposes length_seconds and length_samples properties.

    BeatLockedSample(filepath, beats=N, ...)
        Time-stretched at render time to fill exactly N beats.
        Exposes length_seconds(bpm) and length_samples(bpm) methods.

Both inherit from Sample (shared loading, cropping, panning, play()).
"""

from dawpy.sample._base import Sample
from dawpy.sample.free import FreeSample
from dawpy.sample.beat_locked import BeatLockedSample
from dawpy.sample.inspect import inspect, SampleInfo

__all__ = ["Sample", "FreeSample", "BeatLockedSample", "inspect", "SampleInfo"]
