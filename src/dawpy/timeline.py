# src/dawpy/timeline.py

class Time:
    """
    Represents a musical position or duration.
    Assumes standard 4/4 time signature for now (4 beats per bar).
    """
    def __init__(self, bar=1, beat=1.0):
        # DAWs traditionally start counting at Bar 1, Beat 1
        self.bar = bar
        self.beat = beat

    @property
    def total_beats(self) -> float:
        """Converts the Bar/Beat position into a raw beat count from zero."""
        # Bar 1, Beat 1 = 0.0 total beats
        # Bar 2, Beat 1 = 4.0 total beats
        bars_zero_indexed = self.bar - 1
        beats_zero_indexed = self.beat - 1.0
        return (bars_zero_indexed * 4.0) + beats_zero_indexed

    def __repr__(self):
        return f"Time(bar={self.bar}, beat={self.beat})"


class Project:
    """
    The root container for a song. Holds the master BPM and tracks.
    """
    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm
        self.tracks =[]

    def time_to_seconds(self, t: Time) -> float:
        """Converts a Musical Time object into Absolute Time (seconds)."""
        beats_per_second = self.bpm / 60.0
        return t.total_beats / beats_per_second

    def seconds_to_samples(self, seconds: float, sample_rate: int = 44100) -> int:
        """Converts Absolute Time into Discrete Time (array index)."""
        return int(seconds * sample_rate)