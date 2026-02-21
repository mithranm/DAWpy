import numpy as np
from dawpy.timeline import Timeline


class Project:
    """Manages multiple timelines (tracks) and renders them into a multitrack song.

    A Project is a high-level container that orchestrates multiple Timeline objects,
    each representing a separate instrument or voice. Timelines can play simultaneously,
    and their audio is mixed together into a single output.
    """

    def __init__(self, bpm: int = 120, samplerate: int = 44100):
        """Initialize a new project.

        Args:
            bpm: Tempo in beats per minute. Default is 120 (standard tempo).
                Must be positive.
            samplerate: Audio sample rate in Hz. Default is 44100 (CD quality).
                Common values: 44100, 48000.
                Must be positive.

        Raises:
            ValueError: If bpm or samplerate is not positive.
        """
        if not isinstance(bpm, int) or bpm <= 0:
            raise ValueError(f"bpm must be a positive integer, got {bpm}")
        if not isinstance(samplerate, int) or samplerate <= 0:
            raise ValueError(f"samplerate must be a positive integer, got {samplerate}")

        self.bpm = bpm
        self.samplerate = samplerate
        self.timelines = []

    def add_timeline(self) -> Timeline:
        """Create and add a new timeline (track) to the project.

        Returns:
            The newly created Timeline object.
        """
        timeline = Timeline(bpm=self.bpm, samplerate=self.samplerate)
        self.timelines.append(timeline)
        return timeline

    def get_duration_seconds(self) -> float:
        """Get the total duration of the project.

        This is the maximum duration across all timelines (how long the song plays).

        Returns:
            Total duration in seconds. Returns 0.0 if no timelines exist.
        """
        if not self.timelines:
            return 0.0
        return max(t.get_duration_seconds() for t in self.timelines)

    @property
    def rendered(self) -> np.ndarray:
        """Render all timelines and mix them into a single audio waveform.

        Returns:
            Audio data as a numpy float32 array. Shape is (num_samples,).

        Raises:
            ValueError: If no timelines have been added.
        """
        if not self.timelines:
            raise ValueError(
                "Cannot render empty project. Add timelines with add_timeline()."
            )

        # Render each timeline
        rendered_tracks = []
        for timeline in self.timelines:
            try:
                rendered_tracks.append(timeline.rendered)
            except ValueError:
                # Timeline is empty, skip it
                rendered_tracks.append(np.zeros(0, dtype=np.float32))

        # Pad all tracks to the same length and mix them
        max_len = max(len(t) for t in rendered_tracks) if rendered_tracks else 0

        if max_len == 0:
            raise ValueError(
                "Cannot render empty project. Add timelines with add_timeline()."
            )

        mix = np.zeros(max_len, dtype=np.float32)
        for tarr in rendered_tracks:
            if len(tarr) == 0:
                continue
            if len(tarr) < max_len:
                # Pad with zeros at the end
                pad = np.zeros(max_len - len(tarr), dtype=np.float32)
                tarr = np.concatenate([tarr, pad])
            mix += tarr

        # Prevent clipping by normalizing if needed
        peak = np.max(np.abs(mix)) if mix.size > 0 else 0.0
        if peak > 1.0:
            mix = mix / peak

        return mix.astype(np.float32)

    def __repr__(self) -> str:
        """Return a string representation of the project."""
        return (
            f"Project(bpm={self.bpm}, samplerate={self.samplerate}, "
            f"num_timelines={len(self.timelines)}, duration={self.get_duration_seconds():.2f}s)"
        )
