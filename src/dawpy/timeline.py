import numpy as np
from dawpy.tone import Tone


class Timeline:
    """Sequences musical tones and renders them to audio.

    A Timeline holds a list of Tone objects and renders them sequentially
    into a single audio waveform. Think of it as a track in a DAW where you
    place notes one after another.
    """

    def __init__(self, bpm: int = 120, samplerate: int = 44100):
        """Initialize a new timeline.

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
        self.sequence = []  # List to hold Tone objects

    def add_tone(self, tone: Tone) -> None:
        """Add a tone to the end of the timeline.

        The tone will be placed sequentially after all existing tones.

        Args:
            tone: A Tone object to add to the sequence.

        Raises:
            TypeError: If tone is not a Tone object.
        """
        if not isinstance(tone, Tone):
            raise TypeError(f"tone must be a Tone object, got {type(tone).__name__}")
        self.sequence.append(tone)

    def clear(self) -> None:
        """Remove all tones from the timeline."""
        self.sequence = []

    def get_duration_seconds(self) -> float:
        """Get the total duration of all tones in the timeline.

        Returns:
            Total duration in seconds. Returns 0.0 if timeline is empty.
        """
        if not self.sequence:
            return 0.0

        total = 0.0
        for tone in self.sequence:
            total += tone.get_total_duration_seconds(self.bpm)
        return total

    @property
    def rendered(self) -> np.ndarray:
        """Render all tones in the sequence into one audio waveform.

        Returns:
            Audio data as a numpy float32 array. Shape is (num_samples,).

        Raises:
            ValueError: If the timeline is empty (no tones added).
        """
        if not self.sequence:
            raise ValueError("Cannot render empty timeline. Add tones with add_tone().")

        rendered_chunks = [
            tone.render(self.bpm, self.samplerate) for tone in self.sequence
        ]
        return np.concatenate(rendered_chunks)

    def __repr__(self) -> str:
        """Return a string representation of the timeline."""
        return (
            f"Timeline(bpm={self.bpm}, samplerate={self.samplerate}, "
            f"num_tones={len(self.sequence)}, duration={self.get_duration_seconds():.2f}s)"
        )
