import numpy as np


class Tone:
    """A musical note with volume controls and envelope shaping.

    Think of a Tone as a single note you play on an instrument. Just like a piano
    key creates a sound that starts, sustains, and fades out, a Tone has:
    - A frequency (pitch, in Hz, like 440 Hz for A note)
    - A duration (how long the note plays, in beats)
    - Envelope controls (how the volume changes over time: fade in, hold, fade out)
    - A filter (to brighten or darken the sound)
    """

    def __init__(
        self,
        frequency: float = 261.63,
        duration: float = 1.0,
        volume: float = -12.0,
        attack: float = 0.01,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.5,
        cutoff: float = 5000.0,
        resonance: float = 0.0,
    ) -> None:
        """Create a musical note.

        Args:
            frequency: Pitch in Hz. Middle C is 261.63 Hz.
                Range: 20–20000 Hz (human hearing range).

            duration: How long the note plays, in beats.
                At 120 BPM, 1 beat = 0.5 seconds.
                Can be any positive number: 0.5, 1, 1.5, 2, etc.

            volume: Loudness in dBFS (decibels).
                Must be ≤ 0. Higher (closer to 0) = louder.
                -12 dB is medium volume. -40 dB is quiet.

            attack: Fade-in time in seconds (0 to full volume).
                0 = instant sound. 0.3 = slow fade in like a violin bow.

            decay: Time in seconds to drop from full volume to sustain level.
                Happens right after attack. Short decay = snappy sound.

            sustain: Percentage of full volume to hold during the note.
                0.0 = silent after attack+decay. 1.0 = stay at full volume.
                Range: 0.0–1.0 (as a ratio, not percent).

            release: Fade-out time in seconds (after note ends).
                How long the tail of the sound lasts after the note stops.

            cutoff: Filter brightness in Hz.
                Low values (e.g., 2000 Hz) = muffled/dark sound.
                High values (e.g., 10000 Hz) = bright/clear sound.
                Range: 20–20000 Hz.

            resonance: Filter emphasis (how much the cutoff frequency "pops").
                0 = no emphasis (flat). Higher values = boosts sound at cutoff.
                Range: 0.0+ (typically 0–10).

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        self._validate_all(
            frequency,
            duration,
            volume,
            attack,
            decay,
            sustain,
            release,
            cutoff,
            resonance,
        )

        self.frequency = frequency
        self.duration = duration
        self.volume = volume
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.cutoff = cutoff
        self.resonance = resonance

    @staticmethod
    def _validate_all(
        frequency, duration, volume, attack, decay, sustain, release, cutoff, resonance
    ):
        """Check that all values are valid."""
        errors = []

        if not isinstance(frequency, (int, float)) or not (
            20.0 <= frequency <= 20000.0
        ):
            errors.append(f"frequency must be 20–20000 Hz, got {frequency}")

        if not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"duration must be > 0 beats, got {duration}")

        if not isinstance(volume, (int, float)) or volume > 0.0:
            errors.append(f"volume must be ≤ 0 dBFS, got {volume}")

        if not isinstance(attack, (int, float)) or attack < 0.0:
            errors.append(f"attack must be ≥ 0 seconds, got {attack}")

        if not isinstance(decay, (int, float)) or decay < 0.0:
            errors.append(f"decay must be ≥ 0 seconds, got {decay}")

        if not isinstance(sustain, (int, float)) or not (0.0 <= sustain <= 1.0):
            errors.append(f"sustain must be 0.0–1.0, got {sustain}")

        if not isinstance(release, (int, float)) or release < 0.0:
            errors.append(f"release must be ≥ 0 seconds, got {release}")

        if not isinstance(cutoff, (int, float)) or not (20.0 <= cutoff <= 20000.0):
            errors.append(f"cutoff must be 20–20000 Hz, got {cutoff}")

        if not isinstance(resonance, (int, float)) or resonance < 0.0:
            errors.append(f"resonance must be ≥ 0.0, got {resonance}")

        if errors:
            raise ValueError("\n".join(errors))

    def get_total_duration_seconds(self, bpm: int = 120) -> float:
        """Get the total time this note takes, from start to silence.

        This includes: fade-in + volume drop + sustain + fade-out.

        Args:
            bpm: Tempo in beats per minute (e.g., 120 = standard tempo).

        Returns:
            Total duration in seconds.
        """
        beat_duration = 60.0 / bpm
        sustain_duration = self.duration * beat_duration
        return self.attack + self.decay + sustain_duration + self.release

    def __repr__(self) -> str:
        return (
            f"Tone(frequency={self.frequency}, duration={self.duration}, "
            f"volume={self.volume}, attack={self.attack}, decay={self.decay}, "
            f"sustain={self.sustain}, release={self.release}, "
            f"cutoff={self.cutoff}, resonance={self.resonance})"
        )

    def render(self, bpm: int = 120, fs: int = 44100) -> np.ndarray:
        """Generate the audio waveform for this note.

        Args:
            bpm: Tempo in beats per minute.
            fs: Sample rate (samples per second). 44100 is standard CD quality.

        Returns:
            Audio data as a numpy array.
        """
        from scipy.signal import butter, sosfilt

        # Convert beat duration to seconds
        beat_duration = 60.0 / bpm
        note_duration_sec = self.duration * beat_duration

        # Total audio length: note + release tail
        total_duration_sec = note_duration_sec + self.release
        num_samples = int(fs * total_duration_sec)
        t = np.linspace(0, total_duration_sec, num_samples, endpoint=False)

        # Generate the sine wave at the note's frequency
        audio = np.sin(2 * np.pi * self.frequency * t)

        # Build the envelope (volume shape over time)
        a_samples = int(self.attack * fs)
        d_samples = int(self.decay * fs)
        r_samples = int(self.release * fs)

        # Calculate sustain samples from the total to avoid rounding errors
        s_samples = num_samples - (a_samples + d_samples + r_samples)

        if s_samples < 0:
            s_samples = 0

        envelope = np.concatenate(
            [
                np.linspace(0, 1, a_samples),  # Attack: fade in
                np.linspace(1, self.sustain, d_samples),  # Decay: drop to sustain level
                np.full(s_samples, self.sustain),  # Sustain: hold steady
                np.linspace(self.sustain, 0, r_samples),  # Release: fade out
            ]
        )

        # Apply low-pass filter (to darken/brighten the sound)
        nyquist = fs / 2
        normalized_cutoff = np.clip(self.cutoff / nyquist, 0.001, 0.999)
        order = max(2, min(8, int(2 + self.resonance * 1.5)))
        sos = butter(order, normalized_cutoff, btype="low", output="sos")
        audio = np.asarray(sosfilt(sos, audio))

        # Apply volume and envelope
        gain = 10 ** (self.volume / 20)
        return (audio * envelope * gain).astype(np.float32)
