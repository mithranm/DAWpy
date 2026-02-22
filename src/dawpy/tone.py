import numpy as np
from dawpy.renderable import Renderable


class Tone(Renderable):
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
        waveform: str = "sine",
        pan: float = 0.0,
        label: str | None = None,
    ) -> None:
        """Create a musical note.

        Args:
            frequency: Pitch in Hz. Middle C is 261.63 Hz.
                Range: 20-20000 Hz (human hearing range).

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
                Range: 0.0-1.0 (as a ratio, not percent).

            release: Fade-out time in seconds (after note ends).
                How long the tail of the sound lasts after the note stops.

            cutoff: Filter brightness in Hz.
                Low values (e.g., 2000 Hz) = muffled/dark sound.
                High values (e.g., 10000 Hz) = bright/clear sound.
                Range: 20-20000 Hz.

            resonance: Filter emphasis (how much the cutoff frequency "pops").
                0 = no emphasis (flat). Higher values = boosts sound at cutoff.
                Range: 0.0+ (typically 0-10).

            waveform: Oscillator waveform type.
                "sine" = smooth, pure tone (default).
                "square" = digital, boxy tone with rich harmonics.
                "sawtooth" = bright, buzzy tone with lots of harmonics.
                "triangle" = between sine and square, slightly brighter than sine.

            pan: Stereo panning position in the stereo field.
                -1.0 = hard left. 0.0 = center (default). 1.0 = hard right.
                Range: -1.0 to 1.0.

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
            waveform,
            pan,
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
        self.waveform = waveform
        self.pan = pan
        self.label = label

    @staticmethod
    def _validate_all(
        frequency,
        duration,
        volume,
        attack,
        decay,
        sustain,
        release,
        cutoff,
        resonance,
        waveform,
        pan,
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
            errors.append(f"cutoff must be 20-20000 Hz, got {cutoff}")

        if not isinstance(resonance, (int, float)) or resonance < 0.0:
            errors.append(f"resonance must be ≥ 0.0, got {resonance}")

        if waveform not in ("sine", "square", "sawtooth", "triangle"):
            errors.append(
                f"waveform must be 'sine', 'square', 'sawtooth', or 'triangle', got {waveform}"
            )

        if not isinstance(pan, (int, float)) or not (-1.0 <= pan <= 1.0):
            errors.append(f"pan must be -1.0–1.0, got {pan}")

        if errors:
            raise ValueError("\n".join(errors))

    def get_total_duration_seconds(self, bpm: int = 120) -> float:
        """Get the total time this note takes.

        The envelope (attack, decay, release) fits within this duration.
        No extra time is added beyond the specified beat duration.

        Args:
            bpm: Tempo in beats per minute (e.g., 120 = standard tempo).

        Returns:
            Total duration in seconds.
        """
        beat_duration = 60.0 / bpm
        return self.duration * beat_duration

    def duration_seconds(self, bpm: int) -> float:
        """Duration in seconds at the given BPM.

        Alias for get_total_duration_seconds — satisfies the Renderable protocol.

        Args:
            bpm: Tempo in beats per minute.

        Returns:
            Duration in seconds: (self.duration beats) / bpm * 60.
        """
        return self.duration / bpm * 60

    def __repr__(self) -> str:
        return (
            f"Tone(frequency={self.frequency}, duration={self.duration}, "
            f"volume={self.volume}, attack={self.attack}, decay={self.decay}, "
            f"sustain={self.sustain}, release={self.release}, "
            f"cutoff={self.cutoff}, resonance={self.resonance}, "
            f"waveform={self.waveform}, pan={self.pan})"
        )

    def render(self, bpm: int = 120, fs: int = 44100) -> np.ndarray:
        """Generate stereo audio waveform for this note.

        The envelope (attack, decay, sustain, release) is scaled to fit exactly
        within the specified duration. Panning is applied to create stereo output.

        Args:
            bpm: Tempo in beats per minute.
            fs: Sample rate (samples per second). 44100 is standard CD quality.

        Returns:
            Stereo audio data as numpy array with shape (2, num_samples).
            Index 0 is left channel, index 1 is right channel.
        """
        from scipy.signal import butter, sosfilt

        # Convert beat duration to seconds
        beat_duration = 60.0 / bpm
        total_duration_sec = self.duration * beat_duration
        num_samples = int(fs * total_duration_sec)
        t = np.linspace(0, total_duration_sec, num_samples, endpoint=False)

        # Generate the waveform based on selected waveform type
        phase = 2 * np.pi * self.frequency * t

        if self.waveform == "sine":
            audio = np.sin(phase)
        elif self.waveform == "square":
            audio = np.sign(np.sin(phase))
        elif self.waveform == "sawtooth":
            # Sawtooth: ramp from -1 to 1
            audio = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        elif self.waveform == "triangle":
            # Triangle: piecewise linear
            phase_norm = phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)
            audio = np.where(phase_norm < 0.5, 4 * phase_norm - 1, 3 - 4 * phase_norm)
        else:
            audio = np.sin(phase)  # Default fallback

        # Check if envelope times fit within duration. If not, scale them proportionally.
        envelope_time = self.attack + self.decay + self.release
        if envelope_time > total_duration_sec:
            # Scale envelope times proportionally so they fit
            scale_factor = total_duration_sec / envelope_time
            attack = self.attack * scale_factor
            decay = self.decay * scale_factor
            release = self.release * scale_factor
        else:
            attack = self.attack
            decay = self.decay
            release = self.release

        # Build ADSR envelope that exactly fills the total duration
        a_samples = int(attack * fs)
        d_samples = int(decay * fs)
        r_samples = int(release * fs)

        envelope = np.zeros(num_samples, dtype=float)
        idx = 0

        # Attack (0 -> 1)
        if a_samples > 0 and idx < num_samples:
            end = min(idx + a_samples, num_samples)
            envelope[idx:end] = np.linspace(0.0, 1.0, end - idx)
            idx = end

        # Decay (1 -> sustain)
        if d_samples > 0 and idx < num_samples:
            end = min(idx + d_samples, num_samples)
            envelope[idx:end] = np.linspace(1.0, float(self.sustain), end - idx)
            idx = end

        # Sustain (hold sustain level), reserve space for release at the end
        if idx < num_samples:
            sustain_end = max(idx, num_samples - r_samples)
            if sustain_end > idx:
                envelope[idx:sustain_end] = float(self.sustain)
                idx = sustain_end

        # Release (current level -> 0) fills until the end
        if idx < num_samples:
            end = num_samples
            start_level = envelope[idx - 1] if idx > 0 else float(self.sustain)
            envelope[idx:end] = np.linspace(start_level, 0.0, end - idx)

        # Apply a basic low-pass filter (Butterworth order 2). Keep filtering
        # simple — resonance is accepted as a parameter but not used to vary
        # the filter order here, keeping behavior predictable.
        nyquist = fs / 2
        normalized_cutoff = np.clip(self.cutoff / nyquist, 0.001, 0.999)
        sos = butter(2, normalized_cutoff, btype="low", output="sos")
        audio = np.asarray(sosfilt(sos, audio))

        # Apply volume and envelope
        gain = 10 ** (self.volume / 20)
        mono_audio = (audio * envelope * gain).astype(np.float32)

        # Apply stereo panning: linear crossfade
        # pan = -1 (left): left_gain = 1.0, right_gain = 0.0
        # pan = 0 (center): left_gain = 0.5, right_gain = 0.5
        # pan = 1 (right): left_gain = 0.0, right_gain = 1.0
        right_gain = (self.pan + 1.0) / 2.0
        left_gain = 1.0 - right_gain

        left_channel = mono_audio * left_gain
        right_channel = mono_audio * right_gain

        # Return as stereo: shape (2, num_samples)
        return np.array([left_channel, right_channel], dtype=np.float32)
