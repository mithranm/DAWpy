"""Renderable ABC — the base class all composable audio items must extend.

Subclass Renderable and implement render() and duration_seconds().
play() and save() are provided for free.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf


class Renderable(ABC):
    """Abstract base class for anything that can produce stereo audio.

    Subclasses: Tone, Silence, Arrangement, Sample (and its subclasses).
    All render methods return a (2, num_samples) float32 stereo array.

    Subclasses must implement:
        render(bpm, sample_rate) -> np.ndarray
        duration_seconds(bpm) -> float

    play() and save() are provided here and require no override.
    """

    # Optional human-readable label shown in the visualizer.
    # Concrete subclasses set this in their own __init__.
    label: str | None

    @abstractmethod
    def render(self, bpm: int, sample_rate: int) -> np.ndarray:
        """Render to a stereo float32 array.

        Args:
            bpm: Tempo in beats per minute. Items in musical time use this to
                 convert beat durations to samples. Items in wall-clock time
                 (e.g. free AudioSamples) ignore it.
            sample_rate: Output sample rate in Hz (e.g. 44100).

        Returns:
            Stereo audio as shape (2, num_samples), dtype float32.
            Row 0 = left channel, row 1 = right channel.
        """

    @abstractmethod
    def duration_seconds(self, bpm: int) -> float:
        """Duration of this item in seconds at the given BPM.

        Args:
            bpm: Tempo in beats per minute. Items with BPM-independent
                 duration (e.g. FreeSample) accept but may ignore this.

        Returns:
            Duration in seconds as a float.
        """

    def play(self, bpm: int = 120, sample_rate: int = 44100) -> None:
        """Render and play through the system audio output.

        Blocks until playback finishes. Ctrl+C stops cleanly.

        Args:
            bpm:         Tempo in beats per minute (default 120).
            sample_rate: Output sample rate in Hz (default 44100).
        """
        Renderable._play_audio(self.render(bpm, sample_rate), sample_rate)

    def save(self, path: str, bpm: int = 120, sample_rate: int = 44100) -> None:
        """Render and save to an audio file.

        The output format is inferred from the file extension.
        Supports .wav, .flac, .ogg, and any other soundfile-supported format.
        Parent directories are created automatically.

        Args:
            path:        Output file path (e.g. ``"out.wav"`` or ``"out.flac"``).
            bpm:         Tempo in beats per minute (default 120).
            sample_rate: Output sample rate in Hz (default 44100).
        """
        Renderable._save_audio(self.render(bpm, sample_rate), sample_rate, path)

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
        """Play a (2, num_samples) stereo array through the system audio output."""
        pcm = audio.T.astype(np.float32)

        def _play():
            sd.play(pcm, samplerate=sample_rate)
            sd.wait()

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                thread.join(timeout=0.1)
        except KeyboardInterrupt:
            sd.stop()
            print("\nPlayback stopped.")

    @staticmethod
    def _save_audio(audio: np.ndarray, sample_rate: int, path: str) -> None:
        """Write a (2, num_samples) stereo array to an audio file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, audio.T.astype(np.float32), sample_rate)
