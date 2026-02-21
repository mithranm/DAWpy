from dawpy import Timeline, Tone

from constants import NOTES, FS

import sounddevice as sd
import soundfile as sf
import threading


# Note duration constants (in beats at 160 BPM)
whole = 4.0
half = 2.0
quarter = 1.0
eighth = 0.5


# Bass pattern - slow harmonic foundation
# Whole and half notes that create a bed under the melody
BASS_PATTERN = [
    (NOTES["C"], whole),
    (NOTES["G"], whole),
    (NOTES["A"], whole),
]

BASS_SETTINGS = {
    "volume": -12.0,
    "attack": 0.05,
    "decay": 0.15,
    "sustain": 0.75,
    "release": 0.3,
    "cutoff": 3500.0,
    "resonance": 1.5,
}


def create_timeline():
    """Create and populate a Timeline with the bass pattern."""
    timeline = Timeline(bpm=160, samplerate=FS)
    for _ in range(2):
        for note, duration in BASS_PATTERN:
            tone = Tone(frequency=note, duration=duration, **BASS_SETTINGS)
            timeline.add_tone(tone)
    return timeline


if __name__ == "__main__":
    # Standalone: build and play/save the bass track
    try:
        timeline = create_timeline()
        print("Playing... (Press Ctrl+C to stop)\n")
        audio = timeline.rendered

        # Play audio in a thread so Ctrl+C can interrupt on Windows
        def play_audio():
            sd.play(data=audio, samplerate=timeline.samplerate)
            sd.wait()

        play_thread = threading.Thread(target=play_audio, daemon=True)
        play_thread.start()

        while play_thread.is_alive():
            play_thread.join(timeout=0.1)

        sf.write(data=audio, file="track2.wav", samplerate=timeline.samplerate)
        print("Saved to track2.wav")
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped.")
