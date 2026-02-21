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
dotted_quarter = 1.5


# Melody pattern - bright, energetic lead
MELODY_PATTERN = [
    # Ascending phrase
    (NOTES["C_high"], eighth),
    (NOTES["E_high"], eighth),
    (NOTES["G_high"], eighth),
    (NOTES["A_high"], eighth),
    # Descending back
    (NOTES["G_high"], eighth),
    (NOTES["E_high"], eighth),
    (NOTES["C_high"], eighth),
    (NOTES["D_high"], eighth),
    # Sustained notes
    (NOTES["E_high"], dotted_quarter),
    (NOTES["G_high"], dotted_quarter),
    (NOTES["A_high"], quarter),
    # Resolution
    (NOTES["G_high"], quarter),
    (NOTES["E_high"], quarter),
    (NOTES["C_high"], half),
]

MELODY_SETTINGS = {
    "volume": -10.0,
    "attack": 0.02,
    "decay": 0.05,
    "sustain": 0.7,
    "release": 0.2,
    "cutoff": 8000.0,
    "resonance": 2.0,
}


def create_timeline():
    """Create and populate a Timeline with the melody pattern."""
    timeline = Timeline(bpm=160, samplerate=FS)
    for _ in range(2):
        for note, duration in MELODY_PATTERN:
            tone = Tone(frequency=note, duration=duration, **MELODY_SETTINGS)
            timeline.add_tone(tone)
    return timeline


if __name__ == "__main__":
    # Standalone: build and play/save the melody track
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

        sf.write(data=audio, file="track1.wav", samplerate=timeline.samplerate)
        print("Saved to track1.wav")
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped.")
