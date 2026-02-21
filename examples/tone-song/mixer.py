from dawpy import Project, Tone

from constants import FS
from track1 import MELODY_PATTERN, MELODY_SETTINGS
from track2 import BASS_PATTERN, BASS_SETTINGS

import sounddevice as sd
import soundfile as sf
import threading


if __name__ == "__main__":
    project = Project(bpm=160, samplerate=FS)

    # Track 1: Melodic lead
    track1 = project.add_timeline()
    for _ in range(2):
        for note, duration in MELODY_PATTERN:
            tone = Tone(frequency=note, duration=duration, **MELODY_SETTINGS)
            track1.add_tone(tone)

    # Track 2: Rhythmic bass
    track2 = project.add_timeline()
    for _ in range(2):
        for note, duration in BASS_PATTERN:
            tone = Tone(frequency=note, duration=duration, **BASS_SETTINGS)
            track2.add_tone(tone)

    # Render and play the combined song
    print(f"Project: {project}")
    print(f"Rendering {project.get_duration_seconds():.2f}s of audio...")
    print("Playing... (Press Ctrl+C to stop)\n")

    audio = project.rendered

    # Play audio in a thread so Ctrl+C can interrupt on Windows
    def play_audio():
        sd.play(data=audio, samplerate=project.samplerate)
        sd.wait()

    play_thread = threading.Thread(target=play_audio, daemon=True)
    play_thread.start()

    try:
        while play_thread.is_alive():
            play_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped.")
    else:
        sf.write(data=audio, file="song.wav", samplerate=project.samplerate)
        print("Saved to song.wav")
