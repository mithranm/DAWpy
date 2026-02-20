from dawpy import Timeline, Tone

import sounddevice as sd
import soundfile as sf

timeline = Timeline(bpm=120)
fs = 44100

NOTES = {
    "A": 220.00,
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "G": 392.00,
    "A_high": 440.00,
    "C_high": 523.25,
    "D_high": 587.33,
    "E_high": 659.25,
    "G_high": 783.99,
}

# Each entry: (note, duration, volume, attack, sustain, cutoff, resonance)
# Or with optional per-note decay & release:
# (note, duration, volume, attack, sustain, decay, release, cutoff, resonance)
notes_sequence = [
    # Intro - atmospheric and slow
    ("A", 2, -10, 0.3, 0.8, 3000, 0.0),
    ("A", 2, -10, 0.3, 0.8, 3200, 0.0),
    ("E", 1.5, -9, 0.2, 0.75, 3500, 0.05),
    ("D", 0.5, -8, 0.1, 0.7, 3400, 0.05),
    # Build energy
    ("E", 2, -8, 0.1, 0.8, 4000, 0.1),
    ("A_high", 1, -6, 0.05, 0.8, 4500, 0.15),
    # Main theme - energetic
    ("A_high", 1, -5, 0.05, 0.75, 5000, 0.2),
    ("E_high", 1, -5, 0.05, 0.75, 5100, 0.2),
    ("D_high", 1, -6, 0.08, 0.75, 4800, 0.15),
    ("C_high", 1, -6, 0.08, 0.75, 4600, 0.1),
    ("A_high", 2, -5, 0.05, 0.8, 5200, 0.2),
    # Rhythmic section
    ("G", 0.75, -7, 0.1, 0.7, 4200, 0.08),
    ("E", 0.75, -7, 0.1, 0.7, 4000, 0.08),
    ("D", 0.75, -8, 0.1, 0.7, 3800, 0.05),
    ("E", 0.75, -7, 0.1, 0.7, 4000, 0.08),
    # Development - higher register
    ("A_high", 1, -5, 0.05, 0.75, 5100, 0.18),
    ("E_high", 1, -5, 0.05, 0.75, 5200, 0.18),
    ("A_high", 1, -5, 0.05, 0.75, 5150, 0.18),
    ("D_high", 1, -6, 0.08, 0.75, 4900, 0.15),
    # Variation with longer notes
    ("C_high", 1.5, -4, 0.1, 0.85, 5300, 0.2),
    ("A", 0.5, -8, 0.05, 0.7, 3500, 0.05),
    ("E", 1.5, -8, 0.15, 0.8, 4000, 0.1),
    # Climactic phrase
    ("A_high", 1, -4, 0.05, 0.8, 5200, 0.2),
    ("A_high", 1, -4, 0.05, 0.8, 5250, 0.2),
    ("G_high", 1, -4, 0.05, 0.8, 5100, 0.2),
    ("E_high", 1, -5, 0.08, 0.8, 4900, 0.15),
    # Final resolution
    ("A_high", 3, -3, 0.15, 0.9, 5300, 0.25),
    ("E", 1, -8, 0.2, 0.8, 4000, 0.1),
]

DEFAULT_DECAY = 0.15
DEFAULT_RELEASE = 0.4

for entry in notes_sequence:
    # support both 7-item (current) and 9-item (with decay & release) formats
    if len(entry) == 7:
        note, duration, volume, attack, sustain, cutoff, resonance = entry
        decay = DEFAULT_DECAY
        release = DEFAULT_RELEASE
    elif len(entry) == 9:
        note, duration, volume, attack, sustain, decay, release, cutoff, resonance = (
            entry
        )
    else:
        raise ValueError(
            f"notes_sequence entries must be 7 or 9 items, got {len(entry)}"
        )

    timeline.add_tone(
        Tone(
            frequency=NOTES[note],
            duration=duration,
            volume=volume,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            cutoff=cutoff,
            resonance=resonance,
        )
    )

sd.play(data=timeline.rendered, samplerate=timeline.samplerate)
sd.wait()
sf.write(data=timeline.rendered, file="track1.wav", samplerate=timeline.samplerate)
