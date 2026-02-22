"""Track 2 — Bass only (D# minor, 128 BPM).

Snap percussion lives in track3.py and is mixed in parallel by the Project,
so bass notes here hit on every beat simultaneously with the snap hits.

Structure (104 beats = 26 bars):
  Intro    8 beats — silence; melody introduces itself
  Verse A 16 beats — root-fifth bass line, 2 beats per note
  Verse B 16 beats — walking bass climbing D# minor scale
  Chorus  16 beats — driving 2-beat root/fifth pattern
  Bridge   8 beats — low pedal tones, sustained G#2 → F#2
  Verse C 16 beats — descending bass counter to melody's ascent
  Chorus 2 16 beats — driving root/fifth reprise
  Outro    8 beats — long held notes, bass winds down
"""

from constants import FS, NOTES

from dawpy import Arrangement, Silence, Tone

# ============================================================================
# Tone preset  —  sawtooth bass, low-pass filtered, D# minor
# ============================================================================
bass_params = {
    "volume": -12.0,
    "attack": 0.04,
    "decay": 0.10,
    "sustain": 0.85,
    "release": 0.25,
    "cutoff": 900.0,
    "resonance": 0.5,
    "waveform": "sawtooth",
}


def B(note: str, dur: float) -> Tone:
    """Bass tone shorthand."""
    return Tone(NOTES[note], dur, label=note, **bass_params)


# ============================================================================
# Sections
# ============================================================================

# Intro (8 beats) — bass silent; snap (track3) establishes the groove
intro = Silence(beats=8)

# Verse A (16 beats) — root-fifth bass line, 2 beats per note
verse_a = Arrangement(
    [
        # Bar 1: D# tonic
        B("D#3", 2.0),
        B("D#3", 2.0),
        # Bar 2: D# → A# (fifth)
        B("D#3", 2.0),
        B("A#3", 2.0),
        # Bar 3: G# minor colour
        B("G#3", 2.0),
        B("G#3", 2.0),
        # Bar 4: F# → D# resolve
        B("F#3", 2.0),
        B("D#3", 2.0),
    ]
)

# Verse B (16 beats) — walking bass climbing D# minor scale
verse_b = Arrangement(
    [
        # Bar 1: D#3 → F3
        B("D#3", 2.0),
        B("F3", 2.0),
        # Bar 2: F#3 → G#3
        B("F#3", 2.0),
        B("G#3", 2.0),
        # Bar 3: A#3 → B3
        B("A#3", 2.0),
        B("B3", 2.0),
        # Bar 4: C#4 → D#4 (octave arrival)
        B("C#4", 2.0),
        B("D#4", 2.0),
    ]
)

# Chorus (16 beats) — driving root/fifth, 2 beats per note
chorus = Arrangement(
    [
        # Bar 1: D# → A# pump
        B("D#3", 2.0),
        B("A#3", 2.0),
        # Bar 2: G# → F#
        B("G#3", 2.0),
        B("F#3", 2.0),
        # Bar 3: climb
        B("D#3", 2.0),
        B("G#3", 2.0),
        # Bar 4: resolve
        B("A#3", 2.0),
        B("D#4", 2.0),
    ]
)

# Outro (8 beats) — long held notes, bass winds down
outro = Arrangement(
    [
        B("D#3", 2.0),
        B("A#3", 2.0),
        B("G#3", 2.0),
        B("D#3", 2.0),
    ]
)

# Bridge (8 beats) — low pedal tones, let the melody breathe above
bridge = Arrangement(
    [
        B("G#2", 4.0),  # deep sustained G#  (4 beats)
        B("F#2", 4.0),  # resolve down to F#  (4 beats)
    ]
)

# Verse C (16 beats) — descending bass counter to the rising counter-melody
verse_c = Arrangement(
    [
        B("D#4", 2.0),  # start at the top
        B("C#4", 2.0),
        B("A#3", 2.0),
        B("G#3", 2.0),
        B("F#3", 2.0),
        B("F3", 2.0),
        B("D#3", 2.0),
        B("D#3", 2.0),  # land on root
    ]
)

# ============================================================================
# TRACK2  (104 beats total = 26 bars)
# ============================================================================
TRACK2 = (
    Arrangement()
    .add(intro)  #  8 beats — silence
    .add(verse_a)  # 16 beats
    .add(verse_b)  # 16 beats
    .add(chorus)  # 16 beats
    .add(bridge)  #  8 beats — deep pedal tones
    .add(verse_c)  # 16 beats — descending counter-bass
    .add(chorus)  # 16 beats — driving reprise
    .add(outro)  #  8 beats
)
