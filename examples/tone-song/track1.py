"""Track 1 — Melody (D# minor, 128 BPM).

Structure (104 beats = 26 bars):
  Intro   8 beats  — sound.mp3 motif x4
  Verse A 16 beats — descending D# minor melody
  Verse B 16 beats — ascending answer phrase
  Chorus  16 beats — high-energy peak
  Bridge   8 beats — sparse high-register floating phrase
  Verse C 16 beats — triangle counter-melody + sound.mp3 motif
  Chorus  16 beats — high-energy reprise
  Outro    8 beats — sound.mp3 motif x4, fade

sound.mp3 analysis (see examples/audio/readme.txt):
  Pitch D#3 (155.3 Hz), duration 0.60s of real audio
  starts at 0.040s, content ends at ~0.640s — cropped accordingly.
"""

from pathlib import Path

from constants import FS, NOTES

from dawpy import Arrangement, FreeSample, Silence, Tone

_AUDIO = Path(__file__).parent.parent / "audio"

# ============================================================================
# Audio samples
# ============================================================================

# sound.mp3: D#3 sustained tone.  Envelope data shows real audio runs
# 0.040s–0.640s; everything outside that is silence / MP3 padding.
# crop_seconds=0.60 keeps exactly that content; pad_to_beats=2 fills a
# clean 2-beat slot (0.9375s @ 128 BPM) with a short tail of silence.
SOUND = FreeSample(
    str(_AUDIO / "sound.mp3"),
    offset_seconds=0.04,
    crop_seconds=0.60,
    pad_to_beats=2,
)

# ============================================================================
# Tone presets  —  D# natural minor: D# F F# G# A# B C#
# ============================================================================
melody_params = {
    "volume": -10.0,
    "attack": 0.02,
    "decay": 0.08,
    "sustain": 0.85,
    "release": 0.20,
    "cutoff": 4000.0,
    "resonance": 1.0,
    "waveform": "sine",
}
counter_params = {
    "volume": -15.0,
    "attack": 0.01,
    "decay": 0.06,
    "sustain": 0.70,
    "release": 0.15,
    "cutoff": 6000.0,
    "resonance": 2.0,
    "waveform": "triangle",
}


def M(note: str, dur: float) -> Tone:
    """Melody tone shorthand."""
    return Tone(NOTES[note], dur, label=note, **melody_params)


def C(note: str, dur: float) -> Tone:
    """Counter-melody tone shorthand."""
    return Tone(NOTES[note], dur, label=note, **counter_params)


# ============================================================================
# Phrases
# ============================================================================

# Intro / outro hook: sound.mp3 motif repeated (4 × 2 beats = 8 beats)
intro_hook = Arrangement([SOUND, SOUND, SOUND, SOUND])

# Verse A — descending question (16 beats)
# D#5 . C#5 . A#4 .. G#4 . F#4 . D#4 ..
# sound motif . G#4 .. A#4 .. F#4 ..
verse_a = Arrangement(
    [
        M("D#5", 1.0),
        M("C#5", 1.0),
        M("A#4", 2.0),
        M("G#4", 1.0),
        M("F#4", 1.0),
        M("D#4", 2.0),
        SOUND,  # 2 beats
        M("G#4", 2.0),
        M("A#4", 2.0),
        M("F#4", 2.0),
    ]
)

# Verse B — ascending answer (16 beats)
# Climbs D# minor scale then descends back home
verse_b = Arrangement(
    [
        M("F4", 1.0),  # 1
        M("G#4", 1.0),  # 2
        M("A#4", 1.0),  # 3
        M("B4", 1.0),  # 4
        M("C#5", 2.0),  # 6
        M("D#5", 2.0),  # 8
        M("A#4", 1.0),  # 9
        M("G#4", 1.0),  # 10
        M("F#4", 1.0),  # 11
        M("F4", 1.0),  # 12
        M("D#4", 2.0),  # 14
        M("D#3", 2.0),  # 16
    ]
)

# Chorus — high-energy peak (16 beats)
# D#5 F5 G#5 . A#5 .. G#5 . F#4 . D#5 ..
# C#5 .. A#4 .. G#4 .. D#4 ..
chorus = Arrangement(
    [
        M("D#5", 0.5),
        M("F5", 0.5),
        M("G#5", 1.0),
        M("A#5", 2.0),
        M("G#5", 1.0),
        M("F#4", 1.0),
        M("D#5", 2.0),
        M("C#5", 2.0),
        M("A#4", 2.0),
        M("G#4", 2.0),
        M("D#4", 2.0),
    ]
)

# Bridge — sparse high-register floating phrase (8 beats)
bridge = Arrangement(
    [
        M("A#5", 2.0),  # soaring high note
        M("G#5", 2.0),
        M("F#5", 2.0),
        M("D#5", 2.0),  # settle back to tonic
    ]
)

# Verse C — triangle counter-melody, builds back toward second chorus (16 beats)
verse_c = Arrangement(
    [
        C("D#5", 1.0),  # 1
        C("F5", 1.0),  # 2
        C("G#5", 2.0),  # 4
        C("A#5", 2.0),  # 6
        C("D#5", 2.0),  # 8
        SOUND,  # 2 beats — motif callback (10)
        C("F5", 2.0),  # 12
        C("D#4", 2.0),  # 14
        C("C#4", 2.0),  # 16
    ]
)

# ============================================================================
# TRACK1  (104 beats total = 26 bars)
# ============================================================================
TRACK1 = (
    Arrangement()
    .add(intro_hook)  #  8 beats — sound.mp3 x4
    .add(verse_a)  # 16 beats
    .add(verse_b)  # 16 beats
    .add(chorus)  # 16 beats
    .add(bridge)  #  8 beats — floating high phrase
    .add(verse_c)  # 16 beats — counter-melody
    .add(chorus)  # 16 beats — second chorus reprise
    .add(intro_hook)  #  8 beats — outro echo
)
