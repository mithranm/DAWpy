"""Track 3 — Snap percussion (128 BPM).

Runs in parallel with TRACK2 (bass) via the Project mixer so snap hits
land simultaneously with bass note attacks rather than sequentially.

Snap pattern per section (104 beats = 26 bars):
  Intro    8 beats  — silence       (melody sets the scene alone)
  Verse A 16 beats  — beats 1 & 3  (sparse, space for melody)
  Verse B 16 beats  — every beat   (urgency builds)
  Chorus  16 beats  — every beat   (maximum drive)
  Bridge   8 beats  — beats 1 & 3  (breathing space before second half)
  Verse C 16 beats  — beats 1 & 3  (sparse counter-melody section)
  Chorus 2 16 beats  — every beat   (full energy reprise)
  Outro    8 beats  — silence       (snap drops out, bass winds down)
"""

from pathlib import Path

from constants import FS

from dawpy import Arrangement, FreeSample, Silence

_AUDIO = Path(__file__).parent.parent / "audio"

# snap.wav: 0.060s bright transient.  pad_to_beats=1 holds a 1-beat slot
# so the arrangement stays on the beat grid.
SNAP = FreeSample(str(_AUDIO / "snap.wav"), pad_to_beats=1)

# ============================================================================
# Snap patterns
# ============================================================================


# Every beat: used in intro, verse B, chorus
def _every_beat(n_beats: int) -> Arrangement:
    return Arrangement([SNAP] * n_beats)


# Beats 1 & 3 of each bar (4/4): snap, rest, snap, rest per bar
def _beats_1_and_3(n_bars: int) -> Arrangement:
    bar = [SNAP, Silence(beats=1), SNAP, Silence(beats=1)]
    items = bar * n_bars
    return Arrangement(items)


# ============================================================================
# Sections
# ============================================================================

intro = Silence(beats=8)  #  8 beats — silence, melody speaks alone
verse_a = _beats_1_and_3(4)  # 16 beats — snap on 1 & 3 only
verse_b = _every_beat(16)  # 16 beats — snap every beat
chorus = _every_beat(16)  # 16 beats — snap every beat
bridge = _beats_1_and_3(2)  #  8 beats — sparse, breathing space
verse_c = _beats_1_and_3(4)  # 16 beats — sparse counter-melody section
outro = Silence(beats=8)  #  8 beats — snap drops out

# ============================================================================
# TRACK3  (104 beats total = 26 bars)
# ============================================================================
TRACK3 = (
    Arrangement()
    .add(intro)  #  8 beats — silence
    .add(verse_a)  # 16 beats — 1 & 3
    .add(verse_b)  # 16 beats — every beat
    .add(chorus)  # 16 beats — every beat
    .add(bridge)  #  8 beats — 1 & 3
    .add(verse_c)  # 16 beats — 1 & 3
    .add(chorus)  # 16 beats — every beat (second chorus)
    .add(outro)  #  8 beats — silence
)
