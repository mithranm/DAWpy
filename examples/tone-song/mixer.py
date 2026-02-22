from pathlib import Path

from constants import FS

from dawpy import Project, ProjectVisualizer

from track1 import TRACK1
from track2 import TRACK2
from track3 import TRACK3

bpm = 128

project = Project(bpm=bpm, sample_rate=FS)

# All three tracks start at t=0 and are summed in parallel by the Project.
#   TRACK1 — melody (sine tones + sound.mp3 motif)
#   TRACK2 — bass (sawtooth, hits simultaneously with snap)
#   TRACK3 — snap percussion (parallel with bass so they land together)
project.add(TRACK1)
project.add(TRACK2)
project.add(TRACK3)

# Export WAV to the exports/ folder next to this file
_EXPORTS = Path(__file__).parent / "exports"
project.save(str(_EXPORTS / "tone_song.wav"))
print(f"Exported → {_EXPORTS / 'tone_song.wav'}")

ProjectVisualizer(project, title="DAWpy — Tone Song").show()
