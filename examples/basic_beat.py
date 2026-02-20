# examples/basic_beat.py

import sys
import os
# Hack to import dawpy without installing it yet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dawpy.timeline import Project, Time

# 120 BPM means 1 beat = 0.5 seconds
song = Project(bpm=120)

# Let's test the grid
positions =

for pos in positions:
    sec = song.time_to_seconds(pos)
    samples = song.seconds_to_samples(sec)
    print(f"{pos} -> {sec:.2f} seconds -> sample index {samples}")