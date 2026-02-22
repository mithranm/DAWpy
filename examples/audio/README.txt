Audio Assets
============

sound.mp3
---------
  Duration  : 1.584 s  (~3.38 beats @ 128 BPM)
  Sample rate: 24000 Hz, stereo
  Levels    : RMS -24.3 dBFS  /  Peak -1.1 dBFS
  Pitch     : D#3  (155.3 Hz)  — detected via FFT peak (near-sine, flatness 0.05)
  Key       : D# minor  (confidence 0.84)
  Character : dark, tonal  (brightness centroid 687 Hz, flatness 0.05 ≈ pure sine)
  Type      : Sustained melodic tone — held D#3 sine-like note with harmonics
  Use as    : FreeSample("audio/sound.mp3", pad_to_beats=4)
              or synthesise with Tone(frequency=155.56, waveform="sine", cutoff=900)

snap.wav
----------------------------------
  Duration  : 0.060 s  (transient one-shot)
  Sample rate: 48000 Hz, stereo
  Levels    : RMS -13.3 dBFS  /  Peak -0.0 dBFS
  Pitch     : n/a — percussive / noise content (flatness 0.83)
  Key       : C minor  (low confidence — noise-floor artefact)
  Character : bright, noisy/percussive  (brightness centroid 12325 Hz, flatness 0.83)
  Type      : Transient snap / click — very short broadband hit
  Use as    : FreeSample("audio/snap.wav")  — no padding needed, plays at original speed
