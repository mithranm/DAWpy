"""sample.inspect — analyse an audio file and learn how to use it in a song.

Usage::

    from dawpy.sample.inspect import inspect

    info = inspect("audio/sound.mp3")
    print(info.summary())

    # Key fields — pitch vs. brightness
    # ──────────────────────────────────────────────────────────────────────
    # info.fundamental_hz      dominant pitch in Hz, e.g. 155.3 for D#3.
    #                          Detected via Harmonic Product Spectrum (HPS).
    #                          Use this when building a matching Tone:
    #
    #                            Tone(frequency=info.fundamental_hz, ...)
    #
    # info.fundamental_note    nearest note name, e.g. "D#3", "A4".
    #
    # info.spectral_centroid_hz  spectral BRIGHTNESS proxy — the
    #                            power-weighted mean across ALL frequency
    #                            bins.  For a D#3 note with harmonics this
    #                            can read 600-700 Hz even though the actual
    #                            pitch is 155 Hz.  Do NOT use this as pitch.
    # ──────────────────────────────────────────────────────────────────────

    # or with GPU-accelerated CLAP semantic tags:
    info = inspect("audio/sound.mp3", tags=True)
    print(info.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy import signal


# ── helpers ───────────────────────────────────────────────────────────────────


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """(2, N) or (1, N) float32 → (N,) mono float32."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=0).astype(np.float32)


def _db(linear: float) -> float:
    """Linear amplitude → dBFS. Returns -inf for silence."""
    return float(20 * np.log10(max(linear, 1e-10)))


# ── BPM estimation (pure numpy / scipy) ───────────────────────────────────────


def _estimate_bpm(
    mono: np.ndarray,
    sample_rate: int,
    hop: int = 512,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
) -> tuple[float, float]:
    """Estimate BPM via onset-strength autocorrelation.

    Returns
    -------
    (bpm, confidence)
        confidence is in [0, 1] — the normalised autocorrelation peak height.
    """
    # STFT — clamp nperseg to clip length so very short clips don't crash
    n_fft = min(2048, len(mono))
    safe_hop = min(hop, n_fft - 1)
    _, _, S = signal.stft(
        mono, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - safe_hop
    )
    mag = np.abs(S)

    # Spectral flux onset strength: positive first difference summed over freqs
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
    if flux.max() < 1e-8:
        return 0.0, 0.0

    # Autocorrelation of the onset envelope
    corr = np.correlate(flux, flux, mode="full")
    corr = corr[len(corr) // 2 :]  # keep non-negative lags
    corr /= corr[0] + 1e-10  # normalise

    # Convert BPM range to lag range (in frames)
    fps = sample_rate / hop
    lag_min = max(1, int(fps * 60.0 / bpm_max))
    lag_max = int(fps * 60.0 / bpm_min)
    lag_max = min(lag_max, len(corr) - 1)

    if lag_min >= lag_max:
        return 0.0, 0.0

    window = corr[lag_min:lag_max]
    peak_idx = int(np.argmax(window))
    lag = lag_min + peak_idx
    bpm = fps * 60.0 / lag
    confidence = float(window[peak_idx])

    return round(bpm, 1), round(min(confidence, 1.0), 3)


# ── Key detection (Krumhansl-Schmuckler) ──────────────────────────────────────

# Standard K-S probe tones (Krumhansl 1990)
_MAJOR_PROFILE = np.array(
    [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88,
    ],
    dtype=np.float32,
)
_MINOR_PROFILE = np.array(
    [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    ],
    dtype=np.float32,
)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(
    mono: np.ndarray,
    sample_rate: int,
    hop: int = 512,
) -> tuple[str, float]:
    """Estimate musical key via chromagram + Krumhansl-Schmuckler correlation.

    Returns
    -------
    (key_string, confidence)
        key_string e.g. "C major", "A minor"
        confidence in [0, 1]
    """
    # Clamp nperseg to clip length so very short clips don't crash
    n_fft = min(4096, len(mono))
    safe_hop = min(hop, n_fft - 1)
    _, _, S = signal.stft(
        mono, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - safe_hop
    )
    mag = np.abs(S)

    # Map FFT bins to chroma bins
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    chroma = np.zeros(12, dtype=np.float32)
    for i, f in enumerate(freqs):
        if f < 20 or f > 5000:
            continue
        midi = 69 + 12 * math.log2(f / 440.0 + 1e-10)
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += float(mag[i].mean())

    if chroma.sum() < 1e-8:
        return "unknown", 0.0

    chroma = chroma / chroma.sum()

    # Correlate against all 12 major and 12 minor profiles
    best_r = -2.0
    best_key = "unknown"
    for root in range(12):
        major_shifted = np.roll(_MAJOR_PROFILE, root)
        minor_shifted = np.roll(_MINOR_PROFILE, root)
        for profile, mode in [(major_shifted, "major"), (minor_shifted, "minor")]:
            r = float(np.corrcoef(chroma, profile / profile.sum())[0, 1])
            if r > best_r:
                best_r = r
                best_key = f"{_NOTE_NAMES[root]} {mode}"

    # Normalise r from [-1, 1] to [0, 1]
    confidence = round((best_r + 1) / 2, 3)
    return best_key, confidence


# ── Spectral character ─────────────────────────────────────────────────────────


def _spectral_features(
    mono: np.ndarray,
    sample_rate: int,
    hop: int = 512,
) -> tuple[float, float, float]:
    """Compute (spectral_centroid_hz, spectral_flatness, zero_crossing_rate)."""
    n_fft = min(2048, len(mono))
    safe_hop = min(hop, n_fft - 1)
    freqs, _, S = signal.stft(
        mono, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - safe_hop
    )
    mag = np.abs(S).mean(axis=1)  # average magnitude per frequency bin

    # Spectral centroid
    total_energy = mag.sum() + 1e-10
    centroid = float((freqs * mag).sum() / total_energy)

    # Spectral flatness (geometric mean / arithmetic mean)
    mag_nz = mag[mag > 1e-10]
    if len(mag_nz) == 0:
        flatness = 0.0
    else:
        log_mean = float(np.mean(np.log(mag_nz + 1e-10)))
        arith_mean = float(np.mean(mag_nz))
        flatness = float(np.exp(log_mean) / (arith_mean + 1e-10))
        flatness = min(flatness, 1.0)

    # Zero-crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(mono)))) / 2)

    return round(centroid, 1), round(flatness, 4), round(zcr, 4)


# ── Fundamental frequency (Harmonic Product Spectrum) ─────────────────────────


def _estimate_fundamental(
    mono: np.ndarray,
    sample_rate: int,
    f_min: float = 30.0,
    f_max: float = 4000.0,
    n_harmonics: int = 5,
    flatness_hint: float | None = None,
) -> tuple[float, str]:
    """Estimate fundamental frequency using HPS or simple peak, auto-selected.

    Strategy
    --------
    Two methods are run and the appropriate one is chosen based on spectral
    flatness:

    * **Near-sine  (flatness < 0.10)** — the signal has almost no harmonics,
      so HPS has nothing to reinforce and can lock onto a sub-octave ghost.
      Simple FFT peak detection is used instead; it reliably finds the one
      dominant sinusoid.

    * **Harmonic-rich  (flatness ≥ 0.10)** — HPS multiplies progressively
      downsampled copies of the magnitude spectrum so that the fundamental —
      which has energy at f, 2f, 3f … — is reinforced over its harmonics.

    In both cases ``spectral_centroid_hz`` is intentionally NOT used.  The
    centroid is a brightness proxy (power-weighted mean across ALL bins) and
    can read 4–5× higher than the actual pitch for richly harmoniced tones.

    Parameters
    ----------
    f_min, f_max:
        Search range in Hz.  Defaults cover the full musical range (B0–C8).
    n_harmonics:
        Number of harmonic layers to multiply for HPS (default 5).
    flatness_hint:
        Pre-computed global spectral flatness from ``_spectral_features``.
        Pass this when available — it is more reliable than re-computing
        flatness on just the sub-band, which can be inflated by the noise
        floor across a wide frequency range.

    Returns
    -------
    (fundamental_hz, note_name)
        fundamental_hz:  0.0 for percussive / silent / undetectable content.
        note_name:       e.g. "D#3", "A4", or "unknown".
    """
    N = len(mono)
    if N < 512:
        return 0.0, "unknown"

    # Full-signal FFT with Hann window to suppress spectral leakage
    windowed = mono * np.hanning(N)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # Restrict to the search range
    mask = (freqs >= f_min) & (freqs <= f_max)
    sub_freqs = freqs[mask]
    sub_mag = spectrum[mask]

    if len(sub_mag) == 0 or sub_mag.sum() < 1e-10:
        return 0.0, "unknown"

    # Choose flatness: prefer the passed-in global value because sub-band
    # flatness can be inflated by broadband noise in the 30–4000 Hz window.
    if flatness_hint is not None:
        effective_flatness = flatness_hint
    else:
        mag_nz = sub_mag[sub_mag > 1e-10]
        if len(mag_nz) > 0:
            log_mean = float(np.mean(np.log(mag_nz + 1e-10)))
            arith_mean = float(np.mean(mag_nz))
            effective_flatness = float(np.exp(log_mean) / (arith_mean + 1e-10))
        else:
            effective_flatness = 0.0

    if effective_flatness < 0.10:
        # Near-sine: simple FFT peak — one dominant sinusoid, no harmonics to
        # multiply, so HPS would just amplify noise and find ghost sub-octaves.
        peak_idx = int(np.argmax(sub_mag))
    else:
        # Harmonic-rich: HPS reinforces the fundamental via its harmonic series
        hps = sub_mag.copy()
        for h in range(2, n_harmonics + 1):
            downsampled = sub_mag[::h]
            n_common = min(len(hps), len(downsampled))
            hps[:n_common] *= downsampled[:n_common]
        peak_idx = int(np.argmax(hps))

    fundamental = float(sub_freqs[peak_idx])

    if fundamental < f_min:
        return 0.0, "unknown"

    # Map Hz → nearest MIDI note → note name
    midi = round(69 + 12 * math.log2(fundamental / 440.0 + 1e-10))
    octave = midi // 12 - 1
    note_name = f"{_NOTE_NAMES[midi % 12]}{octave}"

    return round(fundamental, 2), note_name


# ── CLAP semantic tags (optional — requires torchaudio + transformers) ─────────

_CLAP_PROMPTS = [
    "drum loop",
    "one-shot drum hit",
    "percussion",
    "melodic loop",
    "vocal sample",
    "ambient pad",
    "bass line",
    "synth lead",
    "guitar",
    "piano",
    "orchestral strings",
    "sound effect or noise",
]


def _clap_tags(
    mono: np.ndarray,
    sample_rate: int,
    device: str = "cuda",
) -> dict[str, float]:
    """Run CLAP to get zero-shot semantic tag probabilities.

    Requires transformers >= 4.38 and a CUDA-capable GPU (falls back to CPU).

    Returns
    -------
    dict mapping tag label → probability (0–1), sorted descending.
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import ClapModel, ClapProcessor
    except ImportError as e:
        raise ImportError(
            "CLAP tags require transformers. Install with: pip install transformers"
        ) from e

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_id = "laion/clap-htsat-unfused"
    # Route through Any: transformers stubs don't fully type ClapProcessor/ClapModel.
    from typing import Any as _Any

    _proc: _Any = ClapProcessor.from_pretrained(model_id)
    _mdl: _Any = ClapModel.from_pretrained(model_id)
    _mdl = _mdl.to(device)
    _mdl.eval()

    # Resample to 48000 Hz (CLAP's expected rate)
    target_sr = 48000
    if sample_rate != target_sr:
        n_target = int(len(mono) * target_sr / sample_rate)
        mono = np.asarray(signal.resample(mono, n_target), dtype=np.float32)

    # Encode audio — ClapProcessor accepts a list of raw waveforms
    audio_inputs: dict[str, _Any] = dict(
        _proc(audio=[mono], sampling_rate=target_sr, return_tensors="pt")
    )
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

    # Encode text prompts
    text_inputs: dict[str, _Any] = dict(
        _proc(text=_CLAP_PROMPTS, return_tensors="pt", padding=True)
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        _audio_out: _Any = _mdl.get_audio_features(**audio_inputs)
        _text_out: _Any = _mdl.get_text_features(**text_inputs)
        # transformers ≥5 may return a ModelOutput object instead of a raw tensor
        audio_embeds: torch.Tensor = (
            _audio_out
            if isinstance(_audio_out, torch.Tensor)
            else _audio_out.pooler_output
        )
        text_embeds: torch.Tensor = (
            _text_out
            if isinstance(_text_out, torch.Tensor)
            else _text_out.pooler_output
        )

        # L2-normalise then cosine similarity → scaled softmax probabilities
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = (audio_embeds @ text_embeds.T).squeeze(0)
        probs = torch.softmax(logits * 10, dim=-1).cpu().numpy()

    return dict(
        sorted(
            zip(_CLAP_PROMPTS, probs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )


# ── SampleInfo dataclass ───────────────────────────────────────────────────────


@dataclass
class SampleInfo:
    """All analysis results for a single audio file.

    Attributes
    ----------
    filepath:           Path to the analysed file.
    duration_seconds:   Total duration in seconds.
    native_sample_rate: Sample rate as stored in the file.
    channels:           Number of audio channels (1=mono, 2=stereo).
    rms_db:             RMS loudness in dBFS.
    peak_db:            Peak amplitude in dBFS.
    bpm:                Estimated tempo in BPM (0 if undetectable).
    bpm_confidence:     BPM confidence 0–1.
    key:                Estimated musical key, e.g. "C major", "A minor".
    key_confidence:     Key confidence 0–1.
    spectral_centroid_hz: Spectral BRIGHTNESS proxy — the power-weighted mean
                        frequency across all FFT bins.  Reflects timbre, not
                        pitch.  For a D#3 with harmonics this may read ~680 Hz
                        even though the fundamental is 155 Hz.
                        Do NOT use this as pitch — use fundamental_hz instead.
    spectral_flatness:  0 = pure tone, 1 = white noise.
    zero_crossing_rate: Proxy for percussiveness / noisiness.
    is_percussive:      Heuristic: True if flatness > 0.3 or zcr > 0.08.
    fundamental_hz:     Dominant pitch in Hz, detected via Harmonic Product
                        Spectrum (HPS).  HPS reinforces the true lowest
                        periodic component, not just the loudest FFT bin.
                        0.0 if the content is percussive or undetectable.
                        Use this — not spectral_centroid_hz — when you want
                        to synthesise a matching Tone:
                            Tone(frequency=info.fundamental_hz, ...)
    fundamental_note:   Nearest note name for fundamental_hz, e.g. "D#3",
                        "A4".  "unknown" if fundamental_hz is 0.
    tags:               CLAP semantic tags {label: probability}, or None if
                        not requested.
    suggested_class:    "FreeSample" or "BeatLockedSample".
    suggested_beats:    Nearest beat count for BeatLockedSample (or None).
    suggested_pad_beats: Bar-aligned pad slot for FreeSample (or None).
    """

    filepath: str
    duration_seconds: float
    native_sample_rate: int
    channels: int
    rms_db: float
    peak_db: float
    bpm: float
    bpm_confidence: float
    key: str
    key_confidence: float
    spectral_centroid_hz: float
    spectral_flatness: float
    zero_crossing_rate: float
    is_percussive: bool
    fundamental_hz: float
    fundamental_note: str
    tags: Optional[dict[str, float]]
    suggested_class: str
    suggested_beats: Optional[int]
    suggested_pad_beats: Optional[int]

    # ── helpers ───────────────────────────────────────────────────────────────

    def beats_at(self, bpm: float) -> float:
        """How many beats this sample spans at the given BPM."""
        return self.duration_seconds * bpm / 60.0

    def nearest_beats(self, bpm: float, grid: int = 1) -> int:
        """Round to the nearest multiple of `grid` beats at the given BPM."""
        raw = self.beats_at(bpm)
        return max(grid, round(raw / grid) * grid)

    def nearest_bar(self, bpm: float, beats_per_bar: int = 4) -> int:
        """Round to the nearest whole bar at the given BPM."""
        bars = self.beats_at(bpm) / beats_per_bar
        return max(1, round(bars)) * beats_per_bar

    # ── display ───────────────────────────────────────────────────────────────

    def summary(self, bpm: Optional[float] = None) -> str:
        """Human-readable summary of the sample with usage suggestions.

        Args:
            bpm: Project BPM to contextualise beat counts. If None, uses the
                 detected BPM (or omits beat-count lines if BPM is 0).
        """
        work_bpm = bpm or (self.bpm if self.bpm > 0 else None)

        filename = Path(self.filepath).name
        bar = "─" * max(len(filename) + 2, 48)

        lines: list[str] = [
            f"\n{filename}  ({self.duration_seconds:.3f}s, "
            f"{self.native_sample_rate} Hz, "
            f"{'stereo' if self.channels == 2 else 'mono'})",
            bar,
        ]

        # Levels
        lines.append(
            f"Levels:    RMS {self.rms_db:+.1f} dBFS   "
            f"Peak {self.peak_db:+.1f} dBFS"
        )

        # Rhythm
        if self.bpm > 0:
            beat_line = (
                f"Rhythm:    ~{self.bpm} BPM  (confidence {self.bpm_confidence:.2f})"
            )
            if work_bpm:
                nb = self.nearest_beats(work_bpm)
                raw_beats = self.beats_at(work_bpm)
                beat_line += (
                    f"  →  {raw_beats:.2f} beats @ {work_bpm} BPM"
                    f"  →  nearest: {nb} beats"
                )
            lines.append(beat_line)
        else:
            lines.append("Rhythm:    BPM undetectable (short or atonal sample)")

        # Key
        if self.key != "unknown":
            lines.append(
                f"Key:       {self.key}  (confidence {self.key_confidence:.2f})"
            )

        # Pitch — only for tonal content (always show fundamental, never centroid)
        if not self.is_percussive and self.fundamental_hz > 0:
            lines.append(
                f"Pitch:     {self.fundamental_hz:.1f} Hz  ({self.fundamental_note})"
                f"  ← use this for Tone(frequency=...)"
            )

        # Character — explicitly label centroid as brightness, not pitch
        char = (
            "bright"
            if self.spectral_centroid_hz > 3000
            else "mid-range" if self.spectral_centroid_hz > 1000 else "dark"
        )
        tone = "noisy/percussive" if self.spectral_flatness > 0.3 else "tonal"
        lines.append(
            f"Character: {char}, {tone}  "
            f"(brightness centroid {self.spectral_centroid_hz:.0f} Hz [≠ pitch], "
            f"flatness {self.spectral_flatness:.2f})"
        )

        # CLAP tags
        if self.tags:
            top = [(k, v) for k, v in self.tags.items() if v > 0.05][:5]
            tag_str = "   ".join(f"{k} {v*100:.0f}%" for k, v in top)
            lines.append(f"Tags:      {tag_str}")

        # Usage suggestions
        lines.append("")
        lines.append("Suggested usage:")

        name_arg = f'"{self.filepath}"'

        if self.suggested_class == "BeatLockedSample":
            n = self.suggested_beats
            lines.append(
                f"  BeatLockedSample({name_arg}, beats={n})"
                f"  # time-stretched to {n} beats at project BPM"
            )
            if work_bpm:
                dur = n / work_bpm * 60 if n else "?"
                lines.append(
                    f"  # at {work_bpm} BPM → " f"{dur:.3f}s"
                    if isinstance(dur, float)
                    else f"  # at {work_bpm} BPM"
                )
            lines.append(f"  # or as a free sample:")
        else:
            lines.append(
                "  # sample is too short or BPM confidence too low for beat-locking:"
            )

        n_pad = self.suggested_pad_beats or (
            self.nearest_bar(work_bpm) if work_bpm else None
        )
        if n_pad:
            lines.append(
                f"  FreeSample({name_arg}, pad_to_beats={n_pad})"
                f"  # plays at original speed, fills {n_pad}-beat slot"
            )
        else:
            lines.append(f"  FreeSample({name_arg})")

        # Tone synthesis suggestion — only for non-percussive tonal samples
        if not self.is_percussive and self.fundamental_hz > 0:
            beat_dur = round(self.beats_at(work_bpm), 2) if work_bpm else "?"
            lines.append("")
            lines.append("  # or synthesise with a matching Tone:")
            lines.append(
                f"  Tone(frequency={self.fundamental_hz},  # {self.fundamental_note}"
            )
            lines.append(f"       duration={beat_dur},  # beats at project BPM")
            lines.append(
                f"       waveform='{'sine' if self.spectral_flatness < 0.1 else 'sawtooth'}'"
                f",  # flatness {self.spectral_flatness:.2f}"
                f" → {'near-sine' if self.spectral_flatness < 0.1 else 'harmonic-rich'}"
            )
            lines.append(f"       volume={self.rms_db:.1f}, ...)")

        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return (
            f"SampleInfo('{Path(self.filepath).name}', "
            f"{self.duration_seconds:.3f}s, "
            f"bpm={self.bpm}, key='{self.key}', "
            f"fundamental={self.fundamental_hz}Hz ({self.fundamental_note}), "
            f"class={self.suggested_class})"
        )


# ── inspect() entry point ─────────────────────────────────────────────────────


def inspect(
    filepath: str,
    tags: bool = False,
    device: str = "cuda",
) -> SampleInfo:
    """Analyse an audio file and return a SampleInfo with usage suggestions.

    Parameters
    ----------
    filepath:
        Path to any audio file readable by soundfile (.wav, .mp3, .ogg,
        .flac, etc.).
    tags:
        If True, run CLAP to produce zero-shot semantic tag probabilities.
        Requires ``transformers`` and a CUDA GPU (falls back to CPU).
        Downloads the model on first call (~600 MB, cached in
        ``~/.cache/huggingface``).
    device:
        PyTorch device for CLAP inference. Defaults to "cuda" with
        automatic fallback to "cpu" if CUDA is unavailable.

    Returns
    -------
    SampleInfo
        Dataclass with all analysis results and a ``summary()`` method that
        prints a human-readable usage guide.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist.
    ValueError:
        If the file cannot be read.

    Example
    -------
    ::

        info = inspect("drums.wav", tags=True)
        print(info.summary(bpm=128))

        loop = BeatLockedSample("drums.wav", beats=info.suggested_beats)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not read '{filepath}': {exc}") from exc

    # audio is (num_samples, channels), sf convention
    num_channels = audio.shape[1]
    duration = audio.shape[0] / sample_rate

    # Work in (channels, samples) internally, then mono for analysis
    audio_ch = audio.T  # (channels, samples)
    mono = _to_mono(audio_ch)

    # ── Levels ────────────────────────────────────────────────────────────────
    rms_db = _db(float(np.sqrt(np.mean(mono**2))))
    peak_db = _db(float(np.max(np.abs(mono))))

    # ── Rhythm ────────────────────────────────────────────────────────────────
    bpm, bpm_conf = _estimate_bpm(mono, sample_rate)

    # ── Key ───────────────────────────────────────────────────────────────────
    key, key_conf = _estimate_key(mono, sample_rate)

    # ── Spectral character ────────────────────────────────────────────────────
    centroid, flatness, zcr = _spectral_features(mono, sample_rate)
    is_percussive = flatness > 0.3 or zcr > 0.08

    # ── Fundamental pitch (HPS or simple peak, chosen by flatness) ───────────
    # Use the already-computed global flatness as a hint so that sub-band
    # noise inflation cannot force the wrong detection strategy.
    # Use fundamental_hz — NOT spectral_centroid_hz — to build a matching Tone.
    fundamental_hz, fundamental_note = _estimate_fundamental(
        mono, sample_rate, flatness_hint=flatness
    )

    # ── CLAP tags (optional) ─────────────────────────────────────────────────
    tag_dict: dict[str, float] | None = None
    if tags:
        tag_dict = _clap_tags(mono, sample_rate, device=device)

    # ── Usage suggestions ─────────────────────────────────────────────────────
    # Use detected BPM for beat counting; round to nearest power-of-2 beats
    ref_bpm = bpm if bpm > 0 else 120.0
    raw_beats = duration * ref_bpm / 60.0

    # Snap to nearest power of 2 that's >= 1
    def _nearest_pow2(x: float) -> int:
        if x < 1:
            return 1
        p = round(math.log2(x))
        return max(1, 2**p)

    snapped_beats = _nearest_pow2(raw_beats)

    # Recommend BeatLockedSample if BPM confidence is high and duration is
    # close to a musical subdivision (within 15% of snapped value)
    beat_error = abs(raw_beats - snapped_beats) / max(snapped_beats, 1)
    use_beat_locked = bpm_conf >= 0.35 and beat_error <= 0.20

    if use_beat_locked:
        suggested_class = "BeatLockedSample"
        suggested_beats = snapped_beats
        suggested_pad = None
    else:
        suggested_class = "FreeSample"
        suggested_beats = None
        # Suggest pad_to_beats snapped to nearest bar (4 beats per bar)
        suggested_pad = max(4, round(raw_beats / 4) * 4) if raw_beats >= 1 else None

    return SampleInfo(
        filepath=str(path),
        duration_seconds=round(duration, 6),
        native_sample_rate=sample_rate,
        channels=num_channels,
        rms_db=round(rms_db, 2),
        peak_db=round(peak_db, 2),
        bpm=bpm,
        bpm_confidence=bpm_conf,
        key=key,
        key_confidence=key_conf,
        spectral_centroid_hz=centroid,
        spectral_flatness=flatness,
        zero_crossing_rate=zcr,
        is_percussive=is_percussive,
        fundamental_hz=fundamental_hz,
        fundamental_note=fundamental_note,
        tags=tag_dict,
        suggested_class=suggested_class,
        suggested_beats=suggested_beats,
        suggested_pad_beats=suggested_pad,
    )
