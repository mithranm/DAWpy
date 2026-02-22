"""ProjectVisualizer — a read-only DAW-style tkinter timeline for a Project.

Each arrangement added to the project is shown as a horizontal track lane.
Each Renderable inside the arrangement is drawn as a labelled, colour-coded
block.  Clicking anywhere on the timeline starts playback from that point;
the checkboxes on the left mute / unmute individual tracks in real time.

Usage::

    from dawpy import Project
    from dawpy.visualizer import ProjectVisualizer

    project = Project(bpm=128)
    project.add(melody_arr)
    project.add(bass_arr, offset=2.0)

    ProjectVisualizer(project).show()       # blocks until window is closed
"""

from __future__ import annotations

import base64
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from dawpy.project import Project, _Slot

# ── Colour palette (dark theme) ───────────────────────────────────────────────

_BG = "#1E1E2E"
_RULER_BG = "#181825"
_TRACK_BG = "#252535"
_TRACK_ALT = "#222232"
_LABEL_BG = "#13131F"
_TEXT = "#CDD6F4"
_SUBTEXT = "#585B70"
_GRID = "#2D2D4A"
_PLAYHEAD = "#F38BA8"
_SEP = "#111120"
_MUTED_FILL = "#2D2D40"
_MUTED_OUTLINE = "#3D3D55"
_MUTED_TEXT = "#404055"

# One accent colour per track (cycles for > 8 tracks)
_TRACK_PALETTE: list[str] = [
    "#3B82F6",  # blue
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#84CC16",  # lime
]


def _track_color(index: int, muted: bool = False) -> str:
    """Return the accent colour for track *index* (or the muted stub)."""
    return _MUTED_OUTLINE if muted else _TRACK_PALETTE[index % len(_TRACK_PALETTE)]


# ── Spectrogram helpers ───────────────────────────────────────────────────────


def _magma_rgb(v: np.ndarray) -> np.ndarray:
    """Vectorised magma-like colormap.  Input: (H, W) float32 in [0, 1].
    Output: (H, W, 3) uint8."""
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(v * 1.8, 0.0, 1.0)
    g = np.clip(v * 1.4 - 0.3, 0.0, 1.0)
    b = np.clip(0.6 - v * 0.9, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _ppm_b64(rgb: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 array as a base64 PPM string for PhotoImage."""
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode()
    return base64.b64encode(header + rgb.tobytes()).decode()


def _darken(hex_color: str, factor: float = 0.65) -> str:
    """Return a darkened version of a hex colour string."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"#{int(r * factor):02X}{int(g * factor):02X}{int(b * factor):02X}"
    except Exception:
        return hex_color


def _tooltip_lines(item: object, dur_s: float, bpm: int) -> list[str]:
    """Return a list of display lines for the hover tooltip."""
    from dawpy.tone import Tone
    from dawpy.silence import Silence
    from dawpy.sample._base import Sample
    from dawpy.composition import Arrangement

    lines: list[str] = []

    if isinstance(item, Tone):
        beats = item.duration
        lines += [
            f"Tone  --  {item.frequency:.2f} Hz",
            "",
            f"  Duration   {beats:.3g} beat{'s' if beats != 1 else ''}  /  {dur_s:.3f} s",
            f"  Waveform   {item.waveform}",
            f"  Volume     {item.volume:.1f} dBFS",
            f"  Pan        {item.pan:+.2f}",
            "",
            f"  Attack     {item.attack * 1000:.0f} ms",
            f"  Decay      {item.decay * 1000:.0f} ms",
            f"  Sustain    {item.sustain * 100:.0f} %",
            f"  Release    {item.release * 1000:.0f} ms",
            "",
            f"  Filter     {item.cutoff:.0f} Hz  resonance {item.resonance:.2f}",
        ]

    elif isinstance(item, Sample):
        import os

        fname = os.path.basename(item.filepath)
        lines += [
            f"Sample  --  {fname}",
            "",
            f"  Duration   {dur_s:.3f} s",
            f"  File SR    {item.sample_rate} Hz",
            f"  Pan        {item.pan:+.2f}",
        ]
        if item.offset_seconds:
            lines.append(f"  Offset     {item.offset_seconds:.3f} s")
        if item.crop_seconds is not None:
            lines.append(f"  Crop       {item.crop_seconds:.3f} s")
        lines.append(f"  Path       {item.filepath}")

    elif isinstance(item, Arrangement):
        n = len(item.items)
        lines += [
            f"Arrangement  --  {n} item{'s' if n != 1 else ''}",
            "",
            f"  Duration   {dur_s:.3f} s  /  {item.duration_seconds(bpm) / (60.0 / bpm):.3f} beats",
        ]

    elif isinstance(item, Silence):
        lines += [f"Silence  --  {dur_s:.3f} s"]

    else:
        lbl = getattr(item, "label", None) or type(item).__name__
        lines += [f"{lbl}  --  {dur_s:.3f} s"]

    # Custom label annotation (shown if set and not already the heading)
    user_lbl = getattr(item, "label", None)
    if user_lbl:
        lines.insert(0, f'  "{user_lbl}"')
        lines.insert(1, "")

    return lines


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _freq_to_note(freq: float) -> str:
    """Return the nearest MIDI note name for *freq* Hz, e.g. ``'D#3'``."""
    if freq <= 0:
        return "?"
    import math

    try:
        midi = round(69 + 12 * math.log2(freq / 440.0))
    except (ValueError, ZeroDivisionError):
        return "?"
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _flatten_timeline(item: Any, bpm: int, t_start: float = 0.0) -> list[tuple]:
    """Recursively expand nested Arrangements to leaf-level Renderables.

    Returns a list of ``(leaf_item, start_seconds, end_seconds, duration_seconds)``
    tuples.  Nested Arrangements are transparently expanded so the caller
    always sees only the primitives (Tone, Sample, Silence, …).

    Args:
        item:    Root item — usually a top-level Arrangement.
        bpm:     Tempo in beats per minute.
        t_start: Absolute start time of *item* in seconds (default 0.0).
    """
    from dawpy.composition import Arrangement
    from dawpy.renderable import Renderable as _Renderable  # noqa: F401

    result: list[tuple] = []
    if isinstance(item, Arrangement):
        cursor = t_start
        for child in item.items:
            child_dur = child.duration_seconds(bpm)
            result.extend(_flatten_timeline(child, bpm, cursor))
            cursor += child_dur
    else:
        dur = item.duration_seconds(bpm)  # type: ignore[union-attr]
        result.append((item, t_start, t_start + dur, dur))
    return result


def _item_label(item: object) -> str:
    """Short display label for a single Renderable.

    Uses the item's own ``label`` attribute when set.  For Tones shows
    the note name (e.g. ``'D#3'``); for Samples shows the filename stem.
    Falls back to the class name for any other type.
    """
    lbl = getattr(item, "label", None)
    if lbl is not None:
        return lbl
    from dawpy.tone import Tone

    if isinstance(item, Tone):
        return _freq_to_note(item.frequency)
    from dawpy.sample._base import Sample

    if isinstance(item, Sample):
        import os

        return os.path.splitext(os.path.basename(item.filepath))[0]
    return type(item).__name__


# ── Main class ────────────────────────────────────────────────────────────────


class ProjectVisualizer:
    """Read-only DAW-style timeline visualizer for a :class:`~dawpy.project.Project`.

    Layout
    ------
    - Top bar   — title, BPM / SR info, Play-from-start and Stop buttons.
    - Left panel — one row per arrangement with a mute checkbox and track info.
    - Ruler      — scrolls horizontally; shows seconds and beat numbers.
    - Timeline   — coloured blocks for every Renderable; scrolls x + y.

    Interactions
    ------------
    - **Click** anywhere on the timeline → start playback from that time.
    - **Checkbox** → mute / unmute a track (re-renders on the fly).
    - **Escape** or **Stop** button → stop playback.
    - **Scroll wheel** → scroll vertically.
    - **Shift + scroll wheel** → scroll horizontally.
    """

    # ── Layout tunables ───────────────────────────────────────────────────────
    ROW_H: int = 58  # pixels per track row
    RULER_H: int = 36  # pixels for the time ruler
    LABEL_W: int = 190  # pixels for the left label panel
    PX_PER_SEC: float = 110.0  # pixels per second  (horizontal zoom)
    BLOCK_PAD: int = 4  # vertical inner padding for block rectangles
    TICK_INTERVAL_MS: int = 40  # playhead refresh interval (ms)
    BEATS_PER_BAR: int = 4  # time signature numerator (4/4 default)

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self, project: "Project", *, title: str = "DAWpy — Project Visualizer"
    ) -> None:
        """
        Args:
            project: The :class:`~dawpy.project.Project` to visualise.
            title:   Window title string.
        """
        self.project = project
        self._window_title = title

        # Playback state
        self._play_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._play_start_proj: float = 0.0
        self._play_start_wall: float = 0.0
        self._playhead_t: float = 0.0

        # Per-track mute vars (populated in _draw_labels)
        self._enabled: list[tk.BooleanVar] = []
        # Spectrogram PhotoImage cache — must be kept alive (GC protection)
        self._spec_images: list[tk.PhotoImage] = []
        # Maps block_tag → (item, duration_seconds) for hover tooltips
        self._block_entry_map: dict[str, tuple[object, float]] = {}

        # Tkinter widgets (populated in _build_ui)
        self._root: tk.Tk
        self._ruler: tk.Canvas
        self._label_canvas: tk.Canvas
        self._tl: tk.Canvas
        self._h_scroll: ttk.Scrollbar
        self._v_scroll: ttk.Scrollbar
        self._time_var: tk.StringVar
        # Tooltip state
        self._tooltip_win: tk.Toplevel | None = None
        self._tooltip_tag: str | None = None
        self._tooltip_after: str | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Open the visualizer window and block until it is closed."""
        self._build_ui()
        self._root.mainloop()

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = tk.Tk()
        root.title(self._window_title)
        root.configure(bg=_BG)
        root.minsize(900, 400)
        self._root = root
        self._time_var = tk.StringVar(value="0:00.00   Bar 1  ♩1")

        n = len(self.project._slots)
        total_dur = max(self.project.duration_seconds(), 1.0)
        total_w = int(total_dur * self.PX_PER_SEC) + 200
        total_h = n * self.ROW_H

        # ── Controls bar ─────────────────────────────────────────────────────
        ctrl = tk.Frame(root, bg=_BG, pady=7, padx=10)
        ctrl.grid(row=0, column=0, columnspan=3, sticky="ew")

        tk.Label(
            ctrl,
            text="DAWpy Visualizer",
            bg=_BG,
            fg=_TEXT,
            font=("Helvetica", 13, "bold"),
        ).pack(side="left", padx=(0, 16))

        tk.Label(
            ctrl,
            text=(
                f"♩{self.project.bpm} BPM  ·  "
                f"{self.project.sample_rate} Hz  ·  "
                f"{total_dur:.2f} s  ·  "
                f"{n} track{'s' if n != 1 else ''}"
            ),
            bg=_BG,
            fg=_SUBTEXT,
            font=("Helvetica", 10),
        ).pack(side="left", padx=(0, 20))

        tk.Label(
            ctrl,
            textvariable=self._time_var,
            bg=_BG,
            fg="#A6E3A1",
            font=("Helvetica", 10, "bold"),
        ).pack(side="left")

        self._stop_btn = tk.Button(
            ctrl,
            text="■  Stop",
            bg="#313244",
            fg=_TEXT,
            activebackground="#45475A",
            relief="flat",
            padx=10,
            pady=3,
            font=("Helvetica", 10),
            command=self._stop_playback,
        )
        self._stop_btn.pack(side="right", padx=4)

        self._play_btn = tk.Button(
            ctrl,
            text="▶  Play from start",
            bg="#313244",
            fg=_TEXT,
            activebackground="#45475A",
            relief="flat",
            padx=10,
            pady=3,
            font=("Helvetica", 10),
            command=lambda: self._start_playback(0.0),
        )
        self._play_btn.pack(side="right", padx=4)

        # ── Corner (non-scrolling header above labels) ────────────────────────
        corner = tk.Frame(root, bg=_LABEL_BG, width=self.LABEL_W, height=self.RULER_H)
        corner.grid(row=1, column=0, sticky="nsew")
        corner.grid_propagate(False)
        tk.Label(
            corner,
            text=f"TRACKS  ♩{self.project.bpm}",
            bg=_LABEL_BG,
            fg=_SUBTEXT,
            font=("Helvetica", 8, "bold"),
        ).place(relx=0.5, rely=0.5, anchor="center")

        # ── Ruler canvas ──────────────────────────────────────────────────────
        ruler = tk.Canvas(
            root,
            bg=_RULER_BG,
            height=self.RULER_H,
            highlightthickness=0,
            xscrollincrement=1,
        )
        ruler.grid(row=1, column=1, sticky="ew")
        ruler.configure(scrollregion=(0, 0, total_w, self.RULER_H))
        self._ruler = ruler

        # ── Label canvas ──────────────────────────────────────────────────────
        label_c = tk.Canvas(
            root,
            bg=_LABEL_BG,
            width=self.LABEL_W,
            highlightthickness=0,
            yscrollincrement=1,
        )
        label_c.grid(row=2, column=0, sticky="nsew")
        label_c.configure(scrollregion=(0, 0, self.LABEL_W, total_h))
        self._label_canvas = label_c

        # ── Timeline canvas ───────────────────────────────────────────────────
        tl = tk.Canvas(
            root,
            bg=_BG,
            highlightthickness=0,
            xscrollincrement=1,
            yscrollincrement=1,
        )
        tl.grid(row=2, column=1, sticky="nsew")
        tl.configure(scrollregion=(0, 0, total_w, total_h))
        self._tl = tl

        # ── Scrollbars ────────────────────────────────────────────────────────
        h_scroll = ttk.Scrollbar(root, orient="horizontal", command=self._hscroll_cmd)
        h_scroll.grid(row=3, column=1, sticky="ew")
        self._h_scroll = h_scroll

        v_scroll = ttk.Scrollbar(root, orient="vertical", command=self._vscroll_cmd)
        v_scroll.grid(row=1, column=2, rowspan=2, sticky="ns")
        self._v_scroll = v_scroll

        tl.configure(
            xscrollcommand=self._on_tl_xscroll,
            yscrollcommand=self._on_tl_yscroll,
        )

        # ── Grid weights ──────────────────────────────────────────────────────
        root.columnconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)

        # ── Draw ──────────────────────────────────────────────────────────────
        self._draw_ruler(total_w)
        self._draw_labels()
        self._draw_tracks()

        # ── Bindings ──────────────────────────────────────────────────────────
        tl.bind("<Button-1>", self._on_click)
        tl.bind("<Motion>", self._on_tl_motion)
        tl.bind("<Leave>", lambda _e: self._hide_tooltip())
        root.bind("<Escape>", lambda _e: self._stop_playback())

        # Arrow-key navigation (bound on root so focus doesn't matter)
        root.bind("<Left>", lambda _e: self._seek_by_beat(-1))
        root.bind("<Right>", lambda _e: self._seek_by_beat(1))
        root.bind("<Control-Left>", lambda _e: self._seek_by_bar(-1))
        root.bind("<Control-Right>", lambda _e: self._seek_by_bar(1))
        root.bind("<Shift-Left>", lambda _e: self._seek_to_boundary(-1))
        root.bind("<Shift-Right>", lambda _e: self._seek_to_boundary(1))

        for widget in (tl, label_c, ruler):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
            # Linux
            widget.bind(
                "<Button-4>", lambda e: self._vscroll_cmd("scroll", -1, "units")
            )
            widget.bind("<Button-5>", lambda e: self._vscroll_cmd("scroll", 1, "units"))

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Scroll plumbing ───────────────────────────────────────────────────────

    def _hscroll_cmd(self, *args: object) -> None:
        self._tl.xview(*args)
        self._ruler.xview(*args)

    def _vscroll_cmd(self, *args: object) -> None:
        self._tl.yview(*args)
        self._label_canvas.yview(*args)

    def _on_tl_xscroll(self, first: float, last: float) -> None:
        # Only update the scrollbar thumb.  The ruler is already kept in sync
        # by _hscroll_cmd; syncing it here too would double-scroll it.
        self._h_scroll.set(first, last)

    def _on_tl_yscroll(self, first: float, last: float) -> None:
        # Only update the scrollbar thumb.  The label canvas is already kept
        # in sync by _vscroll_cmd; syncing it here too would double-scroll it,
        # which is what caused the label/track misalignment.
        self._v_scroll.set(first, last)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 if event.delta > 0 else 1
        self._vscroll_cmd("scroll", delta, "units")

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        delta = -1 if event.delta > 0 else 1
        self._hscroll_cmd("scroll", delta, "units")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_ruler(self, total_w: int) -> None:
        c = self._ruler
        c.delete("all")

        total_dur = self.project.duration_seconds()
        bpm = self.project.bpm
        beat_s = 60.0 / bpm
        bar_s = beat_s * self.BEATS_PER_BAR
        mid = self.RULER_H // 2

        c.create_rectangle(0, 0, total_w, self.RULER_H, fill=_RULER_BG, outline="")
        # Bottom separator line
        c.create_line(
            0, self.RULER_H - 1, total_w, self.RULER_H - 1, fill="#2A2A45", width=1
        )

        beat_idx = 0
        t = 0.0
        while t <= total_dur + beat_s:
            x = int(t * self.PX_PER_SEC)
            is_bar = beat_idx % self.BEATS_PER_BAR == 0
            bar_num = beat_idx // self.BEATS_PER_BAR + 1
            beat_in_bar = beat_idx % self.BEATS_PER_BAR + 1

            if is_bar:
                # Full-height bar line — more prominent
                c.create_line(x, 0, x, self.RULER_H - 1, fill="#4A4A6F", width=1)
                # Bar number (top half)
                mm = int(t) // 60
                ss = t - mm * 60
                time_str = f"{mm}:{ss:04.1f}"
                c.create_text(
                    x + 4,
                    mid // 2,
                    text=f"Bar {bar_num}",
                    anchor="w",
                    fill=_TEXT,
                    font=("Helvetica", 8, "bold"),
                )
                c.create_text(
                    x + 4,
                    mid + mid // 2,
                    text=time_str,
                    anchor="w",
                    fill=_SUBTEXT,
                    font=("Helvetica", 7),
                )
            else:
                # Beat tick — lower half only
                c.create_line(x, mid, x, self.RULER_H - 1, fill="#2D2D48", width=1)
                c.create_text(
                    x + 2,
                    mid + (self.RULER_H - mid) // 2,
                    text=f"♩{beat_in_bar}",
                    anchor="w",
                    fill=_SUBTEXT,
                    font=("Helvetica", 7),
                )

            t = round(t + beat_s, 6)
            beat_idx += 1

    def _draw_labels(self) -> None:
        c = self._label_canvas
        c.delete("all")
        self._enabled.clear()

        for i, slot in enumerate(self.project._slots):
            y0 = i * self.ROW_H
            y1 = y0 + self.ROW_H
            bg = _TRACK_BG if i % 2 == 0 else _TRACK_ALT

            c.create_rectangle(0, y0, self.LABEL_W, y1, fill=bg, outline="")
            c.create_line(0, y1 - 1, self.LABEL_W, y1 - 1, fill=_SEP)

            # Colour swatch
            swatch = self._slot_swatch_color(slot)
            c.create_rectangle(6, y0 + 8, 12, y1 - 8, fill=swatch, outline="", width=0)

            # Track name + info
            arr_dur = slot.arrangement.duration_seconds(self.project.bpm)
            n_items = len(slot.arrangement.items)
            c.create_text(
                18,
                y0 + self.ROW_H // 2 - 7,
                text=f"Track {i + 1}",
                anchor="w",
                fill=_TEXT,
                font=("Helvetica", 9, "bold"),
            )
            clip_info = ""
            if slot.clip_start > 0 or slot.clip_end is not None:
                end_str = (
                    f"{slot.clip_end:.1f}"
                    if slot.clip_end is not None
                    else f"{arr_dur:.1f}"
                )
                clip_info = f"  clip {slot.clip_start:.1f}→{end_str}s"
            c.create_text(
                18,
                y0 + self.ROW_H // 2 + 7,
                text=f"+{slot.offset:.2f}s  ·  {n_items} items{clip_info}",
                anchor="w",
                fill=_SUBTEXT,
                font=("Helvetica", 8),
            )

            # Mute checkbox
            var = tk.BooleanVar(value=True)
            self._enabled.append(var)
            cb = ttk.Checkbutton(
                c, variable=var, command=lambda i=i: self._on_mute_toggle(i)
            )
            c.create_window(
                self.LABEL_W - 18, y0 + self.ROW_H // 2, window=cb, anchor="center"
            )

    def _draw_tracks(self) -> None:
        tl = self._tl
        tl.delete("all")
        self._spec_images.clear()
        self._block_entry_map.clear()

        bpm = self.project.bpm
        total_dur = max(self.project.duration_seconds(), 1.0)
        total_w = int(total_dur * self.PX_PER_SEC) + 200
        total_h = len(self.project._slots) * self.ROW_H

        # Vertical grid lines (every second)
        t = 0.0
        while t <= total_dur + 1.0:
            x = int(t * self.PX_PER_SEC)
            tl.create_line(
                x, 0, x, total_h, fill=_GRID, width=1, dash=(2, 8), tags="grid"
            )
            t = round(t + 1.0, 6)

        for i, slot in enumerate(self.project._slots):
            y0 = i * self.ROW_H
            y1 = y0 + self.ROW_H
            muted = not self._enabled[i].get() if self._enabled else False
            accent = _track_color(i, muted)

            # Row background
            bg = _TRACK_BG if i % 2 == 0 else _TRACK_ALT
            tl.create_rectangle(0, y0, total_w, y1, fill=bg, outline="")
            tl.create_line(0, y1 - 1, total_w, y1 - 1, fill=_SEP)

            arr_dur = slot.arrangement.duration_seconds(bpm)
            effective_end = slot.clip_end if slot.clip_end is not None else arr_dur
            clip_len = max(0.0, min(effective_end, arr_dur) - slot.clip_start)

            clip_x0 = int(slot.offset * self.PX_PER_SEC)
            clip_x1 = int((slot.offset + clip_len) * self.PX_PER_SEC)
            spec_w = max(clip_x1 - clip_x0, 1)
            spec_h = self.ROW_H - 2 * self.BLOCK_PAD

            if not muted and spec_w > 4 and spec_h > 4:
                # Draw spectrogram image behind everything
                img = self._compute_spec_image(slot, spec_w, spec_h)
                if img is not None:
                    self._spec_images.append(img)  # keep ref
                    tl.create_image(
                        clip_x0,
                        y0 + self.BLOCK_PAD,
                        image=img,
                        anchor="nw",
                        tags="spec",
                    )
            elif muted:
                tl.create_rectangle(
                    clip_x0 + self.BLOCK_PAD,
                    y0 + self.BLOCK_PAD,
                    clip_x1 - self.BLOCK_PAD,
                    y1 - self.BLOCK_PAD,
                    fill=_MUTED_FILL,
                    outline=_MUTED_OUTLINE,
                    width=1,
                )

            # Block outlines per Renderable — drawn on top of spectrogram.
            # Nested Arrangements are recursively expanded to their leaf
            # primitives (Tone, Sample, Silence) so each primitive is drawn
            # at its exact position rather than showing an opaque
            # "Arrangement" block.
            try:
                from dawpy.silence import Silence as _Silence

                flat = _flatten_timeline(slot.arrangement, bpm, 0.0)
            except (ValueError, ZeroDivisionError):
                continue

            for leaf_idx, (leaf_item, leaf_start, leaf_end, leaf_dur) in enumerate(
                flat
            ):
                if isinstance(leaf_item, _Silence):
                    continue  # invisible gap — space is preserved by layout math
                clip_s = max(leaf_start, slot.clip_start)
                clip_e = min(leaf_end, effective_end)
                if clip_s >= clip_e:
                    continue

                proj_s = slot.offset + (clip_s - slot.clip_start)
                proj_e = slot.offset + (clip_e - slot.clip_start)

                # No horizontal padding — adjacent notes should be flush
                x1 = int(proj_s * self.PX_PER_SEC)
                x2 = max(int(proj_e * self.PX_PER_SEC), x1 + 1)
                by1 = y0 + self.BLOCK_PAD  # vertical padding kept
                by2 = y1 - self.BLOCK_PAD

                block_tag = f"block_{i}_{leaf_idx}"
                # Store for tooltip lookup
                self._block_entry_map[block_tag] = (leaf_item, leaf_dur)
                tl.create_rectangle(
                    x1,
                    by1,
                    x2,
                    by2,
                    fill="",
                    outline=accent,
                    width=1,
                    tags=("block", block_tag),
                )

                w = x2 - x1
                mid_y = (by1 + by2) // 2
                text_col = _MUTED_TEXT if muted else _TEXT

                if w > 20:
                    label = _item_label(leaf_item)
                    tl.create_text(
                        x1 + 5,
                        mid_y - (6 if w > 50 else 0),
                        text=label,
                        anchor="w",
                        fill=text_col,
                        font=("Helvetica", 8, "bold"),
                        tags=("block_text", block_tag),
                    )
                if w > 50:
                    tl.create_text(
                        x1 + 5,
                        mid_y + 7,
                        text=f"{leaf_dur:.2f}s",
                        anchor="w",
                        fill=_SUBTEXT if not muted else _MUTED_TEXT,
                        font=("Helvetica", 7),
                        tags=("block_dur", block_tag),
                    )

        # Playhead (drawn last so it's on top)
        self._draw_playhead(self._playhead_t)

    def _draw_playhead(self, t: float) -> None:
        tl = self._tl
        tl.delete("playhead")
        total_h = len(self.project._slots) * self.ROW_H
        x = int(t * self.PX_PER_SEC)
        tl.create_line(x, 0, x, total_h, fill=_PLAYHEAD, width=2, tags="playhead")
        tl.tag_raise("playhead")

    # ── Interactions ──────────────────────────────────────────────────────────

    def _on_click(self, event: tk.Event) -> None:
        self._hide_tooltip()
        cx = self._tl.canvasx(event.x)
        t = max(0.0, cx / self.PX_PER_SEC)
        t = min(t, self.project.duration_seconds())
        self._start_playback(t)

    def _on_mute_toggle(self, _index: int) -> None:
        self._hide_tooltip()
        self._draw_tracks()

    def _on_close(self) -> None:
        self._hide_tooltip()
        self._stop_playback()
        self._root.destroy()

    def _on_tl_motion(self, event: tk.Event) -> None:
        """Show a tooltip when hovering over a block."""
        tl = self._tl
        cx = tl.canvasx(event.x)
        cy = tl.canvasy(event.y)
        # Find the topmost "block_*" tagged item under the cursor
        hit_tag: str | None = None
        for item_id in reversed(tl.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1)):
            for tag in tl.gettags(item_id):
                if tag.startswith("block_"):
                    hit_tag = tag
                    break
            if hit_tag:
                break

        if hit_tag == self._tooltip_tag:
            # Same block — just reposition the window if it exists
            if self._tooltip_win and self._tooltip_win.winfo_exists():
                self._tooltip_win.geometry(f"+{event.x_root + 14}+{event.y_root + 14}")
            return

        # Different block (or left a block) — cancel pending show and hide current
        if self._tooltip_after is not None:
            self._root.after_cancel(self._tooltip_after)
            self._tooltip_after = None
        self._hide_tooltip()
        self._tooltip_tag = hit_tag

        if hit_tag and hit_tag in self._block_entry_map:
            item, dur_s = self._block_entry_map[hit_tag]
            rx, ry = event.x_root + 14, event.y_root + 14

            def _show(rx: int = rx, ry: int = ry) -> None:
                self._show_tooltip(rx, ry, item, dur_s)

            self._tooltip_after = self._root.after(350, _show)  # type: ignore[assignment]

    def _show_tooltip(self, rx: int, ry: int, item: object, dur_s: float) -> None:
        """Create (or recreate) the tooltip Toplevel at screen position (rx, ry)."""
        self._hide_tooltip()
        bpm = self.project.bpm
        lines = _tooltip_lines(item, dur_s, bpm)

        win = tk.Toplevel(self._root)
        win.overrideredirect(True)
        win.configure(bg="#1A1A2E")
        win.attributes("-topmost", True)

        # Outer border frame
        border = tk.Frame(win, bg="#45475A", bd=1)
        border.pack(padx=1, pady=1)

        inner = tk.Frame(border, bg="#1A1A2E", padx=10, pady=8)
        inner.pack()

        for line in lines:
            if line == "":
                tk.Frame(inner, bg="#2D2D48", height=1).pack(fill="x", pady=3)
            else:
                is_heading = line.startswith(
                    ("Tone  --", "Sample  --", "Arrangement  --", "Silence  --", '  "')
                )
                tk.Label(
                    inner,
                    text=line,
                    bg="#1A1A2E",
                    fg=_TEXT if not is_heading else "#CBA6F7",
                    font=("Helvetica", 9, "bold" if is_heading else "normal"),
                    justify="left",
                    anchor="w",
                ).pack(anchor="w")

        win.geometry(f"+{rx}+{ry}")
        self._tooltip_win = win

    def _hide_tooltip(self) -> None:
        """Destroy the tooltip window and clear pending show callbacks."""
        if self._tooltip_after is not None:
            try:
                self._root.after_cancel(self._tooltip_after)
            except Exception:
                pass
            self._tooltip_after = None
        if self._tooltip_win is not None:
            try:
                if self._tooltip_win.winfo_exists():
                    self._tooltip_win.destroy()
            except Exception:
                pass
            self._tooltip_win = None
        self._tooltip_tag = None

    # ── Playback ──────────────────────────────────────────────────────────────

    def _start_playback(self, t_start: float) -> None:
        self._stop_playback()
        self._stop_event.clear()
        self._play_start_proj = t_start
        self._play_start_wall = time.perf_counter()
        self._playhead_t = t_start
        self._draw_playhead(t_start)
        self._update_timer(t_start)

        def _run() -> None:
            try:
                audio = self._render_from(t_start)
                if audio.shape[0] == 0:
                    return
                sd.play(audio, samplerate=self.project.sample_rate)
                while True:
                    try:
                        if not sd.get_stream().active:
                            break
                    except sd.PortAudioError:
                        break
                    if self._stop_event.is_set():
                        sd.stop()
                        break
                    time.sleep(0.02)
            except Exception:
                pass

        self._play_thread = threading.Thread(target=_run, daemon=True)
        self._play_thread.start()
        self._root.after(self.TICK_INTERVAL_MS, self._tick_playhead)

    def _stop_playback(self) -> None:
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)

    def _tick_playhead(self) -> None:
        """Called every TICK_INTERVAL_MS to advance the playhead."""
        if self._stop_event.is_set():
            return
        if self._play_thread and not self._play_thread.is_alive():
            return
        elapsed = time.perf_counter() - self._play_start_wall
        t = self._play_start_proj + elapsed
        self._playhead_t = t
        self._draw_playhead(t)
        self._scroll_to_playhead(t)
        self._update_timer(t)
        self._root.after(self.TICK_INTERVAL_MS, self._tick_playhead)

    def _scroll_to_playhead(self, t: float) -> None:
        """Auto-scroll so the playhead stays visible."""
        try:
            sr_str = str(self._tl.cget("scrollregion")).split()
            total_w = float(sr_str[2])
        except (IndexError, ValueError):
            return
        if total_w == 0:
            return
        frac = (t * self.PX_PER_SEC) / total_w
        lo, hi = self._tl.xview()
        span = hi - lo
        if frac > hi - span * 0.15:
            self._hscroll_cmd("moveto", max(0.0, frac - span * 0.25))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_from(self, t_start: float) -> np.ndarray:
        """Render the mix (enabled tracks only) and return audio from t_start.

        Returns:
            (num_samples, 2) float32 array.
        """
        bpm = self.project.bpm
        sr = self.project.sample_rate
        total_dur = self.project.duration_seconds()
        total_samples = int(total_dur * sr)
        if total_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        mix = np.zeros((total_samples, 2), dtype=np.float32)

        for i, slot in enumerate(self.project._slots):
            if self._enabled and not self._enabled[i].get():
                continue
            audio = slot.arrangement.render(bpm, sr).T.astype(np.float32)
            s = int(slot.clip_start * sr)
            e = int(slot.clip_end * sr) if slot.clip_end is not None else audio.shape[0]
            clip = audio[s:e]
            if clip.shape[0] == 0:
                continue
            off = int(slot.offset * sr)
            end = off + clip.shape[0]
            if end > total_samples:
                clip = clip[: total_samples - off]
                end = total_samples
            mix[off:end] += clip

        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak

        start_sample = int(t_start * sr)
        return mix[start_sample:]

    # ── Spectrogram ───────────────────────────────────────────────────────────

    def _compute_spec_image(
        self, slot: "_Slot", px_w: int, px_h: int
    ) -> "tk.PhotoImage | None":
        """Render the slot's clipped audio, compute an STFT spectrogram,
        and return a PhotoImage of size (px_w × px_h) ready for the canvas."""
        try:
            from scipy.signal import spectrogram as _sp, resample as _rs  # type: ignore
        except ImportError:
            return None

        try:
            bpm = self.project.bpm
            # Low-quality render is enough for visualisation
            sr_lo = 8000
            audio = slot.arrangement.render(bpm, sr_lo)
            mono = audio.mean(axis=0)

            s = int(slot.clip_start * sr_lo)
            e = int(slot.clip_end * sr_lo) if slot.clip_end is not None else len(mono)
            mono = mono[s:e]
            if len(mono) < 64:
                return None

            nperseg = min(256, len(mono) // 4)
            nperseg = max(nperseg, 16)
            _, _, Sxx = _sp(
                mono,
                fs=sr_lo,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                window="hann",
            )

            # Keep lower 60 % of frequencies (musical content)
            Sxx = Sxx[: max(1, int(Sxx.shape[0] * 0.6)), :]

            # Log-power, normalised to [0, 1]
            Sxx = 10.0 * np.log10(Sxx + 1e-10)
            lo, hi = Sxx.min(), Sxx.max()
            if hi == lo:
                Sxx_n = np.zeros_like(Sxx)
            else:
                Sxx_n = (Sxx - lo) / (hi - lo)

            # Resize to pixel dimensions via linear interpolation
            from scipy.ndimage import zoom as _zoom  # type: ignore

            h_in, w_in = Sxx_n.shape
            Sxx_n = _zoom(
                Sxx_n,
                (px_h / h_in, px_w / w_in),
                order=1,
            )
            Sxx_n = Sxx_n[::-1, :]  # flip: low freq at bottom

            rgb = _magma_rgb(np.asarray(Sxx_n, dtype=np.float32))
            return tk.PhotoImage(data=_ppm_b64(rgb))
        except Exception:
            return None

    # ── Keyboard navigation ───────────────────────────────────────────────────

    def _seek(self, t: float) -> None:
        """Move playhead to *t* (clamped to project duration).
        If playback is active, restart from *t*."""
        t = max(0.0, min(t, self.project.duration_seconds()))
        self._playhead_t = t
        self._draw_playhead(t)
        self._scroll_to_playhead(t)
        self._update_timer(t)
        if self._play_thread and self._play_thread.is_alive():
            self._start_playback(t)

    def _seek_by_beat(self, direction: int) -> None:
        """Advance/rewind the playhead by one beat."""
        beat_s = 60.0 / self.project.bpm
        self._seek(self._playhead_t + direction * beat_s)

    def _seek_by_bar(self, direction: int) -> None:
        """Advance/rewind the playhead by one bar (BEATS_PER_BAR beats)."""
        bar_s = 60.0 / self.project.bpm * self.BEATS_PER_BAR
        self._seek(self._playhead_t + direction * bar_s)

    def _seek_to_boundary(self, direction: int) -> None:
        """Snap playhead to the previous/next renderable start or end time."""
        bounds = self._all_boundaries()
        t = self._playhead_t
        if direction > 0:
            candidates = [b for b in bounds if b > t + 1e-6]
            self._seek(candidates[0] if candidates else t)
        else:
            candidates = [b for b in bounds if b < t - 1e-6]
            self._seek(candidates[-1] if candidates else t)

    def _all_boundaries(self) -> list[float]:
        """Return every renderable start/end time in project space, sorted."""
        bpm = self.project.bpm
        pts: set[float] = {0.0}
        for slot in self.project._slots:
            try:
                entries = slot.arrangement.timeline(bpm)
            except Exception:
                continue
            arr_dur = slot.arrangement.duration_seconds(bpm)
            eff_end = slot.clip_end if slot.clip_end is not None else arr_dur
            for entry in entries:
                cs = max(entry.start_seconds, slot.clip_start)
                ce = min(entry.end_seconds, eff_end)
                if cs >= ce:
                    continue
                pts.add(round(slot.offset + cs - slot.clip_start, 6))
                pts.add(round(slot.offset + ce - slot.clip_start, 6))
        return sorted(pts)

    def _update_timer(self, t: float) -> None:
        """Refresh the time / bar / beat display in the controls bar."""
        m = int(t) // 60
        s = t - m * 60
        beat_s = 60.0 / self.project.bpm
        beat_idx = int(t / beat_s)  # 0-based absolute beat
        bar_num = beat_idx // self.BEATS_PER_BAR + 1
        beat_in_bar = beat_idx % self.BEATS_PER_BAR + 1
        self._time_var.set(f"{m}:{s:05.2f}   Bar {bar_num}  ♩{beat_in_bar}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _slot_swatch_color(self, slot: "_Slot") -> str:
        """Track accent colour (index-based, not item-type-based)."""
        index = self.project._slots.index(slot)
        return _track_color(index)
