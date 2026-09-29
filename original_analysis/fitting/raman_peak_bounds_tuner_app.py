"""
Raman peak-bound tuner for cluster fitting.

This app is a read-only inspector for baseline-corrected Raman spectra. It scans
ALS_BASELINE_CORRECTED, averages every TXT file across all Y columns
(column 1 is X; columns 2..N are spectra), displays representative temperatures,
and lets you edit peak X/width bounds per family and optional temperature anchor.

The only file it writes is the JSON settings file when the user clicks Save.

Run:
    conda activate plotting
    python raman_peak_bounds_tuner_app.py
"""

from __future__ import annotations

import copy
import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter as tk

import numpy as np

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


APP_TITLE = "Raman Peak Bounds Tuner - AVERAGED TXT / JSON CONFIG"
SETTINGS_FILENAME = "raman_peak_bounds_settings.json"
SOURCE_DIRNAME = "ALS_BASELINE_CORRECTED"
ALL_SEQUENCES = "All UP/DOWN folders"
ANCHOR_PREVIEW_TEMPERATURE_COUNT = 3

TARGET_SAMPLE_FOLDERS = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "Aligned_RO_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
]

FAMILY_LABELS = {
    "Aligned_Au_3A": "Aligned Au 3A",
    "Aligned_Au_8A": "Aligned Au 8A",
    "Aligned_RO_8A": "Aligned RO 8A",
    "MIRA_Au_unaligned_8A": "Unaligned Au 8A",
    "MIRA_RO_unaligned_8A": "Unaligned RO 8A",
}

SAMPLE_COLORS = {
    "Aligned_Au_3A": "#feb715",
    "Aligned_Au_8A": "#f08228",
    "Aligned_RO_8A": "#00a8a8",
    "MIRA_Au_unaligned_8A": "#ae540b",
    "MIRA_RO_unaligned_8A": "#008686",
}

SEQUENCE_STYLES = {
    "Spikes_Removed": "-",
    "Spikes_Removed_DOWN_1": "-",
    "Spikes_Removed_UP_1": "--",
    "Spikes_Removed_UP_2": ":",
}

INTERPOLATED_PEAK_FIELDS = ("center", "x_min", "x_max", "width_min", "width_max", "width_guess")
PROPAGATION_WIDTH_MAX_INCREASE_PER_CM_SHIFT = 2.0
PROPAGATION_DIRECTION_TOLERANCE_CM = 0.05

TEMPERATURE_RE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*K", re.IGNORECASE)
SPLITTER_RE = re.compile(r"[\s,;]+")


@dataclass
class SpectrumRecord:
    family: str
    sequence: str
    temperature: float
    path: Path
    x: np.ndarray
    y_average: np.ndarray
    n_y_columns: int


def family_kind(family: str) -> str:
    return "RO" if "_RO_" in family else "Au"


def temperature_key(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def parse_temperature(path: Path) -> float | None:
    match = TEMPERATURE_RE.search(path.name)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _float_or_none(token: str) -> float | None:
    token = token.strip().replace("\ufeff", "")
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_numeric_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//")):
                continue
            numeric = [value for value in (_float_or_none(tok) for tok in SPLITTER_RE.split(stripped)) if value is not None]
            if len(numeric) >= 2:
                rows.append(numeric)
    if not rows:
        raise ValueError("No numeric rows with at least X and one Y column were found.")

    widths: dict[int, int] = {}
    for row in rows:
        widths[len(row)] = widths.get(len(row), 0) + 1
    width = max(widths, key=lambda n: (widths[n], n))
    usable = [row[:width] for row in rows if len(row) >= width]
    if len(usable) < 2:
        raise ValueError("Not enough numeric rows were found after parsing.")

    data = np.asarray(usable, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("The file must contain one X column and at least one Y column.")
    return data


def average_txt_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    data = read_numeric_table(path)
    x = data[:, 0].astype(float)
    y_columns = data[:, 1:].astype(float)
    y_columns[~np.isfinite(y_columns)] = np.nan
    y_average = np.nanmean(y_columns, axis=1)
    finite = np.isfinite(x) & np.isfinite(y_average)
    return x[finite].copy(), y_average[finite].copy(), y_columns.shape[1]


def make_peak(
    peak_id: str,
    label: str,
    center: float,
    x_min: float,
    x_max: float,
    width_min: float,
    width_max: float,
    width_guess: float,
    required: bool = True,
    enabled: bool = True,
) -> dict:
    return {
        "id": peak_id,
        "label": label,
        "enabled": bool(enabled),
        "required": bool(required),
        "center": float(center),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "width_min": float(width_min),
        "width_max": float(width_max),
        "width_guess": float(width_guess),
    }


def smart_default_peaks(family: str) -> dict[str, dict]:
    """Initial editable defaults. These are starting points, not final truth."""
    if family_kind(family) == "RO":
        peaks = [
            make_peak("RBLM", "RBLM", 311, 306, 315, 6, 34, 15, True),
            make_peak("CH_L1", "CH left 1", 1232, 1225, 1239, 10, 42, 24, True),
            make_peak("CH_L2", "CH left 2 / shoulder", 1247, 1238, 1256, 10, 44, 24, False),
            make_peak("CH_MID", "CH middle", 1297, 1288, 1308, 6, 30, 16, False),
            make_peak("D", "D peak", 1334, 1328, 1342, 8, 28, 15, True),
            make_peak("G", "G peak", 1596, 1590, 1602, 8, 30, 16, True),
        ]
    else:
        peaks = [
            make_peak("RBLM", "RBLM", 312, 304, 318, 6, 34, 16, True),
            make_peak("CH_L1", "CH left 1", 1246, 1237, 1251, 8, 36, 20, True),
            make_peak("CH_L2", "CH left 2", 1256, 1249, 1265, 8, 40, 22, True),
            make_peak("CH_MID", "CH middle", 1298, 1288, 1308, 6, 30, 16, False),
            make_peak("D", "D peak", 1340, 1332, 1348, 8, 28, 16, True),
            make_peak("G", "G peak", 1597, 1588, 1603, 8, 30, 15, True),
        ]
    return {peak["id"]: peak for peak in peaks}


def default_settings(root: Path) -> dict:
    return {
        "schema": "raman_peak_bounds_settings_v1",
        "created_by": APP_TITLE,
        "updated_at": None,
        "data_root": str(root),
        "source_dirname": SOURCE_DIRNAME,
        "temperature_step_k": 30.0,
        "interpolation": {
            "method": "linear_between_temperature_anchors_for_each_peak_field",
            "temperature_scope": "per_family_actual_measured_temperatures_no_global_grid",
            "outside_anchor_range": "nearest_anchor_else_family_default",
            "fields": list(INTERPOLATED_PEAK_FIELDS),
            "resolved_output": "families.<family>.resolved_temperature_bounds",
        },
        "plot": {
            "x_min": 250.0,
            "x_max": 1700.0,
            "normalize": True,
            "offset": 1.15,
        },
        "families": {},
    }


class PeakBoundsTunerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1550x920")
        self.minsize(1180, 820)

        self.app_dir = Path(__file__).resolve().parent
        self.settings_path = self.app_dir / SETTINGS_FILENAME
        self.settings = default_settings(self.app_dir)
        self.records: list[SpectrumRecord] = []
        self.last_click_x: float | None = None

        self.root_var = tk.StringVar(value=str(self.app_dir))
        self.family_var = tk.StringVar(value="")
        self.sequence_var = tk.StringVar(value=ALL_SEQUENCES)
        self.scope_var = tk.StringVar(value="family_default")
        self.temp_anchor_var = tk.StringVar(value="")
        self.step_var = tk.StringVar(value="30")
        self.x_min_var = tk.StringVar(value="250")
        self.x_max_var = tk.StringVar(value="1700")
        self.normalize_var = tk.BooleanVar(value=True)
        self.offset_var = tk.StringVar(value="1.15")
        self.show_points_var = tk.BooleanVar(value=True)
        self.hover_enabled_var = tk.BooleanVar(value=True)
        self.measure_mode_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Scan ALS_BASELINE_CORRECTED to begin.")
        self.click_var = tk.StringVar(value="Last click X: -")
        self.hover_info_var = tk.StringVar(value="Hover: move over a spectrum point")
        self.measure_status_var = tk.StringVar(value="Ruler: off")
        self.hover_points: list[dict] = []
        self.hover_annotation = None
        self.measure_xs: list[float] = []
        self.measure_artists = []

        self.peak_id_var = tk.StringVar(value="")
        self.peak_label_var = tk.StringVar(value="")
        self.peak_enabled_var = tk.BooleanVar(value=True)
        self.peak_required_var = tk.BooleanVar(value=True)
        self.peak_center_var = tk.StringVar(value="")
        self.peak_x_min_var = tk.StringVar(value="")
        self.peak_x_max_var = tk.StringVar(value="")
        self.peak_w_min_var = tk.StringVar(value="")
        self.peak_w_max_var = tk.StringVar(value="")
        self.peak_w_guess_var = tk.StringVar(value="")

        self._build_ui()
        self._load_settings_if_present(silent=True)
        self._scan_records(silent=True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=8)
        side.grid(row=0, column=0, sticky="ns")
        side.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        root_box = ttk.LabelFrame(side, text="Data source", padding=8)
        root_box.grid(row=0, column=0, sticky="ew")
        root_box.columnconfigure(1, weight=1)
        ttk.Label(root_box, text="Root").grid(row=0, column=0, sticky="w")
        ttk.Entry(root_box, textvariable=self.root_var, width=48).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(root_box, text="Browse", command=self._browse_root).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(root_box, text="Scan", command=lambda: self._scan_records(silent=False)).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(root_box, text="Load settings", command=lambda: self._load_settings_if_present(silent=False)).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0)
        )
        ttk.Button(root_box, text="Save JSON (all anchors)", command=self._save_settings).grid(
            row=1, column=2, sticky="ew", padx=(4, 0), pady=(6, 0)
        )

        view_box = ttk.LabelFrame(side, text="View", padding=8)
        view_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        view_box.columnconfigure(1, weight=1)
        ttk.Label(view_box, text="Family").grid(row=0, column=0, sticky="w")
        self.family_combo = ttk.Combobox(view_box, textvariable=self.family_var, values=[], state="readonly", width=28)
        self.family_combo.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.family_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_family_changed())

        ttk.Label(view_box, text="UP/DOWN").grid(row=1, column=0, sticky="w")
        self.sequence_combo = ttk.Combobox(view_box, textvariable=self.sequence_var, values=[ALL_SEQUENCES], state="readonly", width=28)
        self.sequence_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(3, 0))
        self.sequence_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw_plot())

        ttk.Label(view_box, text="Temp step K").grid(row=2, column=0, sticky="w")
        ttk.Entry(view_box, textvariable=self.step_var, width=10).grid(row=2, column=1, sticky="ew", pady=(3, 0))
        ttk.Button(view_box, text="Replot", command=self._draw_plot).grid(row=2, column=2, sticky="ew", padx=(4, 0), pady=(3, 0))

        ttk.Label(view_box, text="X min").grid(row=3, column=0, sticky="w")
        ttk.Entry(view_box, textvariable=self.x_min_var, width=10).grid(row=3, column=1, sticky="ew", pady=(3, 0))
        ttk.Label(view_box, text="X max").grid(row=4, column=0, sticky="w")
        ttk.Entry(view_box, textvariable=self.x_max_var, width=10).grid(row=4, column=1, sticky="ew", pady=(3, 0))

        ttk.Checkbutton(view_box, text="Normalize each averaged TXT", variable=self.normalize_var, command=self._draw_plot).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        ttk.Label(view_box, text="Offset").grid(row=6, column=0, sticky="w")
        ttk.Entry(view_box, textvariable=self.offset_var, width=10).grid(row=6, column=1, sticky="ew", pady=(3, 0))
        ttk.Checkbutton(view_box, text="Show data points", variable=self.show_points_var, command=self._draw_plot).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

        scope_box = ttk.LabelFrame(side, text="Edit scope", padding=8)
        scope_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        scope_box.columnconfigure(1, weight=1)
        ttk.Radiobutton(scope_box, text="Family default", variable=self.scope_var, value="family_default", command=self._refresh_peak_table).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Radiobutton(
            scope_box,
            text=f"Temperature anchor ({ANCHOR_PREVIEW_TEMPERATURE_COUNT} closest)",
            variable=self.scope_var,
            value="temperature_anchor",
            command=self._refresh_peak_table,
        ).grid(
            row=1, column=0, sticky="w"
        )
        self.temp_combo = ttk.Combobox(scope_box, textvariable=self.temp_anchor_var, values=[], state="readonly", width=12)
        self.temp_combo.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        self.temp_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_peak_table())
        ttk.Button(scope_box, text="Copy defaults", command=self._copy_defaults_to_anchor).grid(
            row=2, column=0, sticky="ew", padx=(0, 3), pady=(6, 0)
        )
        ttk.Button(scope_box, text="Save JSON", command=self._save_settings).grid(
            row=2, column=1, sticky="ew", padx=(3, 0), pady=(6, 0)
        )
        ttk.Button(scope_box, text="Track selected", command=lambda: self._auto_track_peak_centers(selected_only=True)).grid(
            row=3, column=0, sticky="ew", padx=(0, 3), pady=(4, 0)
        )
        ttk.Button(scope_box, text="Track all", command=lambda: self._auto_track_peak_centers(selected_only=False)).grid(
            row=3, column=1, sticky="ew", padx=(3, 0), pady=(4, 0)
        )
        ttk.Button(scope_box, text="Propagate locked", command=self._propagate_current_anchor_by_shift).grid(
            row=4, column=0, sticky="ew", padx=(0, 3), pady=(4, 0)
        )
        ttk.Button(scope_box, text="Copy to temps...", command=self._copy_current_anchor_to_temperatures).grid(
            row=4, column=1, sticky="ew", padx=(3, 0), pady=(4, 0)
        )
        ttk.Button(scope_box, text="Reset family smart defaults", command=self._reset_family_defaults).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Label(scope_box, text="Store/edit peaks, then save JSON.", foreground="#444").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        peak_box = ttk.LabelFrame(side, text="Peaks", padding=8)
        peak_box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        side.rowconfigure(3, weight=3)
        peak_box.columnconfigure(0, weight=1)
        peak_box.rowconfigure(0, weight=3)

        columns = ("on", "req", "id", "center", "xmin", "xmax", "wmin", "wmax", "w0")
        self.peak_tree = ttk.Treeview(peak_box, columns=columns, show="headings", height=14)
        for col, title, width in [
            ("on", "On", 36),
            ("req", "Req", 38),
            ("id", "Peak", 88),
            ("center", "Ctr", 58),
            ("xmin", "Xmin", 58),
            ("xmax", "Xmax", 58),
            ("wmin", "Wmin", 54),
            ("wmax", "Wmax", 54),
            ("w0", "W0", 48),
        ]:
            self.peak_tree.heading(col, text=title)
            self.peak_tree.column(col, width=width, stretch=False, anchor="center")
        self.peak_tree.grid(row=0, column=0, sticky="nsew")
        self.peak_tree.bind("<<TreeviewSelect>>", lambda _event: self._load_selected_peak_into_form())
        peak_scroll = ttk.Scrollbar(peak_box, orient="vertical", command=self.peak_tree.yview)
        peak_scroll.grid(row=0, column=1, sticky="ns")
        self.peak_tree.configure(yscrollcommand=peak_scroll.set)

        form = ttk.Frame(peak_box)
        form.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for idx in range(4):
            form.columnconfigure(idx, weight=1)

        ttk.Label(form, text="ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.peak_id_var, width=12).grid(row=0, column=1, sticky="ew")
        ttk.Label(form, text="Label").grid(row=0, column=2, sticky="w")
        ttk.Entry(form, textvariable=self.peak_label_var, width=18).grid(row=0, column=3, sticky="ew")

        ttk.Checkbutton(form, text="Enabled", variable=self.peak_enabled_var).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Checkbutton(form, text="Required", variable=self.peak_required_var).grid(row=1, column=1, sticky="w", pady=(2, 0))

        for row, label, var1, label2, var2 in [
            (2, "Center", self.peak_center_var, "X min", self.peak_x_min_var),
            (3, "X max", self.peak_x_max_var, "W min", self.peak_w_min_var),
            (4, "W max", self.peak_w_max_var, "W guess", self.peak_w_guess_var),
        ]:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=(2, 0))
            ttk.Entry(form, textvariable=var1, width=10).grid(row=row, column=1, sticky="ew", pady=(2, 0))
            ttk.Label(form, text=label2).grid(row=row, column=2, sticky="w", pady=(2, 0))
            ttk.Entry(form, textvariable=var2, width=10).grid(row=row, column=3, sticky="ew", pady=(2, 0))

        ttk.Button(form, text="Store peak in current scope", command=self._apply_peak_form).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(form, text="Disable/delete selected", command=self._delete_or_disable_peak).grid(
            row=5, column=2, columnspan=2, sticky="ew", padx=(6, 0), pady=(6, 0)
        )

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky="ew")

        click_box = ttk.LabelFrame(plot_frame, text="Cursor / ruler", padding=(6, 4))
        click_box.grid(row=2, column=0, sticky="ew", padx=6, pady=(2, 0))
        for idx in range(9):
            click_box.columnconfigure(idx, weight=1 if idx in {0, 4, 8} else 0)
        ttk.Label(click_box, textvariable=self.click_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(click_box, text="Use X min", command=lambda: self._use_click("x_min")).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        ttk.Button(click_box, text="Use center", command=lambda: self._use_click("center")).grid(row=0, column=3, sticky="ew", padx=(4, 0))
        ttk.Button(click_box, text="Use X max", command=lambda: self._use_click("x_max")).grid(row=0, column=4, sticky="w", padx=(4, 10))
        ttk.Checkbutton(
            click_box,
            text="Hover readout",
            variable=self.hover_enabled_var,
            command=self._on_hover_toggle,
        ).grid(row=0, column=5, sticky="w", padx=(0, 8))
        ttk.Checkbutton(
            click_box,
            text="Measure width, two clicks",
            variable=self.measure_mode_var,
            command=self._on_measure_toggle,
        ).grid(row=0, column=6, sticky="w", padx=(0, 8))
        ttk.Button(click_box, text="Clear ruler", command=self._clear_measurement).grid(row=0, column=7, sticky="ew")
        ttk.Label(click_box, textvariable=self.hover_info_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=(3, 0))
        ttk.Label(click_box, textvariable=self.measure_status_var).grid(row=1, column=4, columnspan=4, sticky="w", pady=(3, 0))
        ttk.Label(plot_frame, textvariable=self.status_var, foreground="#444").grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 4))

    def _browse_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Select analysis root folder",
            initialdir=self.root_var.get() if Path(self.root_var.get()).exists() else str(self.app_dir),
        )
        if selected:
            self.root_var.set(selected)
            self._scan_records(silent=False)

    def _settings_source_dir(self) -> Path:
        root = Path(self.root_var.get()).expanduser()
        source_dirname = self.settings.get("source_dirname", SOURCE_DIRNAME)
        if root.name == source_dirname:
            return root
        return root / source_dirname

    def _load_settings_if_present(self, silent: bool) -> None:
        if not self.settings_path.exists():
            if not silent:
                messagebox.showinfo(APP_TITLE, f"No settings file yet:\n{self.settings_path}")
            return
        try:
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if loaded.get("schema") != "raman_peak_bounds_settings_v1":
                raise ValueError("Settings file has an unexpected schema.")
            self.settings = loaded
            self.root_var.set(str(loaded.get("data_root", self.app_dir)))
            self.step_var.set(str(loaded.get("temperature_step_k", 30)))
            plot = loaded.get("plot", {})
            self.x_min_var.set(str(plot.get("x_min", 250)))
            self.x_max_var.set(str(plot.get("x_max", 1700)))
            self.normalize_var.set(bool(plot.get("normalize", True)))
            self.offset_var.set(str(plot.get("offset", 1.15)))
            self.show_points_var.set(bool(plot.get("show_points", True)))
            ui_state = loaded.get("ui_state", {})
            if ui_state.get("family"):
                self.family_var.set(str(ui_state["family"]))
            if ui_state.get("sequence"):
                self.sequence_var.set(str(ui_state["sequence"]))
            if ui_state.get("scope"):
                self.scope_var.set(str(ui_state["scope"]))
            if ui_state.get("temperature_anchor"):
                self.temp_anchor_var.set(str(ui_state["temperature_anchor"]))
            if not silent:
                self.status_var.set(f"Loaded settings: {self.settings_path}")
        except Exception as exc:
            messagebox.showerror("Load settings failed", f"{exc}\n\n{traceback.format_exc()}")

    def _save_settings(self) -> None:
        self.settings["data_root"] = self.root_var.get()
        self.settings["source_dirname"] = SOURCE_DIRNAME
        self.settings["temperature_step_k"] = self._float_value(self.step_var.get(), 30.0)
        self.settings["interpolation"] = {
            "method": "linear_between_temperature_anchors_for_each_peak_field",
            "temperature_scope": "per_family_actual_measured_temperatures_no_global_grid",
            "outside_anchor_range": "nearest_anchor_else_family_default",
            "fields": list(INTERPOLATED_PEAK_FIELDS),
            "resolved_output": "families.<family>.resolved_temperature_bounds",
        }
        self.settings["plot"] = {
            "x_min": self._float_value(self.x_min_var.get(), 250.0),
            "x_max": self._float_value(self.x_max_var.get(), 1700.0),
            "normalize": bool(self.normalize_var.get()),
            "offset": self._float_value(self.offset_var.get(), 1.15),
            "show_points": bool(self.show_points_var.get()),
        }
        self.settings["ui_state"] = {
            "family": self.family_var.get(),
            "sequence": self.sequence_var.get(),
            "scope": self.scope_var.get(),
            "temperature_anchor": self.temp_anchor_var.get(),
        }
        resolved_count = self._update_resolved_temperature_bounds()
        self.settings["updated_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            self.status_var.set(f"Saved JSON settings with resolved bounds for {resolved_count} family-temperature entries: {self.settings_path}")
        except Exception as exc:
            messagebox.showerror("Save settings failed", f"{exc}\n\n{traceback.format_exc()}")

    def _scan_records(self, silent: bool) -> None:
        source_dir = self._settings_source_dir()
        self.records.clear()
        failures: list[str] = []
        if not source_dir.exists():
            if not silent:
                messagebox.showwarning(APP_TITLE, f"Source folder not found:\n{source_dir}")
            self._draw_plot()
            return

        for family in TARGET_SAMPLE_FOLDERS:
            family_dir = source_dir / family
            if not family_dir.exists():
                continue
            self._ensure_family_settings(family)
            for sequence_dir in sorted(family_dir.iterdir(), key=lambda p: p.name.lower()):
                if not sequence_dir.is_dir() or sequence_dir.name == "ALS_BASELINES":
                    continue
                for path in sorted(sequence_dir.glob("*.txt"), key=lambda p: p.name.lower()):
                    temp = parse_temperature(path)
                    if temp is None:
                        continue
                    try:
                        x, y_average, n_y = average_txt_spectrum(path)
                    except Exception as exc:
                        failures.append(f"{path.name}: {exc}")
                        continue
                    self.records.append(
                        SpectrumRecord(
                            family=family,
                            sequence=sequence_dir.name,
                            temperature=temp,
                            path=path,
                            x=x,
                            y_average=y_average,
                            n_y_columns=n_y,
                        )
                    )

        self._refresh_family_combo()
        if self.records:
            if not self.family_var.get():
                self.family_var.set(self._families_with_records()[0])
            self._on_family_changed()
        else:
            self._draw_plot()

        msg = f"Scanned {len(self.records)} averaged TXT spectra from {source_dir}."
        if failures:
            msg += f" {len(failures)} file(s) skipped."
        self.status_var.set(msg)
        if failures and not silent:
            messagebox.showwarning("Some files were skipped", "\n".join(failures[:20]))

    def _families_with_records(self) -> list[str]:
        families = []
        for family in TARGET_SAMPLE_FOLDERS:
            if any(record.family == family for record in self.records):
                families.append(family)
        return families

    def _refresh_family_combo(self) -> None:
        families = self._families_with_records()
        self.family_combo.configure(values=families)
        if families and self.family_var.get() not in families:
            self.family_var.set(families[0])

    def _on_family_changed(self) -> None:
        family = self.family_var.get()
        if not family:
            return
        self._ensure_family_settings(family)
        sequences = sorted({record.sequence for record in self.records if record.family == family})
        values = [ALL_SEQUENCES] + sequences
        self.sequence_combo.configure(values=values)
        if self.sequence_var.get() not in values:
            self.sequence_var.set(ALL_SEQUENCES)
        self._refresh_temp_anchor_combo()
        self._refresh_peak_table()
        self._draw_plot()

    def _refresh_temp_anchor_combo(self) -> None:
        family = self.family_var.get()
        temps = {record.temperature for record in self.records if record.family == family}
        if family:
            self._ensure_family_settings(family)
            for key, anchor in self.settings["families"][family].get("temperature_anchors", {}).items():
                try:
                    temps.add(float(anchor.get("temperature_k", key)))
                except (TypeError, ValueError):
                    continue
        values = [temperature_key(temp) for temp in sorted(temps)]
        self.temp_combo.configure(values=values)
        if values and self.temp_anchor_var.get() not in values:
            self.temp_anchor_var.set(values[0])
        elif not values:
            self.temp_anchor_var.set("")

    def _ensure_family_settings(self, family: str) -> None:
        families = self.settings.setdefault("families", {})
        if family not in families:
            families[family] = {
                "label": FAMILY_LABELS.get(family, family),
                "kind": family_kind(family),
                "peaks": smart_default_peaks(family),
                "temperature_anchors": {},
            }
        else:
            fam_settings = families[family]
            fam_settings.setdefault("label", FAMILY_LABELS.get(family, family))
            fam_settings.setdefault("kind", family_kind(family))
            fam_settings.setdefault("peaks", smart_default_peaks(family))
            fam_settings.setdefault("temperature_anchors", {})

    def _current_temp_key(self) -> str | None:
        key = self.temp_anchor_var.get().strip()
        return key or None

    def _current_anchor_temperature(self) -> float | None:
        key = self._current_temp_key()
        if not key:
            return None
        try:
            temperature = float(key)
        except ValueError:
            return None
        return temperature if math.isfinite(temperature) else None

    def _family_settings(self) -> dict | None:
        family = self.family_var.get()
        if not family:
            return None
        self._ensure_family_settings(family)
        return self.settings["families"][family]

    def _effective_peaks(self, family: str | None = None, temp_key: str | None = None) -> dict[str, dict]:
        family = family or self.family_var.get()
        if not family:
            return {}
        self._ensure_family_settings(family)
        fam_settings = self.settings["families"][family]
        peaks = copy.deepcopy(fam_settings.get("peaks", {}))
        if temp_key:
            overrides = fam_settings.get("temperature_anchors", {}).get(temp_key, {}).get("peaks", {})
            for peak_id, peak in overrides.items():
                peaks[peak_id] = {**peaks.get(peak_id, {}), **copy.deepcopy(peak)}
        return peaks

    def _temperatures_for_family(self, family: str) -> list[float]:
        return sorted({record.temperature for record in self.records if record.family == family})

    def _anchor_points_for_peak_field(self, fam_settings: dict, peak_id: str, field: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for key, anchor in fam_settings.get("temperature_anchors", {}).items():
            try:
                temp = float(anchor.get("temperature_k", key))
            except (TypeError, ValueError):
                continue
            peak = anchor.get("peaks", {}).get(peak_id)
            if not peak or field not in peak:
                continue
            try:
                value = float(peak[field])
            except (TypeError, ValueError):
                continue
            if math.isfinite(temp) and math.isfinite(value):
                points.append((temp, value))
        return sorted(points, key=lambda item: item[0])

    def _nearest_anchor_peak(self, fam_settings: dict, peak_id: str, temperature: float) -> dict | None:
        nearest: tuple[float, dict] | None = None
        for key, anchor in fam_settings.get("temperature_anchors", {}).items():
            try:
                anchor_temp = float(anchor.get("temperature_k", key))
            except (TypeError, ValueError):
                continue
            peak = anchor.get("peaks", {}).get(peak_id)
            if not peak:
                continue
            distance = abs(anchor_temp - temperature)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, peak)
        return copy.deepcopy(nearest[1]) if nearest else None

    def _interpolated_peak_value(self, base_peak: dict, anchor_points: list[tuple[float, float]], field: str, temperature: float) -> float | None:
        if not anchor_points:
            try:
                value = float(base_peak[field])
            except (KeyError, TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        if len(anchor_points) == 1 or temperature <= anchor_points[0][0]:
            return anchor_points[0][1]
        if temperature >= anchor_points[-1][0]:
            return anchor_points[-1][1]

        for (t0, v0), (t1, v1) in zip(anchor_points, anchor_points[1:]):
            if abs(temperature - t0) < 1e-9:
                return v0
            if abs(temperature - t1) < 1e-9:
                return v1
            if t0 <= temperature <= t1:
                if abs(t1 - t0) < 1e-12:
                    return v0
                fraction = (temperature - t0) / (t1 - t0)
                return v0 + fraction * (v1 - v0)
        return anchor_points[-1][1]

    def _resolved_peaks_for_temperature(self, family: str, temperature: float) -> dict[str, dict]:
        self._ensure_family_settings(family)
        fam_settings = self.settings["families"][family]
        default_peaks = fam_settings.get("peaks", {})

        peak_ids = set(default_peaks)
        for anchor in fam_settings.get("temperature_anchors", {}).values():
            peak_ids.update(anchor.get("peaks", {}).keys())

        resolved: dict[str, dict] = {}
        for peak_id in sorted(peak_ids):
            base_peak = copy.deepcopy(default_peaks.get(peak_id, {}))
            nearest_anchor_peak = self._nearest_anchor_peak(fam_settings, peak_id, temperature)
            if nearest_anchor_peak and not base_peak:
                base_peak = nearest_anchor_peak
            elif nearest_anchor_peak:
                for field in ("id", "label", "enabled", "required"):
                    if field in nearest_anchor_peak:
                        base_peak[field] = nearest_anchor_peak[field]

            if not base_peak:
                continue
            peak = copy.deepcopy(base_peak)
            peak["id"] = str(peak.get("id", peak_id))
            peak["label"] = str(peak.get("label", peak_id))
            peak["enabled"] = bool(peak.get("enabled", True))
            peak["required"] = bool(peak.get("required", True))

            for field in INTERPOLATED_PEAK_FIELDS:
                value = self._interpolated_peak_value(base_peak, self._anchor_points_for_peak_field(fam_settings, peak_id, field), field, temperature)
                if value is not None:
                    peak[field] = float(value)

            if "x_min" in peak and "x_max" in peak and float(peak["x_min"]) >= float(peak["x_max"]):
                center = float(peak.get("center", peak["x_min"]))
                peak["x_min"] = center - 0.5
                peak["x_max"] = center + 0.5
            if "width_min" in peak and "width_max" in peak and float(peak["width_min"]) >= float(peak["width_max"]):
                w0 = max(float(peak.get("width_guess", peak["width_min"])), 1.0)
                peak["width_min"] = max(w0 * 0.5, 0.1)
                peak["width_max"] = max(w0 * 1.5, peak["width_min"] + 0.1)

            resolved[peak_id] = peak
        return resolved

    def _update_resolved_temperature_bounds(self) -> int:
        total_temperatures = 0
        for family in TARGET_SAMPLE_FOLDERS:
            if not any(record.family == family for record in self.records):
                continue
            self._ensure_family_settings(family)
            fam_settings = self.settings["families"][family]
            resolved_for_family: dict[str, dict] = {}
            for temperature in self._temperatures_for_family(family):
                key = temperature_key(temperature)
                resolved_for_family[key] = {
                    "temperature_k": float(temperature),
                    "peaks": self._resolved_peaks_for_temperature(family, temperature),
                }
                total_temperatures += 1
            fam_settings["resolved_temperature_bounds"] = resolved_for_family
        return total_temperatures

    def _refresh_peak_table(self) -> None:
        for item in self.peak_tree.get_children():
            self.peak_tree.delete(item)
        family = self.family_var.get()
        if not family:
            return
        temp_key = self._current_temp_key() if self.scope_var.get() == "temperature_anchor" else None
        peaks = self._effective_peaks(family, temp_key)
        for peak_id, peak in sorted(peaks.items(), key=lambda item: (float(item[1].get("center", 0)), item[0])):
            self.peak_tree.insert(
                "",
                "end",
                iid=peak_id,
                values=(
                    "Y" if peak.get("enabled", True) else "N",
                    "Y" if peak.get("required", True) else "N",
                    peak_id,
                    self._fmt(peak.get("center")),
                    self._fmt(peak.get("x_min")),
                    self._fmt(peak.get("x_max")),
                    self._fmt(peak.get("width_min")),
                    self._fmt(peak.get("width_max")),
                    self._fmt(peak.get("width_guess")),
                ),
            )
        self._draw_plot()

    def _load_selected_peak_into_form(self) -> None:
        selected = self.peak_tree.selection()
        if not selected:
            return
        peak_id = selected[0]
        temp_key = self._current_temp_key() if self.scope_var.get() == "temperature_anchor" else None
        peak = self._effective_peaks(temp_key=temp_key).get(peak_id)
        if not peak:
            return
        self.peak_id_var.set(str(peak.get("id", peak_id)))
        self.peak_label_var.set(str(peak.get("label", peak_id)))
        self.peak_enabled_var.set(bool(peak.get("enabled", True)))
        self.peak_required_var.set(bool(peak.get("required", True)))
        self.peak_center_var.set(self._fmt(peak.get("center")))
        self.peak_x_min_var.set(self._fmt(peak.get("x_min")))
        self.peak_x_max_var.set(self._fmt(peak.get("x_max")))
        self.peak_w_min_var.set(self._fmt(peak.get("width_min")))
        self.peak_w_max_var.set(self._fmt(peak.get("width_max")))
        self.peak_w_guess_var.set(self._fmt(peak.get("width_guess")))

    def _apply_peak_form(self) -> None:
        family = self.family_var.get()
        fam_settings = self._family_settings()
        if not family or fam_settings is None:
            return
        try:
            peak_id = self.peak_id_var.get().strip()
            if not peak_id:
                raise ValueError("Peak ID cannot be empty.")
            peak = make_peak(
                peak_id=peak_id,
                label=self.peak_label_var.get().strip() or peak_id,
                center=self._strict_float(self.peak_center_var.get(), "center"),
                x_min=self._strict_float(self.peak_x_min_var.get(), "x min"),
                x_max=self._strict_float(self.peak_x_max_var.get(), "x max"),
                width_min=self._strict_float(self.peak_w_min_var.get(), "width min"),
                width_max=self._strict_float(self.peak_w_max_var.get(), "width max"),
                width_guess=self._strict_float(self.peak_w_guess_var.get(), "width guess"),
                required=self.peak_required_var.get(),
                enabled=self.peak_enabled_var.get(),
            )
            if peak["x_min"] >= peak["x_max"]:
                raise ValueError("X min must be smaller than X max.")
            if peak["width_min"] <= 0 or peak["width_min"] >= peak["width_max"]:
                raise ValueError("Width min must be > 0 and smaller than width max.")
            if not (peak["x_min"] <= peak["center"] <= peak["x_max"]):
                raise ValueError("Center must sit inside the X bounds.")
            if not (peak["width_min"] <= peak["width_guess"] <= peak["width_max"]):
                raise ValueError("Width guess must sit inside the width bounds.")
        except Exception as exc:
            messagebox.showerror("Peak settings invalid", str(exc))
            return

        if self.scope_var.get() == "temperature_anchor":
            temp_key = self._current_temp_key()
            if not temp_key:
                messagebox.showinfo(APP_TITLE, "Select a temperature anchor first.")
                return
            anchors = fam_settings.setdefault("temperature_anchors", {})
            anchor = anchors.setdefault(temp_key, {"temperature_k": float(temp_key), "peaks": {}})
            anchor.setdefault("peaks", {})[peak_id] = peak
            msg = f"Updated {peak_id} for {family} at {temp_key} K anchor."
        else:
            fam_settings.setdefault("peaks", {})[peak_id] = peak
            msg = f"Updated {peak_id} family default for {family}."

        self.status_var.set(msg)
        self._refresh_peak_table()

    def _delete_or_disable_peak(self) -> None:
        selected = self.peak_tree.selection()
        if not selected:
            return
        peak_id = selected[0]
        fam_settings = self._family_settings()
        if fam_settings is None:
            return
        if self.scope_var.get() == "temperature_anchor":
            temp_key = self._current_temp_key()
            if not temp_key:
                return
            peak = self._effective_peaks(temp_key=temp_key).get(peak_id)
            if peak:
                peak["enabled"] = False
                anchors = fam_settings.setdefault("temperature_anchors", {})
                anchor = anchors.setdefault(temp_key, {"temperature_k": float(temp_key), "peaks": {}})
                anchor.setdefault("peaks", {})[peak_id] = peak
                self.status_var.set(f"Disabled {peak_id} at {temp_key} K anchor.")
        else:
            fam_settings.setdefault("peaks", {}).pop(peak_id, None)
            for anchor in fam_settings.setdefault("temperature_anchors", {}).values():
                anchor.setdefault("peaks", {}).pop(peak_id, None)
            self.status_var.set(f"Deleted {peak_id} from family defaults and anchors.")
        self._refresh_peak_table()

    def _copy_defaults_to_anchor(self) -> None:
        fam_settings = self._family_settings()
        temp_key = self._current_temp_key()
        if fam_settings is None or not temp_key:
            return
        anchor = fam_settings.setdefault("temperature_anchors", {}).setdefault(temp_key, {"temperature_k": float(temp_key), "peaks": {}})
        anchor["peaks"] = copy.deepcopy(fam_settings.get("peaks", {}))
        self.scope_var.set("temperature_anchor")
        self.status_var.set(f"Copied family defaults to {temp_key} K anchor.")
        self._refresh_peak_table()

    def _parse_temperature_targets(self, text: str, measured_temps: list[float]) -> list[float]:
        targets: set[float] = set()
        measured_sorted = sorted(measured_temps)
        tokens = [token.strip() for token in re.split(r"[,;]+", text) if token.strip()]
        for token in tokens:
            range_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*-\s*([-+]?\d+(?:\.\d+)?)(?::\s*([-+]?\d+(?:\.\d+)?))?", token)
            if range_match:
                start = float(range_match.group(1))
                end = float(range_match.group(2))
                if start > end:
                    start, end = end, start
                if range_match.group(3):
                    step = abs(float(range_match.group(3)))
                    if step <= 0:
                        raise ValueError(f"Range step must be positive in '{token}'.")
                    value = start
                    while value <= end + step * 1e-6:
                        targets.add(round(value, 6))
                        value += step
                else:
                    for temp in measured_sorted:
                        if start <= temp <= end:
                            targets.add(round(temp, 6))
                continue
            try:
                targets.add(round(float(token), 6))
            except ValueError as exc:
                raise ValueError(f"Could not read temperature target '{token}'.") from exc
        return sorted(targets)

    def _copy_current_anchor_to_temperatures(self) -> None:
        family = self.family_var.get()
        fam_settings = self._family_settings()
        source_key = self._current_temp_key()
        if not family or fam_settings is None or not source_key:
            messagebox.showinfo(APP_TITLE, "Select a family and source temperature anchor first.")
            return
        try:
            source_temp = float(source_key)
        except ValueError:
            messagebox.showerror(APP_TITLE, f"Current anchor temperature is invalid: {source_key}")
            return

        measured_temps = self._temperatures_for_family(family)
        default_example = f"{source_key}-" + (temperature_key(measured_temps[-1]) if measured_temps else source_key)
        target_text = simpledialog.askstring(
            "Copy current anchor",
            "Copy the current anchor bounds to which temperatures?\n\n"
            "Examples:\n"
            "  70, 100, 130\n"
            "  70-160        uses measured temperatures in that range\n"
            "  70-160:30     creates anchors every 30 K\n\n"
            f"Source anchor: {source_key} K",
            initialvalue=default_example,
            parent=self,
        )
        if not target_text:
            return
        try:
            targets = self._parse_temperature_targets(target_text, measured_temps)
        except ValueError as exc:
            messagebox.showerror("Temperature targets invalid", str(exc))
            return
        if not targets:
            messagebox.showinfo(APP_TITLE, "No target temperatures were selected.")
            return

        source_peaks = self._effective_peaks(family=family, temp_key=source_key)
        anchors = fam_settings.setdefault("temperature_anchors", {})
        copied_keys: list[str] = []
        for target in targets:
            if abs(target - source_temp) < 1e-6:
                continue
            key = temperature_key(target)
            anchors[key] = {"temperature_k": float(target), "peaks": copy.deepcopy(source_peaks)}
            copied_keys.append(key)

        self._refresh_temp_anchor_combo()
        self.status_var.set(
            f"Copied {source_key} K anchor to {len(copied_keys)} temperature(s): {', '.join(copied_keys[:12])}"
            + ("..." if len(copied_keys) > 12 else "")
            + ". Click Save JSON now to write the file."
        )
        self._refresh_peak_table()

    def _records_for_tracking(self, family: str) -> list[SpectrumRecord]:
        records = [record for record in self.records if record.family == family]
        sequence = self.sequence_var.get()
        if sequence and sequence != ALL_SEQUENCES:
            records = [record for record in records if record.sequence == sequence]
        return records

    def _smooth_for_tracking(self, y_values: np.ndarray) -> np.ndarray:
        if len(y_values) < 7:
            return y_values.astype(float, copy=True)
        window = 5 if len(y_values) < 80 else 7
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(y_values, kernel, mode="same")

    def _quadratic_peak_center(self, x_values: np.ndarray, y_values: np.ndarray, index: int) -> float:
        if index <= 0 or index >= len(x_values) - 1:
            return float(x_values[index])
        local_x = x_values[index - 1 : index + 2]
        local_y = y_values[index - 1 : index + 2]
        if not (np.all(np.isfinite(local_x)) and np.all(np.isfinite(local_y))):
            return float(x_values[index])
        try:
            a, b, _c = np.polyfit(local_x, local_y, 2)
        except Exception:
            return float(x_values[index])
        if abs(a) < 1e-12 or a >= 0:
            return float(x_values[index])
        center = -b / (2 * a)
        if min(local_x) <= center <= max(local_x):
            return float(center)
        return float(x_values[index])

    def _estimate_fwhm_from_signal(self, x_values: np.ndarray, signal: np.ndarray, index: int) -> float | None:
        if len(x_values) < 5 or index <= 0 or index >= len(x_values) - 1:
            return None
        floor = float(np.nanpercentile(signal, 10))
        peak = float(signal[index])
        if not math.isfinite(peak) or peak <= floor:
            return None
        half = floor + 0.5 * (peak - floor)

        left_x = None
        for i in range(index, 0, -1):
            if signal[i - 1] <= half <= signal[i] or signal[i] <= half <= signal[i - 1]:
                denom = signal[i] - signal[i - 1]
                frac = 0.0 if abs(denom) < 1e-12 else (half - signal[i - 1]) / denom
                left_x = float(x_values[i - 1] + frac * (x_values[i] - x_values[i - 1]))
                break

        right_x = None
        for i in range(index, len(signal) - 1):
            if signal[i] >= half >= signal[i + 1] or signal[i] <= half <= signal[i + 1]:
                denom = signal[i + 1] - signal[i]
                frac = 0.0 if abs(denom) < 1e-12 else (half - signal[i]) / denom
                right_x = float(x_values[i] + frac * (x_values[i + 1] - x_values[i]))
                break

        if left_x is None or right_x is None or right_x <= left_x:
            return None
        fwhm = right_x - left_x
        return fwhm if math.isfinite(fwhm) and fwhm > 0 else None

    def _detect_peak_in_record(self, record: SpectrumRecord, peak: dict) -> dict | None:
        try:
            x_min = float(peak["x_min"])
            x_max = float(peak["x_max"])
        except (KeyError, TypeError, ValueError):
            return None
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        x = record.x
        y = record.y_average
        mask = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            return None

        x_roi = np.asarray(x[mask], dtype=float)
        y_roi = np.asarray(y[mask], dtype=float)
        order = np.argsort(x_roi)
        x_roi = x_roi[order]
        y_roi = y_roi[order]

        edge_count = max(1, min(6, len(y_roi) // 8))
        left_y = float(np.nanmedian(y_roi[:edge_count]))
        right_y = float(np.nanmedian(y_roi[-edge_count:]))
        baseline = np.interp(x_roi, [x_roi[0], x_roi[-1]], [left_y, right_y])
        signal = self._smooth_for_tracking(y_roi - baseline)
        if not np.any(np.isfinite(signal)):
            return None

        index = int(np.nanargmax(signal))
        center = self._quadratic_peak_center(x_roi, signal, index)
        prominence = float(signal[index] - np.nanpercentile(signal, 10))
        fwhm = self._estimate_fwhm_from_signal(x_roi, signal, index)
        return {
            "center": center,
            "score": prominence if math.isfinite(prominence) else 0.0,
            "fwhm": fwhm,
        }

    def _detect_peak_for_temperature(self, records: list[SpectrumRecord], temperature: float, peak: dict) -> dict | None:
        detections = [
            detection
            for record in records
            if abs(record.temperature - temperature) < 1e-6
            for detection in [self._detect_peak_in_record(record, peak)]
            if detection is not None
        ]
        if not detections:
            return None

        centers = np.asarray([item["center"] for item in detections], dtype=float)
        scores = np.asarray([max(float(item["score"]), 0.0) for item in detections], dtype=float)
        finite = np.isfinite(centers)
        if not np.any(finite):
            return None
        centers = centers[finite]
        scores = scores[finite]
        center = float(np.average(centers, weights=scores)) if np.sum(scores) > 1e-12 else float(np.median(centers))

        fwhm_values = np.asarray([item["fwhm"] for item in detections if item["fwhm"] is not None and math.isfinite(item["fwhm"])], dtype=float)
        fwhm = float(np.median(fwhm_values)) if len(fwhm_values) else None
        return {"center": center, "score": float(np.nanmax(scores)) if len(scores) else 0.0, "fwhm": fwhm}

    def _tracked_peak_from_detection(self, peak: dict, detection: dict) -> dict:
        tracked = copy.deepcopy(peak)
        old_center = float(tracked.get("center", detection["center"]))
        shift = float(detection["center"]) - old_center
        tracked["center"] = float(detection["center"])
        for field in ("x_min", "x_max"):
            if field in tracked:
                tracked[field] = float(tracked[field]) + shift
        fwhm = detection.get("fwhm")
        if fwhm is not None and math.isfinite(fwhm) and fwhm > 0:
            try:
                width_min = float(tracked.get("width_min", fwhm))
                width_max = float(tracked.get("width_max", fwhm))
            except (TypeError, ValueError):
                width_min, width_max = fwhm, fwhm
            tracked["width_guess"] = float(min(max(fwhm, width_min), width_max))
        tracked["auto_tracked_center"] = True
        tracked["auto_tracked_at"] = datetime.now().isoformat(timespec="seconds")
        return tracked

    def _max_x_inside_window_for_temperature(
        self,
        records: list[SpectrumRecord],
        temperature: float,
        x_min: float,
        x_max: float,
    ) -> float | None:
        positions: list[float] = []
        lo, hi = sorted((float(x_min), float(x_max)))
        for record in records:
            if abs(record.temperature - temperature) > 1e-6:
                continue
            x = record.x
            y = record.y_average
            mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                continue
            x_roi = np.asarray(x[mask], dtype=float)
            y_roi = np.asarray(y[mask], dtype=float)
            if len(x_roi) == 0:
                continue
            positions.append(float(x_roi[int(np.nanargmax(y_roi))]))
        if not positions:
            return None
        return float(np.median(np.asarray(positions, dtype=float)))

    def _propagated_x_bounds_from_redshift(self, reference_peak: dict, redshift: float) -> tuple[float, float]:
        reference_x_min = float(reference_peak["x_min"])
        reference_x_max = float(reference_peak["x_max"])
        return reference_x_min - redshift, reference_x_max - redshift

    def _propagated_peak_from_reference(self, reference_peak: dict, source_max_x: float, max_x: float, source_key: str, held: bool = False) -> dict:
        propagated = copy.deepcopy(reference_peak)
        redshift = float(source_max_x) - float(max_x)
        signed_shift = -redshift
        propagated_x_min, propagated_x_max = self._propagated_x_bounds_from_redshift(reference_peak, redshift)
        propagated["center"] = float(reference_peak["center"]) - redshift
        propagated["x_min"] = propagated_x_min
        propagated["x_max"] = propagated_x_max

        try:
            reference_width_max = float(reference_peak["width_max"])
        except (KeyError, TypeError, ValueError):
            reference_width_max = 0.0
        propagated["width_max"] = reference_width_max + PROPAGATION_WIDTH_MAX_INCREASE_PER_CM_SHIFT * abs(redshift)
        if "width_guess" in propagated:
            try:
                propagated["width_guess"] = min(float(propagated["width_guess"]), float(propagated["width_max"]))
            except (TypeError, ValueError):
                pass

        propagated["propagated_from_anchor"] = source_key
        propagated["propagated_reference_max_x_cm-1"] = float(source_max_x)
        propagated["propagated_measured_max_x_cm-1"] = float(max_x)
        propagated["propagated_redshift_cm-1"] = float(redshift)
        propagated["propagated_shift_cm-1"] = float(signed_shift)
        propagated["propagated_x_min_shift_cm-1"] = float(propagated_x_min - float(reference_peak["x_min"]))
        propagated["propagated_x_max_shift_cm-1"] = float(propagated_x_max - float(reference_peak["x_max"]))
        propagated["propagated_x_window_expansion_cm-1"] = float(abs(redshift))
        propagated["propagated_width_max_increase_cm-1"] = float(PROPAGATION_WIDTH_MAX_INCREASE_PER_CM_SHIFT * abs(redshift))
        propagated["propagation_held_previous_shift"] = bool(held)
        propagated["propagated_at"] = datetime.now().isoformat(timespec="seconds")
        return propagated

    def _track_peak_from_reference_across_temperatures(
        self,
        records: list[SpectrumRecord],
        temperatures: list[float],
        source_temp: float,
        reference_peak: dict,
        source_key: str,
    ) -> tuple[dict[float, dict], int]:
        try:
            reference_x_min = float(reference_peak["x_min"])
            reference_x_max = float(reference_peak["x_max"])
        except (KeyError, TypeError, ValueError):
            return {}, len(temperatures)

        source_max_x = self._max_x_inside_window_for_temperature(records, source_temp, reference_x_min, reference_x_max)
        if source_max_x is None:
            try:
                source_max_x = float(reference_peak["center"])
            except (KeyError, TypeError, ValueError):
                return {}, len(temperatures)

        propagated: dict[float, dict] = {}
        missed = 0

        previous_redshift = 0.0
        for temperature in sorted(temp for temp in temperatures if temp > source_temp):
            current_x_min, current_x_max = self._propagated_x_bounds_from_redshift(reference_peak, previous_redshift)
            max_x = self._max_x_inside_window_for_temperature(records, temperature, current_x_min, current_x_max)
            held = False
            if max_x is None:
                missed += 1
                held = True
                max_x = source_max_x - previous_redshift
            measured_redshift = float(source_max_x) - float(max_x)
            if measured_redshift < previous_redshift - PROPAGATION_DIRECTION_TOLERANCE_CM:
                missed += 1
                held = True
                measured_redshift = previous_redshift
                max_x = source_max_x - previous_redshift
            previous_redshift = measured_redshift
            propagated[temperature] = self._propagated_peak_from_reference(reference_peak, source_max_x, max_x, source_key, held=held)

        previous_redshift = 0.0
        for temperature in sorted((temp for temp in temperatures if temp < source_temp), reverse=True):
            current_x_min, current_x_max = self._propagated_x_bounds_from_redshift(reference_peak, previous_redshift)
            max_x = self._max_x_inside_window_for_temperature(records, temperature, current_x_min, current_x_max)
            held = False
            if max_x is None:
                missed += 1
                held = True
                max_x = source_max_x - previous_redshift
            measured_redshift = float(source_max_x) - float(max_x)
            if measured_redshift > previous_redshift + PROPAGATION_DIRECTION_TOLERANCE_CM:
                missed += 1
                held = True
                measured_redshift = previous_redshift
                max_x = source_max_x - previous_redshift
            previous_redshift = measured_redshift
            propagated[temperature] = self._propagated_peak_from_reference(reference_peak, source_max_x, max_x, source_key, held=held)

        return propagated, missed

    def _linked_peak_groups_for_propagation(self, family: str, source_peaks: dict[str, dict]) -> list[list[str]]:
        if family_kind(family) != "Au":
            return []
        au_ch_group = [peak_id for peak_id in ("CH_L1", "CH_L2") if peak_id in source_peaks]
        return [au_ch_group] if len(au_ch_group) >= 2 else []

    def _union_peak_window(self, peaks: list[dict]) -> dict:
        return {
            "x_min": min(float(peak["x_min"]) for peak in peaks),
            "x_max": max(float(peak["x_max"]) for peak in peaks),
        }

    def _track_linked_peak_group_from_reference_across_temperatures(
        self,
        records: list[SpectrumRecord],
        temperatures: list[float],
        source_temp: float,
        reference_peaks: dict[str, dict],
        source_key: str,
    ) -> tuple[dict[float, dict[str, dict]], int]:
        group_window = self._union_peak_window(list(reference_peaks.values()))
        source_max_x = self._max_x_inside_window_for_temperature(records, source_temp, group_window["x_min"], group_window["x_max"])
        if source_max_x is None:
            centers = [float(peak["center"]) for peak in reference_peaks.values() if "center" in peak]
            if not centers:
                return {}, len(temperatures)
            source_max_x = float(np.median(np.asarray(centers, dtype=float)))

        propagated: dict[float, dict[str, dict]] = {}
        missed = 0

        previous_redshift = 0.0
        for temperature in sorted(temp for temp in temperatures if temp > source_temp):
            current_x_min, current_x_max = self._propagated_x_bounds_from_redshift(group_window, previous_redshift)
            max_x = self._max_x_inside_window_for_temperature(records, temperature, current_x_min, current_x_max)
            held = False
            if max_x is None:
                missed += 1
                held = True
                max_x = source_max_x - previous_redshift
            measured_redshift = float(source_max_x) - float(max_x)
            if measured_redshift < previous_redshift - PROPAGATION_DIRECTION_TOLERANCE_CM:
                missed += 1
                held = True
                measured_redshift = previous_redshift
                max_x = source_max_x - previous_redshift
            previous_redshift = measured_redshift
            propagated[temperature] = {
                peak_id: self._propagated_peak_from_reference(peak, source_max_x, max_x, source_key, held=held)
                for peak_id, peak in reference_peaks.items()
            }

        previous_redshift = 0.0
        for temperature in sorted((temp for temp in temperatures if temp < source_temp), reverse=True):
            current_x_min, current_x_max = self._propagated_x_bounds_from_redshift(group_window, previous_redshift)
            max_x = self._max_x_inside_window_for_temperature(records, temperature, current_x_min, current_x_max)
            held = False
            if max_x is None:
                missed += 1
                held = True
                max_x = source_max_x - previous_redshift
            measured_redshift = float(source_max_x) - float(max_x)
            if measured_redshift > previous_redshift + PROPAGATION_DIRECTION_TOLERANCE_CM:
                missed += 1
                held = True
                measured_redshift = previous_redshift
                max_x = source_max_x - previous_redshift
            previous_redshift = measured_redshift
            propagated[temperature] = {
                peak_id: self._propagated_peak_from_reference(peak, source_max_x, max_x, source_key, held=held)
                for peak_id, peak in reference_peaks.items()
            }

        return propagated, missed

    def _propagate_current_anchor_by_shift(self) -> None:
        family = self.family_var.get()
        fam_settings = self._family_settings()
        source_key = self._current_temp_key()
        source_temp = self._current_anchor_temperature()
        if not family or fam_settings is None or not source_key or source_temp is None:
            messagebox.showinfo(APP_TITLE, "Select a family and source temperature anchor first.")
            return

        records = self._records_for_tracking(family)
        temperatures = sorted({record.temperature for record in records})
        if source_temp not in temperatures:
            closest = min(temperatures, key=lambda temp: abs(temp - source_temp)) if temperatures else None
            messagebox.showerror(
                APP_TITLE,
                f"The selected source anchor {source_key} K is not an actual measured temperature for the current selection."
                + (f"\nClosest measured temperature is {closest:g} K." if closest is not None else ""),
            )
            return

        source_peaks = {
            peak_id: peak
            for peak_id, peak in self._effective_peaks(family=family, temp_key=source_key).items()
            if peak.get("enabled", True)
        }
        if not source_peaks:
            messagebox.showinfo(APP_TITLE, "No enabled peaks are available in the current source anchor.")
            return

        if not messagebox.askyesno(
            APP_TITLE,
            f"Use {source_key} K as the locked reference and propagate from max-point drift?\n\n"
            f"Family: {family}\n"
            f"UP/DOWN selection: {self.sequence_var.get()}\n"
            f"Peaks: {len(source_peaks)}\n\n"
            "The source anchor itself will not be changed.\n"
            "At each temperature, the app takes the max scatter point inside the current window.\n"
            "redshift = source max X - current max X.\n"
            "The full X window shifts by redshift with constant width.\n"
            "For Au families, CH_L1 and CH_L2 are linked and use the shared CH envelope drift.\n"
            "width_max increases by 2 * abs(drift).\n"
            "This only updates anchors in memory until you click Save JSON now.",
        ):
            return

        anchors = fam_settings.setdefault("temperature_anchors", {})
        original_source_anchor = copy.deepcopy(anchors.get(source_key))
        propagated_count = 0
        missed_count = 0

        linked_groups = self._linked_peak_groups_for_propagation(family, source_peaks)
        linked_peak_ids = {peak_id for group in linked_groups for peak_id in group}
        for group in linked_groups:
            group_peaks = {peak_id: source_peaks[peak_id] for peak_id in group}
            propagated_group_by_temp, missed = self._track_linked_peak_group_from_reference_across_temperatures(
                records=records,
                temperatures=temperatures,
                source_temp=source_temp,
                reference_peaks=group_peaks,
                source_key=source_key,
            )
            missed_count += missed
            for temperature, propagated_peaks in propagated_group_by_temp.items():
                temp_key = temperature_key(temperature)
                if temp_key == source_key:
                    continue
                anchor = anchors.setdefault(temp_key, {"temperature_k": float(temperature), "peaks": {}})
                anchor.setdefault("peaks", {}).update(propagated_peaks)
                propagated_count += len(propagated_peaks)

        for peak_id, reference_peak in source_peaks.items():
            if peak_id in linked_peak_ids:
                continue
            propagated_by_temp, missed = self._track_peak_from_reference_across_temperatures(
                records=records,
                temperatures=temperatures,
                source_temp=source_temp,
                reference_peak=reference_peak,
                source_key=source_key,
            )
            missed_count += missed
            for temperature, propagated_peak in propagated_by_temp.items():
                temp_key = temperature_key(temperature)
                if temp_key == source_key:
                    continue
                anchor = anchors.setdefault(temp_key, {"temperature_k": float(temperature), "peaks": {}})
                anchor.setdefault("peaks", {})[peak_id] = propagated_peak
                propagated_count += 1

        if original_source_anchor is not None:
            anchors[source_key] = original_source_anchor

        self.scope_var.set("temperature_anchor")
        self.temp_anchor_var.set(source_key)
        self._refresh_temp_anchor_combo()
        self._refresh_peak_table()
        self.status_var.set(
            f"Propagated {propagated_count} peak-temperature bounds from {source_key} K; "
            f"{missed_count} point(s) held at the previous drift. Review anchors, then Save JSON now."
        )

    def _auto_track_peak_centers(self, selected_only: bool) -> None:
        family = self.family_var.get()
        fam_settings = self._family_settings()
        if not family or fam_settings is None:
            messagebox.showinfo(APP_TITLE, "Select a family first.")
            return

        records = self._records_for_tracking(family)
        if not records:
            messagebox.showinfo(APP_TITLE, "No spectra are available for the current family/UP-DOWN selection.")
            return

        if selected_only:
            selected = list(self.peak_tree.selection())
            if not selected:
                messagebox.showinfo(APP_TITLE, "Select one peak row first, then click Auto-track selected peak.")
                return
            peak_ids = selected
        else:
            base_peaks = fam_settings.get("peaks", {})
            peak_ids = [peak_id for peak_id, peak in base_peaks.items() if peak.get("enabled", True)]
            if not peak_ids:
                messagebox.showinfo(APP_TITLE, "No enabled family-default peaks are available to auto-track.")
                return
            if not messagebox.askyesno(
                APP_TITLE,
                "Auto-track all enabled peak centers for this family?\n\n"
                "This will create/update temperature anchors in memory. It will not write the JSON until you click Save JSON now.",
            ):
                return

        temperatures = sorted({record.temperature for record in records})
        anchors = fam_settings.setdefault("temperature_anchors", {})
        tracked_count = 0
        missed_count = 0
        sequence_label = self.sequence_var.get() if self.sequence_var.get() != ALL_SEQUENCES else "all UP/DOWN folders"

        for temperature in temperatures:
            temp_key = temperature_key(temperature)
            anchor = anchors.setdefault(temp_key, {"temperature_k": float(temperature), "peaks": {}})
            anchor.setdefault("peaks", {})
            for peak_id in peak_ids:
                start_peak = self._resolved_peaks_for_temperature(family, temperature).get(peak_id)
                if not start_peak or not start_peak.get("enabled", True):
                    missed_count += 1
                    continue
                detection = self._detect_peak_for_temperature(records, temperature, start_peak)
                if detection is None:
                    missed_count += 1
                    continue
                anchor["peaks"][peak_id] = self._tracked_peak_from_detection(start_peak, detection)
                tracked_count += 1

        self.scope_var.set("temperature_anchor")
        if temperatures:
            self.temp_anchor_var.set(temperature_key(temperatures[0]))
        self._refresh_temp_anchor_combo()
        self._refresh_peak_table()
        self.status_var.set(
            f"Auto-tracked {tracked_count} peak-temperature center(s) for {family} using {sequence_label}; "
            f"{missed_count} missed. Review anchors, then Save JSON now."
        )

    def _reset_family_defaults(self) -> None:
        family = self.family_var.get()
        if not family:
            return
        if not messagebox.askyesno(APP_TITLE, f"Reset {family} family defaults to smart starting values?\nTemperature anchors are kept."):
            return
        self.settings["families"][family]["peaks"] = smart_default_peaks(family)
        self.status_var.set(f"Reset smart defaults for {family}.")
        self._refresh_peak_table()

    def _selected_records_for_plot(self) -> list[SpectrumRecord]:
        family = self.family_var.get()
        if not family:
            return []
        records = [record for record in self.records if record.family == family]
        sequence = self.sequence_var.get()
        if sequence and sequence != ALL_SEQUENCES:
            records = [record for record in records if record.sequence == sequence]

        if self.scope_var.get() == "temperature_anchor":
            anchor_temp = self._current_anchor_temperature()
            if anchor_temp is not None:
                return self._nearest_anchor_records(records, anchor_temp)

        step = self._float_value(self.step_var.get(), 30.0)
        if step <= 0:
            return sorted(records, key=lambda r: (r.sequence, r.temperature, r.path.name))

        selected: list[SpectrumRecord] = []
        for sequence_name in sorted({record.sequence for record in records}):
            seq_records = sorted([record for record in records if record.sequence == sequence_name], key=lambda r: (r.temperature, r.path.name))
            if not seq_records:
                continue
            temps = sorted({record.temperature for record in seq_records})
            targets = list(np.arange(min(temps), max(temps) + step * 0.5, step))
            if max(temps) not in targets:
                targets.append(max(temps))
            chosen_temps: set[float] = set()
            for target in targets:
                closest = min(temps, key=lambda temp: abs(temp - target))
                chosen_temps.add(closest)
            for record in seq_records:
                if record.temperature in chosen_temps:
                    selected.append(record)
        return sorted(selected, key=lambda r: (r.sequence, r.temperature, r.path.name))

    def _nearest_anchor_records(self, records: list[SpectrumRecord], anchor_temp: float) -> list[SpectrumRecord]:
        selected: list[SpectrumRecord] = []
        for sequence_name in sorted({record.sequence for record in records}):
            seq_records = [record for record in records if record.sequence == sequence_name]
            temps = sorted({record.temperature for record in seq_records})
            closest = sorted(temps, key=lambda temp: (abs(temp - anchor_temp), temp))[:ANCHOR_PREVIEW_TEMPERATURE_COUNT]
            chosen_temps = set(closest)
            selected.extend(record for record in seq_records if record.temperature in chosen_temps)
        return sorted(selected, key=lambda r: (r.sequence, r.temperature, r.path.name))

    def _draw_plot(self) -> None:
        self.ax.clear()
        self.hover_points = []
        self.hover_annotation = None
        self.measure_artists = []
        if not self.records:
            self.ax.text(0.5, 0.5, "Scan ALS_BASELINE_CORRECTED to tune peak bounds", transform=self.ax.transAxes, ha="center", va="center", fontsize=14)
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        records = self._selected_records_for_plot()
        if not records:
            self.ax.text(0.5, 0.5, "No spectra for this family/sequence", transform=self.ax.transAxes, ha="center", va="center", fontsize=14)
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        self.ax.set_axis_on()
        x_min = self._float_value(self.x_min_var.get(), 250.0)
        x_max = self._float_value(self.x_max_var.get(), 1700.0)
        offset = self._float_value(self.offset_var.get(), 1.15)
        family = self.family_var.get()
        color = SAMPLE_COLORS.get(family, "#333333")
        cmap = matplotlib.colormaps.get_cmap("turbo")
        temps = [record.temperature for record in records]
        t_min, t_max = min(temps), max(temps)
        denom = max(t_max - t_min, 1e-9)

        for idx, record in enumerate(records):
            x = record.x
            y = record.y_average.astype(float, copy=True)
            mask = (x >= x_min) & (x <= x_max) & np.isfinite(y)
            if mask.sum() < 2:
                continue
            x_plot = x[mask]
            y_raw_plot = y[mask].copy()
            y_plot = y_raw_plot.copy()
            if self.normalize_var.get():
                span = np.nanmax(y_plot) - np.nanmin(y_plot)
                if span > 0 and math.isfinite(span):
                    y_plot = (y_plot - np.nanmin(y_plot)) / span
            y_plot = y_plot + idx * offset
            temp_color = cmap((record.temperature - t_min) / denom)
            plot_color = temp_color if self.normalize_var.get() else color
            line_style = SEQUENCE_STYLES.get(record.sequence, "-")
            self.ax.plot(
                x_plot,
                y_plot,
                color=plot_color,
                linestyle=line_style,
                linewidth=1.1,
                alpha=0.9,
                marker="o" if self.show_points_var.get() else None,
                markersize=2.2 if self.show_points_var.get() else 0,
                markerfacecolor=plot_color if self.show_points_var.get() else None,
                markeredgewidth=0 if self.show_points_var.get() else 0,
                label=f"{record.sequence} {record.temperature:g}K",
            )
            self.hover_points.append(
                {
                    "x": x_plot,
                    "y_plot": y_plot,
                    "y_raw": y_raw_plot,
                    "record": record,
                }
            )

        temp_key = self._current_temp_key() if self.scope_var.get() == "temperature_anchor" else None
        peaks = self._effective_peaks(temp_key=temp_key)
        y0, y1 = self.ax.get_ylim()
        y_range = y1 - y0 if y1 > y0 else 1
        label_y = y1 - 0.04 * y_range
        peak_colors = ["#f59e0b", "#8b5cf6", "#10b981", "#ef4444", "#3b82f6", "#14b8a6", "#64748b"]
        for idx, peak in enumerate(sorted(peaks.values(), key=lambda p: float(p.get("center", 0)))):
            if not peak.get("enabled", True):
                continue
            xlo = float(peak.get("x_min", 0))
            xhi = float(peak.get("x_max", 0))
            center = float(peak.get("center", (xlo + xhi) / 2))
            shade = peak_colors[idx % len(peak_colors)]
            self.ax.axvspan(xlo, xhi, color=shade, alpha=0.10, lw=0)
            self.ax.axvline(center, color=shade, linestyle="--", linewidth=1.1, alpha=0.95)
            label = str(peak.get("id", "peak"))
            suffix = "" if peak.get("required", True) else " opt"
            width_text = f"W {self._fmt(peak.get('width_min'))}-{self._fmt(peak.get('width_max'))}"
            self.ax.text(center, label_y, f"{label}{suffix}\n{width_text}", ha="center", va="top", rotation=90, fontsize=8, color="#111827")

        if self.scope_var.get() == "temperature_anchor":
            scope = f"{self.temp_anchor_var.get()} K anchor, showing {ANCHOR_PREVIEW_TEMPERATURE_COUNT} closest measured temperatures per UP/DOWN folder"
        else:
            scope = f"family defaults, every ~{self.step_var.get()} K"
        self.ax.set_title(f"{FAMILY_LABELS.get(family, family)} - averaged TXT spectra - editing {scope}", fontsize=13)
        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Averaged intensity (offset / normalized)" if self.normalize_var.get() else "Averaged intensity (a.u.)")
        self.ax.set_xlim(x_min, x_max)
        self.ax.grid(True, alpha=0.18)

        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(handles[:18], labels[:18], loc="upper right", fontsize=7, frameon=False, ncol=2)
        self._make_hover_annotation()
        self._draw_measurement(redraw=False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_plot_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.last_click_x = float(event.xdata)
        self.click_var.set(f"Last click X: {self.last_click_x:.3f} cm^-1")
        if self.measure_mode_var.get() and getattr(event, "button", None) == 1:
            self._handle_measure_click(self.last_click_x)

    def _make_hover_annotation(self) -> None:
        self.hover_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#374151", "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.8},
            fontsize=8,
            visible=False,
            zorder=20,
        )

    def _nearest_hover_point(self, event) -> dict | None:
        if event.xdata is None or event.ydata is None:
            return None
        best: dict | None = None
        for series in self.hover_points:
            x = series["x"]
            y_plot = series["y_plot"]
            y_raw = series["y_raw"]
            if len(x) == 0:
                continue
            nearest_x_index = int(np.nanargmin(np.abs(x - event.xdata)))
            for candidate in range(nearest_x_index - 4, nearest_x_index + 5):
                if candidate < 0 or candidate >= len(x):
                    continue
                px, py = self.ax.transData.transform((float(x[candidate]), float(y_plot[candidate])))
                distance = math.hypot(px - event.x, py - event.y)
                if best is None or distance < best["distance"]:
                    best = {
                        "distance": distance,
                        "x": float(x[candidate]),
                        "y_plot": float(y_plot[candidate]),
                        "y_raw": float(y_raw[candidate]),
                        "record": series["record"],
                    }
        if best and best["distance"] <= 20:
            return best
        return None

    def _on_plot_hover(self, event) -> None:
        if not self.hover_enabled_var.get() or event.inaxes != self.ax or self.hover_annotation is None:
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return
        point = self._nearest_hover_point(event)
        if point is None:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        record = point["record"]
        label = FAMILY_LABELS.get(record.family, record.family)
        text = (
            f"{label}\n"
            f"{record.sequence} | {record.temperature:g} K | {record.n_y_columns} Y col(s)\n"
            f"X = {point['x']:.3f} cm-1\n"
            f"Intensity = {point['y_raw']:.6g}\n"
            f"Displayed Y = {point['y_plot']:.6g}"
        )
        self.hover_annotation.xy = (point["x"], point["y_plot"])
        self.hover_annotation.set_text(text)
        self.hover_annotation.set_visible(True)
        self.hover_info_var.set(f"Hover: X {point['x']:.3f} cm-1, intensity {point['y_raw']:.6g}, {record.temperature:g} K")
        self.canvas.draw_idle()

    def _on_axes_leave(self, _event) -> None:
        if self.hover_annotation is not None and self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self.canvas.draw_idle()

    def _on_hover_toggle(self) -> None:
        if not self.hover_enabled_var.get() and self.hover_annotation is not None:
            self.hover_annotation.set_visible(False)
            self.hover_info_var.set("Hover: off")
            self.canvas.draw_idle()
        elif self.hover_enabled_var.get():
            self.hover_info_var.set("Hover: move over a spectrum point")

    def _on_measure_toggle(self) -> None:
        if self.measure_mode_var.get():
            self.measure_status_var.set("Ruler: click two X positions on a peak")
        else:
            self.measure_status_var.set("Ruler: off")

    def _handle_measure_click(self, x_value: float) -> None:
        if len(self.measure_xs) >= 2:
            self.measure_xs = [x_value]
            self.measure_status_var.set(f"Ruler: first point {x_value:.3f} cm-1")
        else:
            self.measure_xs.append(x_value)
            if len(self.measure_xs) == 1:
                self.measure_status_var.set(f"Ruler: first point {x_value:.3f} cm-1")
            else:
                delta = abs(self.measure_xs[1] - self.measure_xs[0])
                self.measure_status_var.set(
                    f"Ruler: {self.measure_xs[0]:.3f} to {self.measure_xs[1]:.3f} cm-1, delta = {delta:.3f} cm-1"
                )
        self._draw_measurement(redraw=True)

    def _clear_measure_artists(self) -> None:
        for artist in self.measure_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.measure_artists = []

    def _draw_measurement(self, redraw: bool) -> None:
        self._clear_measure_artists()
        if not self.measure_xs:
            if redraw:
                self.canvas.draw_idle()
            return
        y0, y1 = self.ax.get_ylim()
        y_range = y1 - y0 if y1 > y0 else 1.0
        y_line = y1 - 0.08 * y_range
        y_text = y1 - 0.035 * y_range
        color = "#be185d"

        for x_value in self.measure_xs[:2]:
            self.measure_artists.append(self.ax.axvline(x_value, color=color, linestyle="-.", linewidth=1.4, alpha=0.95, zorder=18))

        if len(self.measure_xs) >= 2:
            x_left, x_right = sorted(self.measure_xs[:2])
            delta = x_right - x_left
            self.measure_artists.append(self.ax.hlines(y_line, x_left, x_right, color=color, linewidth=1.4, zorder=18))
            self.measure_artists.append(self.ax.plot([x_left, x_left], [y_line - 0.015 * y_range, y_line + 0.015 * y_range], color=color, linewidth=1.2, zorder=18)[0])
            self.measure_artists.append(self.ax.plot([x_right, x_right], [y_line - 0.015 * y_range, y_line + 0.015 * y_range], color=color, linewidth=1.2, zorder=18)[0])
            self.measure_artists.append(
                self.ax.text(
                    (x_left + x_right) / 2,
                    y_text,
                    f"delta x = {delta:.3f} cm-1",
                    color=color,
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
                    zorder=19,
                )
            )
        if redraw:
            self.canvas.draw_idle()

    def _clear_measurement(self) -> None:
        self.measure_xs = []
        self._clear_measure_artists()
        self.measure_status_var.set("Ruler: cleared")
        self.canvas.draw_idle()

    def _use_click(self, field: str) -> None:
        if self.last_click_x is None:
            messagebox.showinfo(APP_TITLE, "Click inside the plot first.")
            return
        value = f"{self.last_click_x:.3f}"
        if field == "x_min":
            self.peak_x_min_var.set(value)
        elif field == "x_max":
            self.peak_x_max_var.set(value)
        elif field == "center":
            self.peak_center_var.set(value)

    def _float_value(self, value: str, default: float) -> float:
        try:
            parsed = float(str(value).strip())
        except ValueError:
            return default
        return parsed if math.isfinite(parsed) else default

    def _strict_float(self, value: str, label: str) -> float:
        try:
            parsed = float(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must be finite.")
        return parsed

    def _fmt(self, value) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return ""
        if abs(parsed - round(parsed)) < 1e-8:
            return str(int(round(parsed)))
        return f"{parsed:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    app = PeakBoundsTunerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
