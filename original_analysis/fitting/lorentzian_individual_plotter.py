"""
Paper-quality Lorentzian deconvolution plotter for fitted Raman spectra.

Reads the cluster-generated *_long_results.csv files and the original TXT
spectra, then reconstructs each fitted Lorentzian contribution for one TXT
Y column at a time. The spectra/data folders are treated as read-only.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


APP_DIR = Path(__file__).resolve().parent
DEFAULT_FITTED_DIR = APP_DIR / "Fitted_Spectra_ALL_Cluster_ETH"
DEFAULT_SPECTRA_DIR = APP_DIR / "ALS_BASELINE_CORRECTED"
DEFAULT_OUTPUT_DIR = APP_DIR / "PAPER_LORENTZIAN_FIT_FIGURES"
SETTINGS_FILE = APP_DIR / "paper_lorentzian_fit_plotter_settings.json"
SETTINGS_STYLE_VERSION = 3

TARGET_SAMPLE_FOLDERS = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "Aligned_RO_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
]

FAMILY_TO_KEY = {
    "Aligned_Au_3A": "aligned_au_3a",
    "Aligned_Au_8A": "aligned_au_8a",
    "Aligned_RO_8A": "aligned_ro_8a",
    "MIRA_Au_unaligned_8A": "unaligned_au_8a",
    "MIRA_RO_unaligned_8A": "unaligned_ro_8a",
}

KEY_TO_FAMILY = {v: k for k, v in FAMILY_TO_KEY.items()}

SAMPLE_COLORS = {
    "aligned_au_3a": "#feb715",
    "aligned_au_8a": "#f08228",
    "aligned_ro_8a": "#00c8c8",
    "unaligned_au_8a": "#ae540b",
    "unaligned_ro_8a": "#008686",
}

SAMPLE_LABELS = {
    "aligned_au_3a": "Aligned Au 3A",
    "aligned_au_8a": "Aligned Au 8A",
    "aligned_ro_8a": "Aligned RO 8A",
    "unaligned_au_8a": "Unaligned Au 8A",
    "unaligned_ro_8a": "Unaligned RO 8A",
}

FAMILY_PLOT_ORDER = [
    "aligned_au_3a",
    "aligned_au_8a",
    "unaligned_au_8a",
    "unaligned_ro_8a",
    "aligned_ro_8a",
]

PEAK_ORDER = ["RBLM", "CH_L1", "CH_L2", "CH_MID", "D", "G"]
PEAK_LABELS = {
    "RBLM": "RBLM",
    "CH_L1": "CH L1",
    "CH_L2": "CH L2",
    "CH_MID": "CH MID",
    "D": "D",
    "G": "G",
}
PEAK_COLORS = {
    "RBLM": "#2878b5",
    "CH_L1": "#35a64a",
    "CH_L2": "#8b62cf",
    "CH_MID": "#e573c5",
    "D": "#b7b600",
    "G": "#17becf",
}


def setup_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans L"],
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )

SPECTRUM_GROUP_COLS = [
    "family",
    "sequence",
    "temperature_K",
    "file_path",
    "file_name",
    "spectrum_in_file",
    "y_column_number",
    "y_column_name",
]


@dataclass(frozen=True)
class SpectrumKey:
    family: str
    sequence: str
    temperature_K: float
    file_path: str
    file_name: str
    spectrum_in_file: int
    y_column_number: int
    y_column_name: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "SpectrumKey":
        return cls(
            family=str(row["family"]),
            sequence=str(row["sequence"]),
            temperature_K=float(row["temperature_K"]),
            file_path=str(row["file_path"]),
            file_name=str(row["file_name"]),
            spectrum_in_file=int(row["spectrum_in_file"]),
            y_column_number=int(row["y_column_number"]),
            y_column_name=str(row["y_column_name"]),
        )


def family_key_from_name(family_name: str) -> str:
    return FAMILY_TO_KEY.get(str(family_name), str(family_name).lower())


def family_sort_index(family_name: str) -> int:
    key = family_key_from_name(family_name)
    try:
        return FAMILY_PLOT_ORDER.index(key)
    except ValueError:
        return len(FAMILY_PLOT_ORDER)


def sample_marker_style(family_name: str) -> Dict[str, object]:
    """Central marker style used for both plot data and legend handles."""
    key = family_key_from_name(family_name)
    color = SAMPLE_COLORS.get(key, "#333333")
    marker = "o" if key == "aligned_au_3a" else "D"
    is_unaligned = key in {"unaligned_au_8a", "unaligned_ro_8a"}
    style: Dict[str, object] = {
        "marker": marker,
        "edgecolors": color,
        "linewidths": 1.9 if is_unaligned else 0.9,
    }
    if is_unaligned:
        style["facecolors"] = "none"
    else:
        style["facecolors"] = color
    return style


def family_label(family_name: str) -> str:
    return SAMPLE_LABELS.get(family_key_from_name(family_name), str(family_name).replace("_", " "))


def safe_filename(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", str(text))
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text[:max_len] or "spectrum"


def parse_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    except (TypeError, ValueError):
        pass
    return default


def parse_int(value: object, default: int) -> int:
    try:
        if value is None:
            return default
        value_i = int(float(value))
        return value_i
    except (TypeError, ValueError):
        return default


def lorentzian(x: np.ndarray, height: float, pos: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-12)
    return float(height) / (1.0 + ((x - float(pos)) / (0.5 * width)) ** 2)


def read_txt_spectra(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read TXT with first column as X and all remaining columns as Y spectra."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line.rstrip("\n")
                break
    if not first_line:
        raise ValueError(f"Empty TXT file: {path}")

    if "\t" in first_line:
        sep = "\t"
        first_parts = first_line.split("\t")
    elif "," in first_line:
        sep = ","
        first_parts = first_line.split(",")
    elif ";" in first_line:
        sep = ";"
        first_parts = first_line.split(";")
    else:
        sep = r"\s+"
        first_parts = re.split(r"\s+", first_line.strip())

    def all_numeric(parts: Iterable[str]) -> bool:
        for part in parts:
            try:
                float(part)
            except ValueError:
                return False
        return True

    header = None if all_numeric(first_parts) else 0
    if sep == r"\s+":
        df = pd.read_csv(path, sep=sep, header=header, engine="python", comment="#")
    else:
        df = pd.read_csv(path, sep=sep, header=header, engine="python", comment="#")

    if header is None:
        df.columns = [f"Column {idx + 1}" for idx in range(df.shape[1])]
    else:
        df.columns = [str(col).strip() or f"Column {idx + 1}" for idx, col in enumerate(df.columns)]

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2:
        raise ValueError(f"Expected X plus at least one Y column in {path}")

    x = numeric.iloc[:, 0].to_numpy(dtype=float)
    y = numeric.iloc[:, 1:].to_numpy(dtype=float)
    finite_x = np.isfinite(x)
    x = x[finite_x]
    y = y[finite_x, :]
    y_names = [str(col) for col in df.columns[1:]]
    return x, y, y_names


class LorentzianPaperPlotterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Raman Lorentzian Fit Paper Plotter - family markers and uncertainty bands")
        self.settings = self.load_settings()

        self.results_df = pd.DataFrame()
        self.records_df = pd.DataFrame()
        self.visible_records = pd.DataFrame()
        self.iid_to_key: Dict[str, SpectrumKey] = {}
        self.current_key: Optional[SpectrumKey] = None
        self.current_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.hover_annotation = None
        self.hover_point = None
        self.path_cache: Dict[str, Path] = {}

        self.fitted_dir_var = tk.StringVar(value=self.settings.get("fitted_dir", str(DEFAULT_FITTED_DIR)))
        self.source_spectra_dir_var = tk.StringVar(
            value=self.settings.get("source_spectra_dir", str(DEFAULT_SPECTRA_DIR))
        )
        self.output_dir_var = tk.StringVar(value=self.settings.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
        self.family_var = tk.StringVar(value=self.settings.get("family", "All families"))
        self.sequence_var = tk.StringVar(value=self.settings.get("sequence", "All sequences"))
        self.x_min_var = tk.StringVar(value=str(self.settings.get("x_min", "200")))
        self.x_max_var = tk.StringVar(value=str(self.settings.get("x_max", "2000")))
        self.marker_size_var = tk.StringVar(value=str(self.settings.get("marker_size", "22")))
        self.marker_edge_width_var = tk.StringVar(value=str(self.settings.get("marker_edge_width", "1.3")))
        self.scatter_alpha_var = tk.StringVar(value=str(self.settings.get("scatter_alpha", "0.95")))
        self.data_line_width_var = tk.StringVar(value=str(self.settings.get("data_line_width", "1.1")))
        self.fit_line_width_var = tk.StringVar(value=str(self.settings.get("fit_line_width", "2.5")))
        self.component_line_width_var = tk.StringVar(value=str(self.settings.get("component_line_width", "1.8")))
        self.uncertainty_samples_var = tk.StringVar(value=str(self.settings.get("uncertainty_samples", "300")))
        self.band_alpha_var = tk.StringVar(value=str(self.settings.get("band_alpha", "0.16")))
        self.dpi_var = tk.StringVar(value=str(self.settings.get("dpi", "300")))
        self.figure_width_var = tk.StringVar(value=str(self.settings.get("figure_width", "10.0")))
        self.figure_height_var = tk.StringVar(value=str(self.settings.get("figure_height", "5.8")))
        self.preview_export_aspect_var = tk.BooleanVar(value=bool(self.settings.get("preview_export_aspect", True)))
        self.title_font_var = tk.StringVar(value=str(self.settings.get("title_font", "20")))
        self.axis_label_font_var = tk.StringVar(value=str(self.settings.get("axis_label_font", "20")))
        self.tick_font_var = tk.StringVar(value=str(self.settings.get("tick_font", "16")))
        self.legend_font_var = tk.StringVar(value=str(self.settings.get("legend_font", "12")))
        self.stats_font_var = tk.StringVar(value=str(self.settings.get("stats_font", "13")))
        self.spine_width_var = tk.StringVar(value=str(self.settings.get("spine_width", "1.2")))
        self.tick_width_var = tk.StringVar(value=str(self.settings.get("tick_width", "1.2")))
        self.tick_length_var = tk.StringVar(value=str(self.settings.get("tick_length", "5")))
        self.legend_location_var = tk.StringVar(value=str(self.settings.get("legend_location", "upper right")))
        self.show_legend_var = tk.BooleanVar(value=bool(self.settings.get("show_legend", True)))
        self.show_uncertainty_var = tk.BooleanVar(value=bool(self.settings.get("show_uncertainty", True)))
        self.show_component_uncertainty_var = tk.BooleanVar(
            value=bool(self.settings.get("show_component_uncertainty", False))
        )
        self.show_bounds_var = tk.BooleanVar(value=bool(self.settings.get("show_bounds", False)))
        self.show_title_var = tk.BooleanVar(value=bool(self.settings.get("show_title", True)))
        self.hide_y_ticks_var = tk.BooleanVar(value=bool(self.settings.get("hide_y_ticks", True)))
        self.png_var = tk.BooleanVar(value=bool(self.settings.get("save_png", True)))
        self.pdf_var = tk.BooleanVar(value=bool(self.settings.get("save_pdf", True)))
        self.svg_var = tk.BooleanVar(value=bool(self.settings.get("save_svg", True)))

        self._build_ui()
        if Path(self.fitted_dir_var.get()).exists():
            self.scan_results()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_settings(self) -> Dict[str, object]:
        if SETTINGS_FILE.exists():
            try:
                settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                if settings.get("style_version") != SETTINGS_STYLE_VERSION:
                    settings["show_uncertainty"] = True
                    settings["show_component_uncertainty"] = False
                    settings["show_bounds"] = False
                    settings["hide_y_ticks"] = True
                    settings["band_alpha"] = "0.16"
                    settings["style_version"] = SETTINGS_STYLE_VERSION
                return settings
            except Exception:
                return {}
        return {}

    def save_settings(self) -> None:
        settings = {
            "style_version": SETTINGS_STYLE_VERSION,
            "fitted_dir": self.fitted_dir_var.get(),
            "source_spectra_dir": self.source_spectra_dir_var.get(),
            "output_dir": self.output_dir_var.get(),
            "left_panel_width": self.current_left_panel_width(),
            "family": self.family_var.get(),
            "sequence": self.sequence_var.get(),
            "x_min": self.x_min_var.get(),
            "x_max": self.x_max_var.get(),
            "marker_size": self.marker_size_var.get(),
            "marker_edge_width": self.marker_edge_width_var.get(),
            "scatter_alpha": self.scatter_alpha_var.get(),
            "data_line_width": self.data_line_width_var.get(),
            "fit_line_width": self.fit_line_width_var.get(),
            "component_line_width": self.component_line_width_var.get(),
            "uncertainty_samples": self.uncertainty_samples_var.get(),
            "band_alpha": self.band_alpha_var.get(),
            "dpi": self.dpi_var.get(),
            "figure_width": self.figure_width_var.get(),
            "figure_height": self.figure_height_var.get(),
            "preview_export_aspect": self.preview_export_aspect_var.get(),
            "title_font": self.title_font_var.get(),
            "axis_label_font": self.axis_label_font_var.get(),
            "tick_font": self.tick_font_var.get(),
            "legend_font": self.legend_font_var.get(),
            "stats_font": self.stats_font_var.get(),
            "spine_width": self.spine_width_var.get(),
            "tick_width": self.tick_width_var.get(),
            "tick_length": self.tick_length_var.get(),
            "legend_location": self.legend_location_var.get(),
            "show_legend": self.show_legend_var.get(),
            "show_uncertainty": self.show_uncertainty_var.get(),
            "show_component_uncertainty": self.show_component_uncertainty_var.get(),
            "show_bounds": self.show_bounds_var.get(),
            "show_title": self.show_title_var.get(),
            "hide_y_ticks": self.hide_y_ticks_var.get(),
            "save_png": self.png_var.get(),
            "save_pdf": self.pdf_var.get(),
            "save_svg": self.svg_var.get(),
        }
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def current_left_panel_width(self) -> int:
        if hasattr(self, "left_canvas"):
            width = self.left_canvas.winfo_width()
            if width > 1:
                return int(width)
        return parse_int(self.settings.get("left_panel_width"), 430)

    def _restore_left_panel_width(self) -> None:
        self.set_left_panel_width(parse_int(self.settings.get("left_panel_width"), 430))

    def set_left_panel_width(self, width: int) -> None:
        if not hasattr(self, "left_canvas"):
            return
        root_width = self.root.winfo_width()
        max_width = 1200
        if root_width > 0:
            max_width = max(360, min(max_width, root_width - 520))
        width = max(340, min(max_width, int(width)))
        self.left_canvas.configure(width=width)
        if hasattr(self, "left_outer"):
            self.left_outer.configure(width=width)
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _start_left_resize(self, event) -> None:
        self._left_resize_start_x = event.x_root
        self._left_resize_start_width = self.current_left_panel_width()

    def _resize_left_panel(self, event) -> None:
        start_x = getattr(self, "_left_resize_start_x", event.x_root)
        start_width = getattr(self, "_left_resize_start_width", self.current_left_panel_width())
        self.set_left_panel_width(start_width + (event.x_root - start_x))

    def _finish_left_resize(self, _event) -> None:
        self.save_settings()

    def _on_left_panel_configure(self, _event) -> None:
        if hasattr(self, "left_canvas"):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event) -> None:
        if hasattr(self, "left_canvas_window"):
            self.left_canvas.itemconfigure(self.left_canvas_window, width=event.width)

    def _bind_left_mousewheel(self, _event) -> None:
        self.left_canvas.bind_all("<MouseWheel>", self._on_left_mousewheel)

    def _unbind_left_mousewheel(self, _event=None) -> None:
        if hasattr(self, "left_canvas"):
            self.left_canvas.unbind_all("<MouseWheel>")

    def _on_left_mousewheel(self, event):
        if hasattr(self, "tree") and event.widget is self.tree:
            return None
        self.left_canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame_bg = ttk.Style().lookup("TFrame", "background") or self.root.cget("background")
        initial_left_width = max(360, min(1200, parse_int(self.settings.get("left_panel_width"), 430)))
        left_outer = ttk.Frame(self.root, width=initial_left_width)
        self.left_outer = left_outer
        left_outer.grid(row=0, column=0, sticky="nsw")
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)
        left_outer.grid_propagate(False)

        self.left_canvas = tk.Canvas(
            left_outer,
            width=initial_left_width,
            borderwidth=0,
            highlightthickness=0,
            background=frame_bg,
        )
        left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll.grid(row=0, column=1, sticky="ns")

        left = ttk.Frame(self.left_canvas, padding=8)
        self.left_canvas_window = self.left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", self._on_left_panel_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind("<Enter>", self._bind_left_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_left_mousewheel)
        left.columnconfigure(1, weight=1)

        resize_grip = tk.Frame(self.root, width=8, cursor="sb_h_double_arrow", bg="#b9b9b9")
        resize_grip.grid(row=0, column=1, sticky="ns")
        resize_grip.bind("<ButtonPress-1>", self._start_left_resize)
        resize_grip.bind("<B1-Motion>", self._resize_left_panel)
        resize_grip.bind("<ButtonRelease-1>", self._finish_left_resize)

        plot_area = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        plot_area.grid(row=0, column=2, sticky="nsew")
        plot_area.rowconfigure(0, weight=1)
        plot_area.columnconfigure(0, weight=1)
        self.root.after_idle(self._restore_left_panel_width)

        row = 0
        ttk.Label(left, text="Fitted result folder").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Entry(left, textvariable=self.fitted_dir_var, width=42).grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1
        ttk.Button(left, text="Browse", command=self.browse_fitted_dir).grid(row=row, column=0, sticky="ew")
        ttk.Button(left, text="Load fitted CSVs", command=self.scan_results).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(left, text="Baseline-corrected TXT source").grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        row += 1
        ttk.Entry(left, textvariable=self.source_spectra_dir_var, width=42).grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1
        ttk.Button(left, text="Browse TXT source", command=self.browse_source_spectra_dir).grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        ttk.Separator(left).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(left, text="Family").grid(row=row, column=0, sticky="w")
        self.family_combo = ttk.Combobox(left, textvariable=self.family_var, state="readonly", width=30)
        self.family_combo.grid(row=row, column=1, sticky="ew")
        self.family_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_filter_changed())
        row += 1
        ttk.Label(left, text="UP/DOWN folder").grid(row=row, column=0, sticky="w")
        self.sequence_combo = ttk.Combobox(left, textvariable=self.sequence_var, state="readonly", width=30)
        self.sequence_combo.grid(row=row, column=1, sticky="ew")
        self.sequence_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_tree())
        row += 1

        limits = ttk.LabelFrame(left, text="Plot controls", padding=6)
        limits.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        limits.columnconfigure(1, weight=1)
        ttk.Label(limits, text="X min").grid(row=0, column=0, sticky="w")
        self.x_min_entry = ttk.Entry(limits, textvariable=self.x_min_var, width=12)
        self.x_min_entry.grid(row=0, column=1, sticky="ew")
        self.x_min_entry.bind("<Return>", lambda _event: self.plot_selected())
        ttk.Label(limits, text="X max").grid(row=1, column=0, sticky="w")
        self.x_max_entry = ttk.Entry(limits, textvariable=self.x_max_var, width=12)
        self.x_max_entry.grid(row=1, column=1, sticky="ew")
        self.x_max_entry.bind("<Return>", lambda _event: self.plot_selected())
        x_buttons = ttk.Frame(limits)
        x_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 4))
        x_buttons.columnconfigure(0, weight=1)
        x_buttons.columnconfigure(1, weight=1)
        ttk.Button(x_buttons, text="Use current X zoom", command=self.use_current_x_zoom).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(x_buttons, text="Full 200-2000", command=self.use_default_x_range).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ttk.Label(limits, text="Scatter size").grid(row=3, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.marker_size_var, width=12).grid(row=3, column=1, sticky="ew")
        ttk.Label(limits, text="Marker edge").grid(row=4, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.marker_edge_width_var, width=12).grid(row=4, column=1, sticky="ew")
        ttk.Label(limits, text="Scatter alpha").grid(row=5, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.scatter_alpha_var, width=12).grid(row=5, column=1, sticky="ew")
        ttk.Label(limits, text="Data line width").grid(row=6, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.data_line_width_var, width=12).grid(row=6, column=1, sticky="ew")
        ttk.Label(limits, text="Fit line width").grid(row=7, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.fit_line_width_var, width=12).grid(row=7, column=1, sticky="ew")
        ttk.Label(limits, text="Component width").grid(row=8, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.component_line_width_var, width=12).grid(row=8, column=1, sticky="ew")
        ttk.Label(limits, text="Spine width").grid(row=9, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.spine_width_var, width=12).grid(row=9, column=1, sticky="ew")
        ttk.Label(limits, text="Tick width").grid(row=10, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.tick_width_var, width=12).grid(row=10, column=1, sticky="ew")
        ttk.Label(limits, text="Tick length").grid(row=11, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.tick_length_var, width=12).grid(row=11, column=1, sticky="ew")
        ttk.Label(limits, text="Legend loc").grid(row=12, column=0, sticky="w")
        self.legend_location_combo = ttk.Combobox(
            limits,
            textvariable=self.legend_location_var,
            values=[
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "center left",
                "upper center",
                "lower center",
            ],
            state="readonly",
            width=12,
        )
        self.legend_location_combo.grid(row=12, column=1, sticky="ew")
        self.legend_location_combo.bind("<<ComboboxSelected>>", lambda _event: self.plot_selected())
        ttk.Checkbutton(limits, text="Show legend", variable=self.show_legend_var, command=self.plot_selected).grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(limits, text="Show title", variable=self.show_title_var, command=self.plot_selected).grid(row=14, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(limits, text="Show total-fit uncertainty band", variable=self.show_uncertainty_var, command=self.plot_selected).grid(row=15, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(limits, text="Also shade individual peaks", variable=self.show_component_uncertainty_var, command=self.plot_selected).grid(row=16, column=0, columnspan=2, sticky="w")
        ttk.Label(limits, text="Band samples").grid(row=17, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.uncertainty_samples_var, width=12).grid(row=17, column=1, sticky="ew")
        ttk.Label(limits, text="Band alpha").grid(row=18, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self.band_alpha_var, width=12).grid(row=18, column=1, sticky="ew")
        ttk.Checkbutton(limits, text="Show fit bounds", variable=self.show_bounds_var, command=self.plot_selected).grid(row=19, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(limits, text="Hide Y tick labels", variable=self.hide_y_ticks_var, command=self.plot_selected).grid(row=20, column=0, columnspan=2, sticky="w")
        preset_buttons = ttk.Frame(limits)
        preset_buttons.grid(row=21, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        preset_buttons.columnconfigure(0, weight=1)
        preset_buttons.columnconfigure(1, weight=1)
        ttk.Button(preset_buttons, text="Single-peak preset", command=self.apply_single_peak_preset).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(preset_buttons, text="Wide preset", command=self.apply_wide_preset).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ttk.Button(limits, text="Replot selected", command=self.replot_current_view).grid(row=22, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        row += 1

        saves = ttk.LabelFrame(left, text="Save figures", padding=6)
        saves.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        saves.columnconfigure(1, weight=1)
        ttk.Label(saves, text="Output folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(saves, textvariable=self.output_dir_var, width=26).grid(row=0, column=1, sticky="ew")
        ttk.Button(saves, text="Browse", command=self.browse_output_dir).grid(row=1, column=0, sticky="ew")
        ttk.Button(saves, text="Open", command=self.open_output_dir).grid(row=1, column=1, sticky="ew")
        ttk.Checkbutton(saves, text="PNG", variable=self.png_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(saves, text="PDF", variable=self.pdf_var).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(saves, text="SVG", variable=self.svg_var).grid(row=3, column=0, sticky="w")
        ttk.Label(saves, text="DPI").grid(row=3, column=1, sticky="w", padx=(46, 0))
        ttk.Entry(saves, textvariable=self.dpi_var, width=6).grid(row=3, column=1, sticky="e")
        ttk.Label(saves, text="Width (in)").grid(row=4, column=0, sticky="w")
        self.figure_width_entry = ttk.Entry(saves, textvariable=self.figure_width_var, width=8)
        self.figure_width_entry.grid(row=4, column=1, sticky="ew")
        self.figure_width_entry.bind("<Return>", lambda _event: self.replot_current_view())
        ttk.Label(saves, text="Height (in)").grid(row=5, column=0, sticky="w")
        self.figure_height_entry = ttk.Entry(saves, textvariable=self.figure_height_var, width=8)
        self.figure_height_entry.grid(row=5, column=1, sticky="ew")
        self.figure_height_entry.bind("<Return>", lambda _event: self.replot_current_view())
        ttk.Checkbutton(
            saves,
            text="Preview uses saved size",
            variable=self.preview_export_aspect_var,
            command=self.replot_current_view,
        ).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(saves, text="Title font").grid(row=7, column=0, sticky="w")
        ttk.Entry(saves, textvariable=self.title_font_var, width=8).grid(row=7, column=1, sticky="ew")
        ttk.Label(saves, text="Axis font").grid(row=8, column=0, sticky="w")
        ttk.Entry(saves, textvariable=self.axis_label_font_var, width=8).grid(row=8, column=1, sticky="ew")
        ttk.Label(saves, text="Tick font").grid(row=9, column=0, sticky="w")
        ttk.Entry(saves, textvariable=self.tick_font_var, width=8).grid(row=9, column=1, sticky="ew")
        ttk.Label(saves, text="Legend font").grid(row=10, column=0, sticky="w")
        ttk.Entry(saves, textvariable=self.legend_font_var, width=8).grid(row=10, column=1, sticky="ew")
        ttk.Label(saves, text="Stats font").grid(row=11, column=0, sticky="w")
        ttk.Entry(saves, textvariable=self.stats_font_var, width=8).grid(row=11, column=1, sticky="ew")
        ttk.Button(saves, text="Apply preview style", command=self.replot_current_view).grid(row=12, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(saves, text="Save selected", command=self.save_selected).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(saves, text="Save all visible", command=self.save_all_visible).grid(row=14, column=0, columnspan=2, sticky="ew")
        ttk.Button(saves, text="Save settings", command=self.save_settings).grid(row=15, column=0, columnspan=2, sticky="ew")
        row += 1

        ttk.Label(left, text="Spectra to plot").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        tree_frame = ttk.Frame(left)
        tree_frame.grid(row=row, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(row, weight=1)
        cols = ("family", "seq", "temp", "file", "col", "r2", "flag")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=18, selectmode="extended")
        headings = {
            "family": "Family",
            "seq": "Folder",
            "temp": "T",
            "file": "TXT",
            "col": "Y col",
            "r2": "R2",
            "flag": "Flag",
        }
        widths = {"family": 100, "seq": 130, "temp": 45, "file": 210, "col": 55, "r2": 65, "flag": 70}
        for col in cols:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w", stretch=False)
        yscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self.plot_selected(debounce=True))
        self.tree.bind("<Double-1>", lambda _event: self.plot_selected())

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, wraplength=360).grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.fig = Figure(figsize=self.export_figure_size(), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0)
        toolbar_frame = ttk.Frame(plot_area)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("figure_leave_event", self.on_figure_leave)

    def browse_fitted_dir(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.fitted_dir_var.get() or str(APP_DIR))
        if folder:
            self.fitted_dir_var.set(folder)
            self.scan_results()

    def browse_source_spectra_dir(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.source_spectra_dir_var.get() or str(APP_DIR))
        if folder:
            self.source_spectra_dir_var.set(folder)
            self.path_cache.clear()
            self.save_settings()
            self.plot_selected()

    def browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(APP_DIR))
        if folder:
            self.output_dir_var.set(folder)

    def open_output_dir(self) -> None:
        out = Path(self.output_dir_var.get())
        out.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(out)
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def use_current_x_zoom(self) -> None:
        try:
            x_min, x_max = self.ax.get_xlim()
        except Exception as exc:
            messagebox.showerror("No active plot", f"Could not read the current X zoom:\n{exc}")
            return
        if not (math.isfinite(x_min) and math.isfinite(x_max)):
            messagebox.showerror("Invalid X zoom", "The current plot X limits are not finite.")
            return
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        self.x_min_var.set(f"{x_min:.3f}".rstrip("0").rstrip("."))
        self.x_max_var.set(f"{x_max:.3f}".rstrip("0").rstrip("."))
        self.save_settings()
        self.status_var.set(
            f"Saved current X zoom as export range: {self.x_min_var.get()} to {self.x_max_var.get()} cm-1."
        )
        self.plot_selected()

    def use_default_x_range(self) -> None:
        self.x_min_var.set("200")
        self.x_max_var.set("2000")
        self.save_settings()
        self.plot_selected()

    def apply_single_peak_preset(self) -> None:
        self.figure_width_var.set("6")
        self.figure_height_var.set("6")
        self.marker_size_var.set("85")
        self.marker_edge_width_var.set("1.8")
        self.scatter_alpha_var.set("0.95")
        self.data_line_width_var.set("1.5")
        self.fit_line_width_var.set("3.4")
        self.component_line_width_var.set("2.4")
        self.title_font_var.set("16")
        self.axis_label_font_var.set("28")
        self.tick_font_var.set("23")
        self.legend_font_var.set("18")
        self.stats_font_var.set("12")
        self.spine_width_var.set("2.2")
        self.tick_width_var.set("2.2")
        self.tick_length_var.set("8")
        self.legend_location_var.set("upper left")
        self.show_title_var.set(False)
        self.hide_y_ticks_var.set(False)
        self.preview_export_aspect_var.set(True)
        self.save_settings()
        self.replot_current_view()

    def apply_wide_preset(self) -> None:
        self.figure_width_var.set("10")
        self.figure_height_var.set("5.8")
        self.marker_size_var.set("34")
        self.marker_edge_width_var.set("1.3")
        self.scatter_alpha_var.set("0.92")
        self.data_line_width_var.set("1.2")
        self.fit_line_width_var.set("2.8")
        self.component_line_width_var.set("1.9")
        self.title_font_var.set("18")
        self.axis_label_font_var.set("20")
        self.tick_font_var.set("16")
        self.legend_font_var.set("12")
        self.stats_font_var.set("12")
        self.spine_width_var.set("1.3")
        self.tick_width_var.set("1.3")
        self.tick_length_var.set("5")
        self.legend_location_var.set("upper right")
        self.show_title_var.set(True)
        self.hide_y_ticks_var.set(True)
        self.preview_export_aspect_var.set(True)
        self.save_settings()
        self.replot_current_view()

    def scan_results(self) -> None:
        fitted_dir = Path(self.fitted_dir_var.get())
        if not fitted_dir.exists():
            messagebox.showerror("Missing fitted folder", f"Folder not found:\n{fitted_dir}")
            return

        csv_paths = sorted(
            p
            for p in fitted_dir.rglob("*_long_results.csv")
            if "_checkpoints" not in {part.lower() for part in p.parts}
        )
        if not csv_paths:
            messagebox.showerror("No fitted CSVs", f"No *_long_results.csv files found in:\n{fitted_dir}")
            return

        frames = []
        required = set(SPECTRUM_GROUP_COLS + ["peak_id", "height", "position_cm-1", "width_fwhm_cm-1"])
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                messagebox.showwarning("CSV skipped", f"Could not read:\n{csv_path}\n\n{exc}")
                continue
            missing = required - set(df.columns)
            if missing:
                messagebox.showwarning("CSV skipped", f"Missing columns in:\n{csv_path}\n\n{sorted(missing)}")
                continue
            frames.append(df)

        if not frames:
            messagebox.showerror("No usable results", "No readable fitted long-result CSV had the expected columns.")
            return

        self.results_df = pd.concat(frames, ignore_index=True)
        for col in ["temperature_K", "spectrum_in_file", "y_column_number", "r_squared"]:
            if col in self.results_df.columns:
                self.results_df[col] = pd.to_numeric(self.results_df[col], errors="coerce")

        summary_cols = SPECTRUM_GROUP_COLS + ["quality_flag", "r_squared", "rmse", "n_points_fit"]
        available_summary = [col for col in summary_cols if col in self.results_df.columns]
        records = self.results_df[available_summary].drop_duplicates(subset=SPECTRUM_GROUP_COLS).copy()
        records["_family_order"] = records["family"].map(family_sort_index)
        records = records.sort_values(
            ["_family_order", "sequence", "temperature_K", "file_name", "y_column_number"],
            kind="mergesort",
        )
        self.records_df = records.reset_index(drop=True)

        family_labels = ["All families"]
        available_families = list(dict.fromkeys(records["family"].astype(str)))
        available_families = sorted(available_families, key=family_sort_index)
        family_labels.extend(family_label(name) for name in available_families)
        self.family_combo["values"] = family_labels
        if self.family_var.get() not in family_labels:
            self.family_var.set(family_labels[1] if len(family_labels) > 1 else "All families")

        self.on_filter_changed(select_first=True)
        self.save_settings()
        self.status_var.set(f"Loaded {len(self.records_df)} fitted spectra columns from {len(csv_paths)} family CSVs.")

    def label_to_family_name(self, label: str) -> Optional[str]:
        if label == "All families":
            return None
        for key, pretty in SAMPLE_LABELS.items():
            if pretty == label:
                return KEY_TO_FAMILY.get(key)
        return label.replace(" ", "_")

    def on_filter_changed(self, select_first: bool = False) -> None:
        if self.records_df.empty:
            return

        family_name = self.label_to_family_name(self.family_var.get())
        records = self.records_df
        if family_name:
            records = records[records["family"].astype(str) == family_name]

        sequences = ["All sequences"] + sorted(records["sequence"].astype(str).dropna().unique().tolist())
        self.sequence_combo["values"] = sequences
        if self.sequence_var.get() not in sequences:
            self.sequence_var.set("All sequences")
        self.refresh_tree(select_first=select_first)

    def refresh_tree(self, select_first: bool = True) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.iid_to_key.clear()

        if self.records_df.empty:
            self.visible_records = pd.DataFrame()
            return

        records = self.records_df.copy()
        family_name = self.label_to_family_name(self.family_var.get())
        if family_name:
            records = records[records["family"].astype(str) == family_name]
        if self.sequence_var.get() != "All sequences":
            records = records[records["sequence"].astype(str) == self.sequence_var.get()]

        self.visible_records = records.reset_index(drop=True)
        for idx, row in self.visible_records.iterrows():
            key = SpectrumKey.from_row(row)
            iid = str(idx)
            self.iid_to_key[iid] = key
            temp = f"{key.temperature_K:g}K"
            r2 = parse_float(row.get("r_squared", np.nan), np.nan)
            r2_txt = "" if not math.isfinite(r2) else f"{r2:.4f}"
            flag = str(row.get("quality_flag", ""))
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    family_label(key.family),
                    key.sequence,
                    temp,
                    key.file_name,
                    f"{key.y_column_number}",
                    r2_txt,
                    flag,
                ),
            )

        if select_first and self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)
            self.plot_selected()
        else:
            self.status_var.set(f"{len(self.visible_records)} spectra columns visible.")

    def get_selected_keys(self) -> List[SpectrumKey]:
        selected = []
        for iid in self.tree.selection():
            key = self.iid_to_key.get(iid)
            if key is not None:
                selected.append(key)
        return selected

    def resolve_spectrum_path(self, key: SpectrumKey) -> Path:
        source_root = Path(self.source_spectra_dir_var.get())
        cache_key = f"{source_root}|{key.file_path}"
        if cache_key in self.path_cache and self.path_cache[cache_key].exists():
            return self.path_cache[cache_key]

        rel = Path(key.file_path)
        candidates = [
            source_root / rel,
            source_root / key.family / key.sequence / key.file_name,
            APP_DIR / rel,
            APP_DIR / key.family / key.sequence / key.file_name,
            Path(self.fitted_dir_var.get()).parent / rel,
            Path(self.fitted_dir_var.get()).parent / key.family / key.sequence / key.file_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                self.path_cache[cache_key] = candidate
                return candidate

        search_roots = [source_root / key.family, APP_DIR / key.family]
        for family_root in search_roots:
            if not family_root.exists():
                continue
            matches = [
                match
                for match in family_root.rglob(key.file_name)
                if " - Copy" not in str(match.relative_to(family_root))
            ]
            if matches:
                best = sorted(matches, key=lambda p: len(p.parts))[0]
                self.path_cache[cache_key] = best
                return best

        raise FileNotFoundError(
            "Could not find the TXT spectrum for:\n"
            f"{key.file_path}\n\n"
            f"Checked baseline-corrected source first:\n{source_root}"
        )

    def peak_rows_for_key(self, key: SpectrumKey) -> pd.DataFrame:
        df = self.results_df
        mask = (
            (df["family"].astype(str) == key.family)
            & (df["sequence"].astype(str) == key.sequence)
            & (pd.to_numeric(df["temperature_K"], errors="coerce") == key.temperature_K)
            & (df["file_path"].astype(str) == key.file_path)
            & (pd.to_numeric(df["spectrum_in_file"], errors="coerce") == key.spectrum_in_file)
            & (pd.to_numeric(df["y_column_number"], errors="coerce") == key.y_column_number)
        )
        peak_rows = df[mask].copy()
        order_map = {peak: idx for idx, peak in enumerate(PEAK_ORDER)}
        peak_rows["_peak_order"] = peak_rows["peak_id"].map(lambda x: order_map.get(str(x), 999))
        return peak_rows.sort_values("_peak_order")

    def load_xy_for_key(self, key: SpectrumKey) -> Tuple[np.ndarray, np.ndarray, Path]:
        path = self.resolve_spectrum_path(key)
        x, y_matrix, _y_names = read_txt_spectra(path)
        y_idx = key.y_column_number - 2
        if y_idx < 0 or y_idx >= y_matrix.shape[1]:
            raise IndexError(
                f"{path.name} has {y_matrix.shape[1]} Y columns, but fitted result asks for TXT column {key.y_column_number}."
            )
        y = y_matrix[:, y_idx]
        finite = np.isfinite(x) & np.isfinite(y)
        return x[finite], y[finite], path

    def selected_x_mask(self, x: np.ndarray) -> np.ndarray:
        x_min, x_max = self.selected_x_range(x)
        return (x >= x_min) & (x <= x_max)

    def selected_x_range(self, x: np.ndarray) -> Tuple[float, float]:
        x_min = parse_float(self.x_min_var.get(), float(np.nanmin(x)))
        x_max = parse_float(self.x_max_var.get(), float(np.nanmax(x)))
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        return x_min, x_max

    def set_visible_y_limits(
        self,
        ax,
        y_plot: np.ndarray,
        total: np.ndarray,
        components: Dict[str, np.ndarray],
        full_band: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        arrays = [y_plot, total]
        arrays.extend(components.values())
        if full_band is not None:
            arrays.extend([full_band[0], full_band[1]])
        finite_values = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in arrays if np.asarray(arr).size])
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return
        y_min = float(np.nanmin(finite_values))
        y_max = float(np.nanmax(finite_values))
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            return
        if y_min == y_max:
            pad = max(abs(y_min) * 0.05, 1.0)
        else:
            pad = 0.08 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    def reconstruct_components(self, x: np.ndarray, peak_rows: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        components: Dict[str, np.ndarray] = {}
        total = np.zeros_like(x, dtype=float)
        for _, row in peak_rows.iterrows():
            peak = str(row["peak_id"])
            height = parse_float(row.get("height"), np.nan)
            pos = parse_float(row.get("position_cm-1"), np.nan)
            width = parse_float(row.get("width_fwhm_cm-1"), np.nan)
            if not (math.isfinite(height) and math.isfinite(pos) and math.isfinite(width) and width > 0):
                continue
            comp = lorentzian(x, height, pos, width)
            components[peak] = comp
            total += comp
        return components, total

    def smooth_fit_x(self, x_min: float, x_max: float, measured_count: int) -> np.ndarray:
        n_points = max(500, measured_count * 4)
        n_points = min(n_points, 1800)
        return np.linspace(float(x_min), float(x_max), int(n_points))

    def uncertainty_bands(
        self, x: np.ndarray, peak_rows: pd.DataFrame
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]:
        n_samples = max(20, min(500, parse_int(self.uncertainty_samples_var.get(), 300)))
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(20260623)
        individual: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        summed = np.zeros((n_samples, len(x)), dtype=float)
        have_samples = False

        for _, row in peak_rows.iterrows():
            peak = str(row["peak_id"])
            h = parse_float(row.get("height"), np.nan)
            p = parse_float(row.get("position_cm-1"), np.nan)
            w = parse_float(row.get("width_fwhm_cm-1"), np.nan)
            hs = abs(parse_float(row.get("height_std"), 0.0))
            ps = abs(parse_float(row.get("position_std_cm-1"), 0.0))
            ws = abs(parse_float(row.get("width_fwhm_std_cm-1"), 0.0))
            if not (math.isfinite(h) and math.isfinite(p) and math.isfinite(w) and w > 0):
                continue

            if hs == 0 and ps == 0 and ws == 0:
                curve = lorentzian(x, h, p, w)
                individual[peak] = (curve, curve)
                summed += curve
            else:
                h_draw = np.clip(rng.normal(h, hs, n_samples), 0.0, None)
                p_draw = rng.normal(p, ps, n_samples)
                w_draw = np.clip(rng.normal(w, ws, n_samples), max(w * 0.02, 1e-6), None)
                samples = h_draw[:, None] / (1.0 + ((x[None, :] - p_draw[:, None]) / (0.5 * w_draw[:, None])) ** 2)
                low, high = np.nanpercentile(samples, [16, 84], axis=0)
                individual[peak] = (low, high)
                summed += samples

            have_samples = True

        if not have_samples:
            return individual, None

        full_low, full_high = np.nanpercentile(summed, [16, 84], axis=0)
        return individual, (full_low, full_high)

    def draw_plot(self, ax, key: SpectrumKey, for_saving: bool = False) -> None:
        x, y, path = self.load_xy_for_key(key)
        peak_rows = self.peak_rows_for_key(key)
        x_min, x_max = self.selected_x_range(x)
        mask = self.selected_x_mask(x)
        if not np.any(mask):
            raise ValueError("The selected X range contains no data points.")

        x_plot = x[mask]
        y_plot = y[mask]
        x_fit = self.smooth_fit_x(x_min, x_max, len(x_plot))
        components, total = self.reconstruct_components(x_fit, peak_rows)
        visible_peaks = set()
        for _, row in peak_rows.iterrows():
            peak = str(row["peak_id"])
            pos = parse_float(row.get("position_cm-1"), np.nan)
            if peak in PEAK_ORDER and math.isfinite(pos) and x_min <= pos <= x_max:
                visible_peaks.add(peak)
        visible_components = {peak: comp for peak, comp in components.items() if peak in visible_peaks}

        ax.clear()
        if not for_saving:
            self.apply_preview_aspect(ax)
        ax.grid(False)
        ax.set_facecolor("white")

        marker_size = parse_float(self.marker_size_var.get(), 22.0)
        marker_edge_width = max(0.0, parse_float(self.marker_edge_width_var.get(), 1.3))
        scatter_alpha = max(0.0, min(1.0, parse_float(self.scatter_alpha_var.get(), 0.95)))
        data_lw = parse_float(self.data_line_width_var.get(), 1.1)
        fit_lw = parse_float(self.fit_line_width_var.get(), 2.5)
        comp_lw = parse_float(self.component_line_width_var.get(), 1.8)
        band_alpha = max(0.0, min(0.6, parse_float(self.band_alpha_var.get(), 0.18)))
        title_font, axis_font, tick_font, legend_font, stats_font = self.plot_font_sizes()
        spine_width = max(0.2, parse_float(self.spine_width_var.get(), 1.2))
        tick_width = max(0.2, parse_float(self.tick_width_var.get(), 1.2))
        tick_length = max(0.0, parse_float(self.tick_length_var.get(), 5.0))
        legend_location = self.legend_location_var.get() or "upper right"

        marker_style = sample_marker_style(key.family)
        marker_style["linewidths"] = marker_edge_width
        ax.plot(
            x_plot,
            y_plot,
            color="#3f3f3f",
            linewidth=data_lw,
            alpha=0.92,
            label="Baseline-corrected spectrum",
            zorder=3,
        )
        ax.scatter(
            x_plot,
            y_plot,
            s=marker_size,
            label=family_label(key.family),
            zorder=4,
            alpha=scatter_alpha,
            **marker_style,
        )

        individual_bands: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        full_band: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self.show_uncertainty_var.get():
            individual_bands, full_band = self.uncertainty_bands(x_fit, peak_rows)

        if full_band is not None:
            ax.fill_between(
                x_fit,
                full_band[0],
                full_band[1],
                color="#d62728",
                alpha=band_alpha,
                linewidth=0,
                label="Total fit 68% band",
                zorder=1,
            )

        for peak in PEAK_ORDER:
            if peak not in visible_peaks:
                continue
            comp = components.get(peak)
            if comp is None:
                continue
            color = PEAK_COLORS.get(peak, "#777777")
            band = individual_bands.get(peak)
            if band is not None and self.show_component_uncertainty_var.get():
                ax.fill_between(x_fit, band[0], band[1], color=color, alpha=band_alpha * 0.75, linewidth=0, zorder=1)
            ax.plot(
                x_fit,
                comp,
                linestyle="--",
                linewidth=comp_lw,
                color=color,
                label=PEAK_LABELS.get(peak, peak),
                zorder=10,
                dash_capstyle="round",
            )

        ax.plot(
            x_fit,
            total,
            color="#d62728",
            linewidth=fit_lw,
            label="Total Lorentzian fit",
            zorder=6,
            solid_capstyle="round",
        )

        if self.show_bounds_var.get():
            ymin, ymax = ax.get_ylim()
            for _, row in peak_rows.iterrows():
                peak = str(row["peak_id"])
                if peak not in visible_peaks:
                    continue
                color = PEAK_COLORS.get(peak, "#aaaaaa")
                xmin = parse_float(row.get("x_bound_min"), np.nan)
                xmax = parse_float(row.get("x_bound_max"), np.nan)
                pos = parse_float(row.get("position_cm-1"), np.nan)
                if math.isfinite(xmin) and math.isfinite(xmax):
                    left = max(min(xmin, xmax), float(np.nanmin(x_plot)))
                    right = min(max(xmin, xmax), float(np.nanmax(x_plot)))
                    if left < right:
                        ax.axvspan(left, right, color=color, alpha=0.08, zorder=0)
                if math.isfinite(pos):
                    ax.axvline(pos, color=color, linestyle=":", linewidth=1.0, alpha=0.75, zorder=2)
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(x_min, x_max)
        self.set_visible_y_limits(ax, y_plot, total, visible_components, full_band)

        fam = family_label(key.family)
        title = (
            f"{fam} | {key.sequence} | {key.temperature_K:g} K | "
            f"{path.name} | Column {key.y_column_number}"
        )
        ax.set_title(title if self.show_title_var.get() else "", fontsize=title_font, loc="left", pad=14)
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=axis_font)
        ax.set_ylabel("Intensity (a.u.)", fontsize=axis_font)

        if self.hide_y_ticks_var.get():
            ax.set_yticks([])
            ax.tick_params(axis="y", which="both", length=0)

        ax.tick_params(
            axis="both",
            labelsize=tick_font,
            width=tick_width,
            direction="out",
            length=tick_length,
            top=False,
            right=not self.hide_y_ticks_var.get(),
        )
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
            spine.set_color("#333333")

        handles, labels = ax.get_legend_handles_labels()
        reordered_handles: List[object] = []
        reordered_labels: List[str] = []
        visible_peak_labels = []
        for peak in PEAK_ORDER:
            if peak in visible_peaks:
                visible_peak_labels.append(PEAK_LABELS.get(peak, peak))
        desired = [
            family_label(key.family),
            "Total Lorentzian fit",
            "Total fit 68% band",
        ] + visible_peak_labels
        for wanted in desired:
            for handle, label in zip(handles, labels):
                if label == wanted and label not in reordered_labels:
                    reordered_handles.append(handle)
                    reordered_labels.append(label)
        if self.show_legend_var.get() and reordered_handles:
            ax.legend(
                reordered_handles,
                reordered_labels,
                loc=legend_location,
                frameon=False,
                fontsize=legend_font,
                ncol=1,
                borderpad=0.0,
                borderaxespad=0.35,
                handlelength=2.0,
                handletextpad=0.6,
                labelspacing=0.35,
                columnspacing=0.7,
            )

        r2 = parse_float(peak_rows["r_squared"].iloc[0] if "r_squared" in peak_rows else np.nan, np.nan)
        rmse = parse_float(peak_rows["rmse"].iloc[0] if "rmse" in peak_rows else np.nan, np.nan)
        info_parts = []
        if math.isfinite(r2):
            info_parts.append(f"R2={r2:.5f}")
        if math.isfinite(rmse):
            info_parts.append(f"RMSE={rmse:.3g}")
        if info_parts:
            ax.text(
                0.012,
                0.985,
                " | ".join(info_parts),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=stats_font,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
            )

        ax.margins(x=0.0, y=0.0)
        self.current_xy = (x_plot, y_plot)
        self.hover_annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=max(8.0, min(stats_font, 14.0)),
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#555555", alpha=0.94),
            arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.8),
            zorder=30,
        )
        self.hover_annotation.set_visible(False)
        self.hover_point = ax.scatter(
            [],
            [],
            s=max(marker_size * 2.2, 45),
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.5,
            zorder=31,
        )

    def plot_selected(self, debounce: bool = False) -> None:
        keys = self.get_selected_keys()
        if not keys:
            return
        if debounce:
            self.root.after(60, self._plot_key_if_still_selected, keys[0])
        else:
            self._plot_key(keys[0])

    def replot_current_view(self) -> None:
        selected_keys = self.get_selected_keys()
        self.on_filter_changed(select_first=False)

        restored = False
        for wanted in selected_keys:
            for iid, key in self.iid_to_key.items():
                if key == wanted:
                    self.tree.selection_set(iid)
                    self.tree.focus(iid)
                    self.tree.see(iid)
                    restored = True
                    break
            if restored:
                break

        if not restored and self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)
            self.tree.see(first)

        self.plot_selected()

    def _plot_key_if_still_selected(self, key: SpectrumKey) -> None:
        selected = self.get_selected_keys()
        if selected and selected[0] == key:
            self._plot_key(key)

    def _plot_key(self, key: SpectrumKey) -> None:
        try:
            if self.preview_export_aspect_var.get():
                self.sync_preview_figure_size()
            self.draw_plot(self.ax, key, for_saving=False)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            self.current_key = key
            fig_width, fig_height = self.export_figure_size()
            self.status_var.set(
                f"Showing {family_label(key.family)} {key.sequence} {key.temperature_K:g} K, "
                f"{key.file_name}, column {key.y_column_number}. Export size {fig_width:g} x {fig_height:g} in."
            )
            self.save_settings()
        except Exception as exc:
            self.current_xy = None
            self.status_var.set(f"Plot failed: {exc}")
            messagebox.showerror("Plot failed", f"{exc}\n\n{traceback.format_exc()}")

    def save_selected(self) -> None:
        keys = self.get_selected_keys()
        if not keys:
            messagebox.showinfo("No selection", "Select one or more spectra rows first.")
            return
        self.save_keys(keys)

    def save_all_visible(self) -> None:
        if self.visible_records.empty:
            messagebox.showinfo("No visible spectra", "No spectra rows are visible.")
            return
        keys = [SpectrumKey.from_row(row) for _, row in self.visible_records.iterrows()]
        if len(keys) > 50:
            proceed = messagebox.askyesno(
                "Save all visible?",
                f"This will save {len(keys)} spectra columns in the selected formats.\n\nContinue?",
            )
            if not proceed:
                return
        self.save_keys(keys)

    def selected_formats(self) -> List[str]:
        formats = []
        if self.png_var.get():
            formats.append("png")
        if self.pdf_var.get():
            formats.append("pdf")
        if self.svg_var.get():
            formats.append("svg")
        return formats or ["png"]

    def export_figure_size(self) -> Tuple[float, float]:
        width = parse_float(self.figure_width_var.get(), 10.0)
        height = parse_float(self.figure_height_var.get(), 5.8)
        width = min(max(width, 2.0), 30.0)
        height = min(max(height, 1.5), 30.0)
        return width, height

    def plot_font_sizes(self) -> Tuple[float, float, float, float, float]:
        title_font = min(max(parse_float(self.title_font_var.get(), 20.0), 6.0), 72.0)
        axis_font = min(max(parse_float(self.axis_label_font_var.get(), 20.0), 6.0), 72.0)
        tick_font = min(max(parse_float(self.tick_font_var.get(), 16.0), 6.0), 72.0)
        legend_font = min(max(parse_float(self.legend_font_var.get(), 12.0), 6.0), 72.0)
        stats_font = min(max(parse_float(self.stats_font_var.get(), 13.0), 6.0), 72.0)
        return title_font, axis_font, tick_font, legend_font, stats_font

    def sync_preview_figure_size(self) -> None:
        width, height = self.export_figure_size()
        self.fig.set_size_inches(width, height, forward=True)
        if hasattr(self, "canvas_widget"):
            self.canvas_widget.configure(width=int(width * self.fig.dpi), height=int(height * self.fig.dpi))

    def apply_preview_aspect(self, ax) -> None:
        try:
            ax.set_box_aspect(None)
        except Exception:
            pass

    def filename_for_key(self, key: SpectrumKey) -> str:
        temp = f"{key.temperature_K:g}K"
        x_min = parse_float(self.x_min_var.get(), float("nan"))
        x_max = parse_float(self.x_max_var.get(), float("nan"))
        if math.isfinite(x_min) and math.isfinite(x_max):
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            x_tag = f"_X{x_min:g}-{x_max:g}"
        else:
            x_tag = ""
        unique = hashlib.sha1(
            f"{key.file_path}|{key.y_column_number}|{x_tag}".encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        return safe_filename(
            f"T{temp}_Y{key.y_column_number:02d}{x_tag}_{unique}_Lorentzian",
            max_len=64,
        )

    def save_keys(self, keys: List[SpectrumKey]) -> None:
        out_root = Path(self.output_dir_var.get())
        formats = self.selected_formats()
        dpi = parse_int(self.dpi_var.get(), 300)
        fig_width, fig_height = self.export_figure_size()
        out_root.mkdir(parents=True, exist_ok=True)

        failures = []
        saved = 0
        for idx, key in enumerate(keys, start=1):
            try:
                fig = Figure(figsize=(fig_width, fig_height), dpi=100)
                ax = fig.add_subplot(111)
                self.draw_plot(ax, key, for_saving=True)
                fig.tight_layout()
                family_dir = out_root / safe_filename(key.family) / safe_filename(key.sequence)
                family_dir.mkdir(parents=True, exist_ok=True)
                base = family_dir / self.filename_for_key(key)
                for fmt in formats:
                    output_path = base.with_suffix(f".{fmt}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                    saved += 1
                self.status_var.set(f"Saved {idx}/{len(keys)} spectra columns...")
                self.root.update_idletasks()
            except Exception as exc:
                failures.append(f"{key.file_path} column {key.y_column_number}: {exc}")

        self.save_settings()
        if failures:
            messagebox.showwarning(
                "Save completed with warnings",
                f"Saved {saved} figure files.\n\nFailures:\n" + "\n".join(failures[:12]),
            )
        else:
            messagebox.showinfo("Save complete", f"Saved {saved} figure files to:\n{out_root}")
        self.status_var.set(f"Saved {saved} figure files to {out_root}.")

    def on_motion(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self.hide_hover_annotation()
            return
        if self.current_xy is None:
            self.status_var.set(f"x={event.xdata:.3f}, y={event.ydata:.3g}")
            return
        x, y = self.current_xy
        if len(x) == 0:
            return
        idx = int(np.nanargmin(np.abs(x - event.xdata)))
        nearest_x = float(x[idx])
        nearest_y = float(y[idx])
        self.status_var.set(
            f"Cursor x={event.xdata:.3f}, y={event.ydata:.3g} | nearest data x={nearest_x:.3f}, y={nearest_y:.6g}"
        )
        if self.hover_annotation is not None:
            self.hover_annotation.xy = (nearest_x, nearest_y)
            self.hover_annotation.set_text(f"x = {nearest_x:.3f} cm$^{{-1}}$\ny = {nearest_y:.6g}")
            self.hover_annotation.set_visible(True)
        if self.hover_point is not None:
            self.hover_point.set_offsets(np.asarray([[nearest_x, nearest_y]], dtype=float))
            self.hover_point.set_visible(True)
        self.canvas.draw_idle()

    def hide_hover_annotation(self) -> None:
        changed = False
        if self.hover_annotation is not None and self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            changed = True
        if self.hover_point is not None and self.hover_point.get_visible():
            self.hover_point.set_visible(False)
            changed = True
        if changed:
            self.canvas.draw_idle()

    def on_figure_leave(self, _event) -> None:
        self.hide_hover_annotation()

    def on_close(self) -> None:
        try:
            self.save_settings()
        finally:
            self._unbind_left_mousewheel()
            self.root.destroy()


def main() -> None:
    setup_paper_style()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    app = LorentzianPaperPlotterApp(root)
    root.minsize(1180, 720)
    root.mainloop()


if __name__ == "__main__":
    main()
