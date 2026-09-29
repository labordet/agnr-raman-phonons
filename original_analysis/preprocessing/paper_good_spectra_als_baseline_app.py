"""
ALS baseline-correction app for GOOD_SPECTRA Raman exports.

Purpose
-------
This GUI scans a paper-analysis root folder for the family-local GOOD_SPECTRA
exports produced after spike cleanup, for example:

    Aligned_RO_8A/GOOD_SPECTRA/Aligned_RO_8A/Spikes_Removed_UP_1
    Aligned_Au_8A/GOOD_SPECTRA/Aligned_Au_8A/Spikes_Removed_DOWN_1

It intentionally skips RAW folders and only reads folders whose names contain
"spike" and "removed". Input spectra are treated as read-only. Corrected data
are written only when the user clicks "Save corrected family", into a new
ALS_BASELINE_CORRECTED folder so the original GOOD_SPECTRA data are preserved.

Each file may contain many Y spectra: the first numeric column is X and every
remaining numeric column is baseline-corrected independently.

Run
---
    python paper_good_spectra_als_baseline_app.py

Dependencies
------------
    numpy pandas matplotlib scipy
"""

from __future__ import annotations

import importlib.util
import json
import math
import queue
import re
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _dependency_message(missing: Sequence[str]) -> str:
    return (
        "Missing required Python package(s): " + ", ".join(missing) + "\n\n"
        "Install them in the plotting environment, for example:\n"
        "    conda activate plotting\n"
        "    python -m pip install numpy pandas matplotlib scipy\n"
    )


def _check_required_dependencies() -> None:
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
    ]
    missing = [package for package, module in required if importlib.util.find_spec(module) is None]
    if importlib.util.find_spec("tkinter") is None:
        missing.append("tkinter / Python Tk support")
    if missing:
        msg = _dependency_message(missing)
        print("\n" + "=" * 78)
        print(msg)
        print("=" * 78 + "\n")
        try:
            import tkinter as _tk
            from tkinter import messagebox as _messagebox

            root = _tk.Tk()
            root.withdraw()
            _messagebox.showerror("Missing dependencies", msg)
            root.destroy()
        except Exception:
            pass
        raise SystemExit(1)


_check_required_dependencies()

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


def setup_paper_fonts() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans L"],
            "axes.titlesize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )


# ---------------------------------------------------------------------------
# User-facing constants
# ---------------------------------------------------------------------------

APP_TITLE = "GOOD_SPECTRA ALS baseline correction - READ ONLY INPUTS"
SETTINGS_FILENAME = "paper_good_spectra_als_settings.json"
SUPPORTED_EXTENSIONS = (".txt", ".csv", ".tsv", ".dat", ".asc", ".xy")
GOOD_SPECTRA_DIRNAME = "GOOD_SPECTRA"
OUTPUT_DIRNAME = "ALS_BASELINE_CORRECTED"
PAPER_FIGURE_DIRNAME = "ALS_PAPER_EXAMPLE_FIGURES"
FIGURE_FORMATS = ("png", "pdf", "svg")
SPIKE_ALL = "All spike-removed folders"

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
    "Aligned_RO_8A": "#00c8c8",
    "MIRA_Au_unaligned_8A": "#ae540b",
    "MIRA_RO_unaligned_8A": "#008686",
}

TEMPERATURE_RE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*K", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlsParams:
    lam: float = 1e6
    p: float = 0.0005
    niter: int = 10


@dataclass
class SpectrumData:
    path: Path
    x_col: str
    y_cols: List[str]
    x: np.ndarray
    y_matrix: np.ndarray

    @property
    def y_count(self) -> int:
        return len(self.y_cols)


@dataclass
class SpectrumFileRecord:
    family_name: str
    family_label: str
    sample_folder: Path
    good_family_root: Path
    spike_dir: Path
    spike_rel_dir: str
    path: Path
    load_order: int
    temperature: Optional[float]
    temperature_label: str

    @property
    def display_name(self) -> str:
        temp = self.temperature_label or "UNKNOWN"
        return f"{temp:>8} | {self.spike_rel_dir} | {self.path.name}"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def normalized_name(text: object) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def safe_filename_part(text: object) -> str:
    s = str(text).strip()
    s = re.sub(r"[<>:\"/\\|?*]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._ ") or "UNKNOWN"


def short_paper_figure_stem(rec: "SpectrumFileRecord", y_index: int) -> str:
    family = safe_filename_part(rec.family_name)
    spike = safe_filename_part(rec.spike_rel_dir)
    temp = safe_filename_part(rec.temperature_label)
    return safe_filename_part(f"{family}_{temp}_{spike}_Y{y_index + 1:02d}")[:96].strip("._ ")


def sample_marker_style(family_name: str) -> Dict[str, object]:
    color = SAMPLE_COLORS.get(family_name, "#707070")
    style: Dict[str, object] = {
        "marker": "D",
        "facecolors": color,
        "edgecolors": color,
        "linewidths": 0.9,
        "markerfacecolor": color,
        "markeredgecolor": color,
        "markeredgewidth": 0.9,
        "hollow": False,
    }
    if family_name == "Aligned_Au_3A":
        style["marker"] = "o"
    elif family_name in {"MIRA_Au_unaligned_8A", "MIRA_RO_unaligned_8A"}:
        style.update(
            {
                "marker": "D",
                "facecolors": "none",
                "edgecolors": color,
                "linewidths": 2.2,
                "markerfacecolor": "none",
                "markeredgecolor": color,
                "markeredgewidth": 2.2,
                "hollow": True,
            }
        )
    return style


def is_supported_data_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def is_spike_removed_folder(path: Path) -> bool:
    n = normalized_name(path.name)
    return "spike" in n and "removed" in n and n != "raw"


def has_raw_part(path: Path) -> bool:
    return any(normalized_name(part) == "raw" for part in path.parts)


def extract_temperature_from_name(name: object) -> Tuple[Optional[float], str]:
    matches = list(TEMPERATURE_RE.finditer(str(name)))
    if not matches:
        return None, "UNKNOWN"
    raw = matches[-1].group(1).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None, "UNKNOWN"
    if abs(value - round(value)) < 1e-9:
        return value, f"{int(round(value))}K"
    return value, f"{value:.12g}K"


def unique_output_dir(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base.with_name(f"{base.name}_{stamp}")
    if not candidate.exists():
        return candidate
    for i in range(2, 10000):
        numbered = base.with_name(f"{base.name}_{stamp}_{i}")
        if not numbered.exists():
            return numbered
    raise FileExistsError(f"Could not create a unique output folder for {base}")


def first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                return line.strip()
    return ""


def split_preview_tokens(line: str) -> List[str]:
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    if ";" in line:
        return [part.strip() for part in line.split(";") if part.strip()]
    if "," in line and len(re.findall(r"\s+", line)) == 0:
        return [part.strip() for part in line.split(",") if part.strip()]
    return [part.strip() for part in re.split(r"\s+", line) if part.strip()]


def token_is_number(token: str, decimal: str = ".") -> bool:
    s = token.strip()
    if decimal == ",":
        s = s.replace(",", ".")
    try:
        float(s)
        return True
    except Exception:
        return False


def delimiter_for_line(line: str) -> str:
    if "\t" in line:
        return "\t"
    if ";" in line:
        return ";"
    if "," in line and len(re.findall(r"\s+", line)) == 0:
        return ","
    return r"\s+"


def read_spectrum_file(path: Path, decimal: str = ".") -> SpectrumData:
    line = first_nonempty_line(path)
    if not line:
        raise ValueError("Empty file")
    tokens = split_preview_tokens(line)
    has_header = not tokens or not all(token_is_number(tok, decimal=decimal) for tok in tokens)
    sep = delimiter_for_line(line)
    header = 0 if has_header else None
    df = pd.read_csv(path, sep=sep, engine="python", header=header, decimal=decimal)
    if header is None:
        df.columns = [f"Column {i + 1}" for i in range(len(df.columns))]
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No numeric rows found")

    numeric_cols: List[str] = []
    numeric_series: List[pd.Series] = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(str(col))
            numeric_series.append(series)

    if len(numeric_cols) < 2:
        raise ValueError("Expected first numeric column as X and at least one Y column")

    numeric_df = pd.concat(numeric_series, axis=1)
    numeric_df.columns = numeric_cols
    x_col = numeric_cols[0]
    y_cols = numeric_cols[1:]
    keep = numeric_df[x_col].notna()
    numeric_df = numeric_df.loc[keep].reset_index(drop=True)
    if numeric_df.empty:
        raise ValueError("X column contains no numeric values")

    x = numeric_df[x_col].to_numpy(dtype=float)
    y_matrix = numeric_df[y_cols].to_numpy(dtype=float)
    return SpectrumData(path=path, x_col=x_col, y_cols=y_cols, x=x, y_matrix=y_matrix)


def write_spectrum_file(path: Path, x_col: str, y_cols: Sequence[str], x: np.ndarray, y_matrix: np.ndarray, decimal: str = ".") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({x_col: x})
    for idx, col in enumerate(y_cols):
        out[str(col)] = y_matrix[:, idx]
    out.to_csv(path, sep="\t", index=False, decimal=decimal)


def crop_spectrum_to_x_range(spec: SpectrumData, x_range: Tuple[Optional[float], Optional[float]]) -> SpectrumData:
    x_start, x_end = x_range
    mask = np.isfinite(spec.x)
    if x_start is not None:
        mask &= spec.x >= x_start
    if x_end is not None:
        mask &= spec.x <= x_end
    if not np.any(mask):
        lo = "-inf" if x_start is None else f"{x_start:g}"
        hi = "inf" if x_end is None else f"{x_end:g}"
        raise ValueError(f"No X values found inside selected range {lo} to {hi}")
    return SpectrumData(
        path=spec.path,
        x_col=spec.x_col,
        y_cols=list(spec.y_cols),
        x=spec.x[mask],
        y_matrix=spec.y_matrix[mask, :],
    )


def crop_arrays_to_x_range(
    x: np.ndarray,
    arrays: Sequence[np.ndarray],
    x_range: Tuple[Optional[float], Optional[float]],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    x_start, x_end = x_range
    mask = np.isfinite(x)
    if x_start is not None:
        mask &= x >= x_start
    if x_end is not None:
        mask &= x <= x_end
    if not np.any(mask):
        lo = "-inf" if x_start is None else f"{x_start:g}"
        hi = "inf" if x_end is None else f"{x_end:g}"
        raise ValueError(f"No X values found inside figure range {lo} to {hi}")
    return x[mask], [np.asarray(arr)[mask] for arr in arrays]


def interpolate_nonfinite(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(out)
    if finite.all():
        return out
    if np.count_nonzero(finite) < 2:
        fill = float(out[finite][0]) if np.count_nonzero(finite) == 1 else 0.0
        out[~finite] = fill
        return out
    idx = np.arange(out.size)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


# ---------------------------------------------------------------------------
# ALS baseline correction
# ---------------------------------------------------------------------------


def als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.0005, niter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline for one spectrum.

    No masks are used. Non-finite values are linearly interpolated before the
    sparse solve so that one bad point does not break an entire file.
    """
    signal = interpolate_nonfinite(np.asarray(y, dtype=float))
    length = int(signal.size)
    if length < 3:
        return signal.copy()
    lam = float(lam)
    p = float(p)
    niter = int(niter)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("Lambda must be a positive number")
    if not np.isfinite(p) or p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")
    if niter < 1:
        raise ValueError("Iterations must be at least 1")

    diagonals = [np.ones(length), -2.0 * np.ones(length), np.ones(length)]
    d2 = diags(diagonals, [0, -1, -2], shape=(length, length - 2), format="csc")
    penalty: csc_matrix = (lam * d2.dot(d2.transpose())).tocsc()
    weights = np.ones(length)
    baseline = signal.copy()
    for _ in range(niter):
        weight_matrix = diags(weights, 0, shape=(length, length), format="csc")
        system = weight_matrix + penalty
        baseline = spsolve(system, weights * signal)
        weights = p * (signal > baseline) + (1.0 - p) * (signal <= baseline)
    return np.asarray(baseline, dtype=float)


def correct_spectrum_matrix(spec: SpectrumData, params: AlsParams) -> Tuple[np.ndarray, np.ndarray]:
    baselines = np.empty_like(spec.y_matrix, dtype=float)
    corrected = np.empty_like(spec.y_matrix, dtype=float)
    for idx in range(spec.y_count):
        y = spec.y_matrix[:, idx]
        baseline = als_baseline(y, lam=params.lam, p=params.p, niter=params.niter)
        baselines[:, idx] = baseline
        corrected[:, idx] = interpolate_nonfinite(y) - baseline
    return corrected, baselines


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_target_family_folders(root: Path) -> List[Path]:
    children = [p for p in root.iterdir() if p.is_dir()]
    by_norm = {normalized_name(p.name): p for p in children}
    out: List[Path] = []
    for target in TARGET_SAMPLE_FOLDERS:
        path = by_norm.get(normalized_name(target))
        if path is not None:
            out.append(path)
    return out


def good_family_roots_for_sample(sample_folder: Path) -> List[Path]:
    good_parent = sample_folder / GOOD_SPECTRA_DIRNAME
    if not good_parent.exists() or not good_parent.is_dir():
        return []

    exact = good_parent / sample_folder.name
    roots: List[Path] = []
    if exact.exists() and exact.is_dir():
        roots.append(exact)

    for child in sorted(good_parent.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or normalized_name(child.name) == "raw":
            continue
        if child not in roots:
            roots.append(child)

    if not roots:
        roots.append(good_parent)
    return roots


def discover_good_spectra(root: Path) -> Tuple[List[SpectrumFileRecord], List[str]]:
    warnings: List[str] = []
    records: List[SpectrumFileRecord] = []
    family_folders = find_target_family_folders(root)
    if not family_folders:
        warnings.append("No target sample-family folders found in the selected root.")
        return records, warnings

    seen_paths: set[Path] = set()
    order = 0
    for sample_folder in family_folders:
        family_name = sample_folder.name
        family_label = FAMILY_LABELS.get(family_name, family_name)
        good_roots = good_family_roots_for_sample(sample_folder)
        if not good_roots:
            warnings.append(f"{family_name}: no GOOD_SPECTRA folder found.")
            continue

        family_count = 0
        for good_root in good_roots:
            spike_dirs = [
                d
                for d in good_root.rglob("*")
                if d.is_dir() and is_spike_removed_folder(d) and not has_raw_part(d.relative_to(good_root))
            ]
            for spike_dir in sorted(spike_dirs, key=lambda p: str(p.relative_to(good_root)).lower()):
                try:
                    spike_rel = str(spike_dir.relative_to(good_root))
                except Exception:
                    spike_rel = spike_dir.name
                files = [p for p in spike_dir.iterdir() if is_supported_data_file(p)]
                for path in sorted(files, key=lambda p: p.name.lower()):
                    resolved = path.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    temp, temp_label = extract_temperature_from_name(path.name)
                    records.append(
                        SpectrumFileRecord(
                            family_name=family_name,
                            family_label=family_label,
                            sample_folder=sample_folder,
                            good_family_root=good_root,
                            spike_dir=spike_dir,
                            spike_rel_dir=spike_rel,
                            path=path,
                            load_order=order,
                            temperature=temp,
                            temperature_label=temp_label,
                        )
                    )
                    order += 1
                    family_count += 1
        if family_count == 0:
            warnings.append(f"{family_name}: no spike-removed spectra found below GOOD_SPECTRA.")

    records.sort(
        key=lambda r: (
            TARGET_SAMPLE_FOLDERS.index(r.family_name) if r.family_name in TARGET_SAMPLE_FOLDERS else 999,
            r.spike_rel_dir.lower(),
            math.inf if r.temperature is None else r.temperature,
            r.path.name.lower(),
        )
    )
    return records, warnings


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class GoodSpectraAlsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        setup_paper_fonts()
        self.title(APP_TITLE)
        self.geometry("1540x940")
        self.minsize(1120, 760)
        self.settings_path = Path(__file__).with_name(SETTINGS_FILENAME)
        self.settings = self._load_settings_file()
        saved_root = self._settings_text("last_root_folder", str(Path.cwd()))

        self.root_folder: Optional[Path] = Path(saved_root) if saved_root else Path.cwd()
        self.records: List[SpectrumFileRecord] = []
        self.records_by_family: Dict[str, List[SpectrumFileRecord]] = {}
        self.file_label_to_record: Dict[str, SpectrumFileRecord] = {}
        self.tree_iid_to_record: Dict[str, SpectrumFileRecord] = {}
        self.tree_iid_to_y_index: Dict[str, Optional[int]] = {}
        self.y_columns_cache: Dict[Tuple[str, str], List[str]] = {}
        self.family_params: Dict[str, AlsParams] = {}
        self.family_settings: Dict[str, Dict[str, object]] = self._settings_dict("family_settings")
        self.active_family_name: Optional[str] = None
        self.current_spec: Optional[SpectrumData] = None
        self.current_spec_decimal: str = "."
        self.worker_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.pending_preview_job: Optional[str] = None
        self.pending_paper_preview_job: Optional[str] = None
        self.hover_annotation = None
        self.hover_lines = []
        self.fit_hover_lines = []
        self.paper_hover_lines = []

        self.root_var = tk.StringVar(value=str(self.root_folder))
        self.status_var = tk.StringVar(value="Select the analysis root, then scan GOOD_SPECTRA folders.")
        self.summary_var = tk.StringVar(value="No GOOD_SPECTRA data scanned yet.")
        self.family_var = tk.StringVar(value="")
        self.spike_folder_var = tk.StringVar(value=SPIKE_ALL)
        self.file_var = tk.StringVar(value="")
        self.column_var = tk.StringVar(value="")
        self.lambda_var = tk.StringVar(value="1e6")
        self.p_var = tk.StringVar(value="0.0005")
        self.iter_var = tk.StringVar(value="10")
        self.decimal_var = tk.StringVar(value=".")
        self.x_start_var = tk.StringVar(value="40")
        self.x_end_var = tk.StringVar(value="2000")
        self.figure_x_start_var = tk.StringVar(value="80")
        self.figure_x_end_var = tk.StringVar(value="1950")
        self.save_baselines_var = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.last_preview_signature: Optional[Tuple[str, int, Optional[float], Optional[float]]] = None
        self.hover_canvas = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._try_zoom)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Button(top, text="Select analysis root", command=self.select_root).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Scan GOOD_SPECTRA", command=self.scan_root).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Preview", command=self.preview_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Preview paper example", command=self.preview_paper_example).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Save paper example figures", command=self.save_paper_example_figures).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Save corrected family", command=self.save_corrected_family).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top, text="Save settings", command=lambda: self.save_settings(show_message=True)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(top, textvariable=self.root_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 6))

        left = ttk.Frame(main, width=520)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_left_panel(left)
        self._build_plot_panel(right)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.progress = ttk.Progressbar(bottom, variable=self.progress_var, maximum=100.0)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(bottom, textvariable=self.status_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="GOOD_SPECTRA selection")
        controls.pack(fill=tk.X, padx=(0, 8), pady=(0, 8))
        c = ttk.Frame(controls)
        c.pack(fill=tk.X, padx=8, pady=8)

        self._labeled_combo(c, "Family", self.family_var, [], row=0, width=28)
        self.family_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_family_changed())
        self._labeled_combo(c, "Spike folder", self.spike_folder_var, [SPIKE_ALL], row=1, width=28)
        self.spike_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_spike_folder_changed())
        self._labeled_combo(c, "File", self.file_var, [], row=2, width=44)
        self.file_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_file_changed())
        self._labeled_combo(c, "Y spectrum", self.column_var, [], row=3, width=28)
        self.column_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_preview())

        params = ttk.LabelFrame(parent, text="ALS parameters for current family")
        params.pack(fill=tk.X, padx=(0, 8), pady=(0, 8))
        p = ttk.Frame(params)
        p.pack(fill=tk.X, padx=8, pady=8)
        self._labeled_entry(p, "Lambda", self.lambda_var, row=0, col=0, width=12)
        self._labeled_entry(p, "p", self.p_var, row=0, col=2, width=10)
        self._labeled_entry(p, "Iterations", self.iter_var, row=1, col=0, width=8)
        self._labeled_entry(p, "Decimal", self.decimal_var, row=1, col=2, width=6)
        self._labeled_entry(p, "X start", self.x_start_var, row=2, col=0, width=8)
        self._labeled_entry(p, "X end", self.x_end_var, row=2, col=2, width=8)
        self._labeled_entry(p, "Figure X start", self.figure_x_start_var, row=3, col=0, width=8)
        self._labeled_entry(p, "Figure X end", self.figure_x_end_var, row=3, col=2, width=8)
        ttk.Checkbutton(p, text="Save baseline curves too", variable=self.save_baselines_var).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(6, 2)
        )
        ttk.Label(
            p,
            text="ALS X range fits the baseline; Figure X range trims only the paper example after correction.",
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=4, sticky="ew", padx=4, pady=(6, 0))
        for var in (self.lambda_var, self.p_var, self.iter_var, self.decimal_var, self.x_start_var, self.x_end_var):
            var.trace_add("write", lambda *_args: self._schedule_preview())
        for var in (self.figure_x_start_var, self.figure_x_end_var):
            var.trace_add("write", lambda *_args: self._schedule_paper_preview())

        table_box = ttk.LabelFrame(parent, text="Scanned spike-removed files")
        table_box.pack(fill=tk.BOTH, expand=True, padx=(0, 8), pady=(0, 8))
        columns = ("family", "spike", "temp", "file")
        self.records_tree = ttk.Treeview(table_box, columns=columns, show="tree headings", height=16)
        self.records_tree.heading("#0", text="Spectrum")
        self.records_tree.column("#0", width=145, minwidth=90, stretch=False)
        headings = {"family": "Family", "spike": "Spike folder", "temp": "T", "file": "File"}
        widths = {"family": 95, "spike": 130, "temp": 58, "file": 205}
        for col in columns:
            self.records_tree.heading(col, text=headings[col])
            self.records_tree.column(col, width=widths[col], minwidth=40, stretch=(col == "file"))
        vsb = ttk.Scrollbar(table_box, orient=tk.VERTICAL, command=self.records_tree.yview)
        hsb = ttk.Scrollbar(table_box, orient=tk.HORIZONTAL, command=self.records_tree.xview)
        self.records_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.records_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_box.rowconfigure(0, weight=1)
        table_box.columnconfigure(0, weight=1)
        self.records_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_tree_selection())
        self.records_tree.bind("<Double-1>", lambda _e: self.preview_selected())

        summary = ttk.LabelFrame(parent, text="Summary")
        summary.pack(fill=tk.X, padx=(0, 8), pady=(0, 0))
        ttk.Label(summary, textvariable=self.summary_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=8)

    def _labeled_combo(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: Sequence[str], row: int, width: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=list(values), state="readonly", width=width)
        combo.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        parent.columnconfigure(1, weight=1)
        if label == "Family":
            self.family_combo = combo
        elif label == "Spike folder":
            self.spike_combo = combo
        elif label == "File":
            self.file_combo = combo
        else:
            self.column_combo = combo

    def _labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, col: int, width: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=4)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=col + 1, sticky="ew", padx=4, pady=4)
        parent.columnconfigure(col + 1, weight=1)

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        self.plot_notebook = ttk.Notebook(parent)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        self.fit_frame = ttk.Frame(self.plot_notebook)
        self.paper_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.fit_frame, text="ALS tuning preview")
        self.plot_notebook.add(self.paper_frame, text="Paper example")

        self.fig = Figure(figsize=(10.5, 7), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fit_frame)
        self._pack_canvas(self.fit_frame, self.canvas)

        self.paper_fig = Figure(figsize=(12.5, 7.2), dpi=100, facecolor="white")
        self.paper_canvas = FigureCanvasTkAgg(self.paper_fig, master=self.paper_frame)
        self._build_paper_tab_controls(self.paper_frame)
        self._pack_canvas(self.paper_frame, self.paper_canvas)

        self._draw_empty_plot("Scan GOOD_SPECTRA, choose one Y spectrum, then preview ALS baseline.")
        self._draw_empty_paper_plot("Preview a paper example after tuning ALS.")

    def _build_paper_tab_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Label(controls, text="Paper crop X start").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(controls, textvariable=self.figure_x_start_var, width=9).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(controls, text="X end").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(controls, textvariable=self.figure_x_end_var, width=9).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(controls, text="Update paper preview", command=self.preview_paper_example).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Save paper figures", command=self.save_paper_example_figures).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(
            controls,
            text="ALS is fit with the wider X start/end; this crops only the paper image after baseline correction.",
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _pack_canvas(self, parent: ttk.Frame, canvas: FigureCanvasTkAgg) -> None:
        toolbar_host = ttk.Frame(parent)
        toolbar_host.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        canvas.mpl_connect("axes_leave_event", lambda _event: self._hide_hover_annotation())
        canvas.mpl_connect("figure_leave_event", lambda _event: self._hide_hover_annotation())

    # ------------------------------------------------------------------
    # Scanning and selection
    # ------------------------------------------------------------------
    def select_root(self) -> None:
        folder = filedialog.askdirectory(title="Select the analysis root containing the sample-family folders")
        if not folder:
            return
        self.root_folder = Path(folder)
        self.root_var.set(str(self.root_folder))
        self.status_var.set("Root selected. Click Scan GOOD_SPECTRA.")
        self.save_settings(show_message=False)

    def scan_root(self) -> None:
        if self.root_folder is None:
            self.select_root()
            if self.root_folder is None:
                return
        try:
            records, warnings = discover_good_spectra(self.root_folder)
        except Exception as exc:
            messagebox.showerror("Scan failed", f"{exc}\n\n{traceback.format_exc(limit=2)}")
            return

        self.records = records
        self.records_by_family = {}
        for rec in records:
            self.records_by_family.setdefault(rec.family_name, []).append(rec)

        family_values = [FAMILY_LABELS.get(name, name) for name in TARGET_SAMPLE_FOLDERS if name in self.records_by_family]
        extras = sorted(name for name in self.records_by_family if name not in TARGET_SAMPLE_FOLDERS)
        family_values.extend(extras)
        self.family_combo.configure(values=family_values)
        if family_values:
            saved_family = self._settings_text("selected_family", "")
            if saved_family in self.records_by_family:
                self.family_var.set(FAMILY_LABELS.get(saved_family, saved_family))
            elif saved_family in family_values:
                self.family_var.set(saved_family)
            else:
                self.family_var.set(family_values[0])
            self.active_family_name = self._current_family_name()
            self._load_params_for_family(self.active_family_name)
        else:
            self.family_var.set("")
            self.active_family_name = None

        self._refresh_spike_choices()
        self._refresh_tree()
        self._refresh_file_choices(preview=False)
        self._update_summary()
        self.status_var.set(f"Scanned {len(records)} spike-removed file(s) from GOOD_SPECTRA.")
        if warnings:
            short = "\n".join(warnings[:12])
            if len(warnings) > 12:
                short += f"\n... and {len(warnings) - 12} more warning(s)."
            messagebox.showwarning("Scan warnings", short)
        if records:
            self.after(100, self.preview_selected)

    def _current_family_name(self) -> Optional[str]:
        label = self.family_var.get()
        for name, nice in FAMILY_LABELS.items():
            if label == nice or label == name:
                return name
        return label if label in self.records_by_family else None

    def _records_for_current_family(self) -> List[SpectrumFileRecord]:
        family = self._current_family_name()
        if family is None:
            return []
        return list(self.records_by_family.get(family, []))

    def _records_for_current_view(self) -> List[SpectrumFileRecord]:
        records = self._records_for_current_family()
        spike = self.spike_folder_var.get()
        if spike and spike != SPIKE_ALL:
            records = [r for r in records if r.spike_rel_dir == spike]
        return records

    def _on_family_changed(self) -> None:
        if self.active_family_name:
            self._store_params_for_family(self.active_family_name)
            self.save_settings(show_message=False)
        self.active_family_name = self._current_family_name()
        self._load_params_for_family(self.active_family_name)
        self._refresh_spike_choices()
        self._refresh_tree()
        self._refresh_file_choices(preview=True)
        self._update_summary()

    def _on_spike_folder_changed(self) -> None:
        self._refresh_tree()
        self._refresh_file_choices(preview=True)
        self._update_summary()

    def _refresh_spike_choices(self) -> None:
        records = self._records_for_current_family()
        spike_values = [SPIKE_ALL] + sorted({r.spike_rel_dir for r in records}, key=normalized_name)
        self.spike_combo.configure(values=spike_values)
        if self.spike_folder_var.get() not in spike_values:
            self.spike_folder_var.set(SPIKE_ALL)

    def _refresh_file_choices(self, preview: bool) -> None:
        records = self._records_for_current_view()
        self.file_label_to_record = {rec.display_name: rec for rec in records}
        values = list(self.file_label_to_record)
        self.file_combo.configure(values=values)
        if values:
            if self.file_var.get() not in values:
                self.file_var.set(values[0])
            self._on_file_changed(preview=preview)
        else:
            self.file_var.set("")
            self.column_var.set("")
            self.column_combo.configure(values=[])
            self.current_spec = None
            self._draw_empty_plot("No spike-removed files for this family/filter.")
            self.status_var.set("No spike-removed files for this family/filter.")

    def _on_file_changed(self, preview: bool = True) -> None:
        rec = self.file_label_to_record.get(self.file_var.get())
        self.current_spec = None
        self.current_spec_decimal = self.decimal_var.get().strip() or "."
        if rec is None:
            self.column_combo.configure(values=[])
            self.column_var.set("")
            return
        try:
            self.current_spec = read_spectrum_file(rec.path, decimal=self.current_spec_decimal)
        except Exception as exc:
            self.column_combo.configure(values=[])
            self.column_var.set("")
            self._draw_empty_plot(f"Could not read file:\n{rec.path.name}\n\n{exc}")
            self.status_var.set(f"Read failed for {rec.path.name}: {exc}")
            return
        values = [f"Y{idx + 1:02d}: {name}" for idx, name in enumerate(self.current_spec.y_cols)]
        self.column_combo.configure(values=values)
        if values and self.column_var.get() not in values:
            self.column_var.set(values[0])
        self.status_var.set(f"Loaded preview file with {self.current_spec.y_count} Y spectrum column(s): {rec.path.name}")
        if preview:
            self._schedule_preview()

    def _y_column_labels_for_record(self, rec: SpectrumFileRecord) -> List[str]:
        decimal = self.decimal_var.get().strip() or "."
        try:
            key = (str(rec.path.resolve()), decimal)
        except Exception:
            key = (str(rec.path), decimal)
        if key not in self.y_columns_cache:
            self.y_columns_cache[key] = read_spectrum_file(rec.path, decimal=decimal).y_cols
        return self.y_columns_cache[key]

    def _refresh_tree(self) -> None:
        self.records_tree.delete(*self.records_tree.get_children())
        self.tree_iid_to_record.clear()
        self.tree_iid_to_y_index.clear()
        records = self._records_for_current_view() if self._current_family_name() is not None else self.records
        for idx, rec in enumerate(records):
            parent_iid = f"row_{idx}"
            self.tree_iid_to_record[parent_iid] = rec
            self.tree_iid_to_y_index[parent_iid] = None
            self.records_tree.insert(
                "",
                tk.END,
                iid=parent_iid,
                text=rec.path.name,
                open=False,
                values=(rec.family_label, rec.spike_rel_dir, rec.temperature_label, rec.path.name),
            )
            try:
                y_cols = self._y_column_labels_for_record(rec)
            except Exception as exc:
                child_iid = f"{parent_iid}_error"
                self.tree_iid_to_record[child_iid] = rec
                self.tree_iid_to_y_index[child_iid] = None
                self.records_tree.insert(
                    parent_iid,
                    tk.END,
                    iid=child_iid,
                    text="Could not read Y columns",
                    values=("", "", "", str(exc)),
                )
                continue
            for y_index, y_label in enumerate(y_cols):
                child_iid = f"{parent_iid}_y_{y_index}"
                self.tree_iid_to_record[child_iid] = rec
                self.tree_iid_to_y_index[child_iid] = y_index
                self.records_tree.insert(
                    parent_iid,
                    tk.END,
                    iid=child_iid,
                    text=f"Y{y_index + 1:02d}",
                    values=("", "", rec.temperature_label, y_label),
                )

    def _on_tree_selection(self) -> None:
        selected = self.records_tree.selection()
        if not selected:
            return
        iid = selected[0]
        rec = self.tree_iid_to_record.get(iid)
        if rec is None:
            return
        y_index = self.tree_iid_to_y_index.get(iid)
        file_label = rec.display_name
        if file_label not in self.file_label_to_record:
            self.file_label_to_record[file_label] = rec
            current_values = list(self.file_combo.cget("values"))
            if file_label not in current_values:
                self.file_combo.configure(values=current_values + [file_label])
        self.file_var.set(file_label)
        self._on_file_changed(preview=False)
        if y_index is not None and self.current_spec is not None and y_index < self.current_spec.y_count:
            self.column_var.set(f"Y{y_index + 1:02d}: {self.current_spec.y_cols[y_index]}")
            self.status_var.set(
                f"Selected {rec.family_label} / {rec.temperature_label} / {rec.path.name} / "
                f"Y{y_index + 1:02d}: {self.current_spec.y_cols[y_index]}"
            )
            self._schedule_preview()
        elif self.current_spec is not None:
            self.status_var.set(
                f"Selected file {rec.path.name} with {self.current_spec.y_count} individual Y spectrum column(s). "
                "Expand it and select a Y row to preview a specific column."
            )

    def _update_summary(self) -> None:
        total = len(self.records)
        families = sorted({r.family_label for r in self.records})
        spike_count = len({(r.family_name, r.spike_rel_dir) for r in self.records})
        view_count = len(self._records_for_current_view())
        self.summary_var.set(
            f"Files scanned: {total}\n"
            f"Families: {', '.join(families) if families else 'none'}\n"
            f"Spike folders: {spike_count}\n"
            f"Visible files in current family/filter: {view_count}\n"
            "Input folders are read-only. Save creates a new ALS_BASELINE_CORRECTED folder."
        )

    # ------------------------------------------------------------------
    # Parameters and preview
    # ------------------------------------------------------------------
    def _parse_params(self) -> AlsParams:
        lam = float(self.lambda_var.get().strip())
        p = float(self.p_var.get().strip())
        niter = int(float(self.iter_var.get().strip()))
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("Lambda must be positive.")
        if not np.isfinite(p) or p <= 0 or p >= 1:
            raise ValueError("p must be between 0 and 1.")
        if niter < 1:
            raise ValueError("Iterations must be at least 1.")
        return AlsParams(lam=lam, p=p, niter=niter)

    def _store_params_for_family(self, family_name: Optional[str]) -> None:
        if family_name is None:
            return
        self.family_settings[family_name] = {
            "lambda": self.lambda_var.get(),
            "p": self.p_var.get(),
            "iterations": self.iter_var.get(),
            "decimal": self.decimal_var.get(),
            "x_start": self.x_start_var.get(),
            "x_end": self.x_end_var.get(),
            "figure_x_start": self.figure_x_start_var.get(),
            "figure_x_end": self.figure_x_end_var.get(),
            "save_baselines": bool(self.save_baselines_var.get()),
            "selected_spike_folder": self.spike_folder_var.get(),
            "selected_file": self.file_var.get(),
            "selected_column": self.column_var.get(),
        }
        try:
            self.family_params[family_name] = self._parse_params()
        except Exception:
            pass

    def _load_params_for_family(self, family_name: Optional[str]) -> None:
        settings = self.family_settings.get(family_name or "", {})
        params = self.family_params.get(family_name or "", AlsParams())
        self.lambda_var.set(str(settings.get("lambda", f"{params.lam:.12g}")))
        self.p_var.set(str(settings.get("p", f"{params.p:.12g}")))
        self.iter_var.set(str(settings.get("iterations", params.niter)))
        self.decimal_var.set(str(settings.get("decimal", ".")))
        self.x_start_var.set(str(settings.get("x_start", "40")))
        self.x_end_var.set(str(settings.get("x_end", "2000")))
        self.figure_x_start_var.set(str(settings.get("figure_x_start", "80")))
        self.figure_x_end_var.set(str(settings.get("figure_x_end", "1950")))
        self.save_baselines_var.set(bool(settings.get("save_baselines", True)))
        self.spike_folder_var.set(str(settings.get("selected_spike_folder", SPIKE_ALL)))
        self.file_var.set(str(settings.get("selected_file", "")))
        self.column_var.set(str(settings.get("selected_column", "")))

    def _schedule_preview(self) -> None:
        if self.pending_preview_job is not None:
            try:
                self.after_cancel(self.pending_preview_job)
            except Exception:
                pass
        self.pending_preview_job = self.after(450, self.preview_selected)

    def _schedule_paper_preview(self) -> None:
        self._store_params_for_family(self._current_family_name())
        self.save_settings(show_message=False)
        if not self.file_var.get():
            return
        if self.pending_paper_preview_job is not None:
            try:
                self.after_cancel(self.pending_paper_preview_job)
            except Exception:
                pass
        self.pending_paper_preview_job = self.after(450, self.preview_paper_example)

    def _selected_y_index(self) -> int:
        text = self.column_var.get()
        match = re.match(r"Y(\d+):", text)
        if match:
            return int(match.group(1)) - 1
        return 0

    def _parse_x_range(self) -> Tuple[Optional[float], Optional[float]]:
        return self._parse_range_vars(self.x_start_var, self.x_end_var, "ALS X range")

    def _parse_figure_x_range(self) -> Tuple[Optional[float], Optional[float]]:
        return self._parse_range_vars(self.figure_x_start_var, self.figure_x_end_var, "Figure X range")

    def _parse_range_vars(
        self,
        start_var: tk.StringVar,
        end_var: tk.StringVar,
        label: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        def parse_one(text: str) -> Optional[float]:
            s = str(text).strip()
            if not s:
                return None
            return float(s)

        x_start = parse_one(start_var.get())
        x_end = parse_one(end_var.get())
        if x_start is not None and x_end is not None and x_end <= x_start:
            raise ValueError(f"{label}: end must be larger than start.")
        return x_start, x_end

    def _capture_plot_view(self) -> Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        axes = [ax for ax in self.fig.axes if ax.has_data()]
        if not axes:
            return None
        return [(tuple(ax.get_xlim()), tuple(ax.get_ylim())) for ax in axes]

    def _restore_plot_view(self, view: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]]) -> None:
        if not view:
            return
        axes = [ax for ax in self.fig.axes if ax.has_data()]
        if len(axes) != len(view):
            return
        for ax, (xlim, ylim) in zip(axes, view):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    def preview_selected(self) -> None:
        self.pending_preview_job = None
        rec = self.file_label_to_record.get(self.file_var.get())
        if rec is None:
            return
        decimal = self.decimal_var.get().strip() or "."
        if self.current_spec is None or self.current_spec.path != rec.path or self.current_spec_decimal != decimal:
            try:
                self.current_spec = read_spectrum_file(rec.path, decimal=decimal)
                self.current_spec_decimal = decimal
            except Exception as exc:
                self._draw_empty_plot(f"Could not read file:\n{rec.path.name}\n\n{exc}")
                return
        try:
            params = self._parse_params()
            x_range = self._parse_x_range()
            y_index = min(max(self._selected_y_index(), 0), self.current_spec.y_count - 1)
            preview_spec = crop_spectrum_to_x_range(self.current_spec, x_range)
            signature = (str(rec.path.resolve()), y_index, x_range[0], x_range[1])
            saved_view = self._capture_plot_view() if self.last_preview_signature == signature else None
            x = preview_spec.x
            y = preview_spec.y_matrix[:, y_index]
            baseline = als_baseline(y, lam=params.lam, p=params.p, niter=params.niter)
            corrected = interpolate_nonfinite(y) - baseline
            self._draw_preview(rec, preview_spec.y_cols[y_index], x, y, baseline, corrected, params, x_range=x_range)
            self._restore_plot_view(saved_view)
            self.canvas.draw_idle()
            self.plot_notebook.select(self.fit_frame)
            self.last_preview_signature = signature
            self._store_params_for_family(self._current_family_name())
            self.status_var.set(
                f"Previewed {rec.family_label} / {rec.spike_rel_dir} / {rec.path.name} "
                f"with X={x[0]:.6g} to {x[-1]:.6g}, lambda={params.lam:.3g}, p={params.p:.3g}, iterations={params.niter}."
            )
        except Exception as exc:
            self.status_var.set(f"Preview failed: {exc}")

    def _current_example_data(self, trim_for_paper: bool = False):
        rec = self.file_label_to_record.get(self.file_var.get())
        if rec is None:
            raise ValueError("Select a family/file/Y spectrum first.")
        decimal = self.decimal_var.get().strip() or "."
        if self.current_spec is None or self.current_spec.path != rec.path or self.current_spec_decimal != decimal:
            self.current_spec = read_spectrum_file(rec.path, decimal=decimal)
            self.current_spec_decimal = decimal
        params = self._parse_params()
        x_range = self._parse_x_range()
        y_index = min(max(self._selected_y_index(), 0), self.current_spec.y_count - 1)
        spec_roi = crop_spectrum_to_x_range(self.current_spec, x_range)
        x = spec_roi.x
        y = spec_roi.y_matrix[:, y_index]
        baseline = als_baseline(y, lam=params.lam, p=params.p, niter=params.niter)
        corrected = interpolate_nonfinite(y) - baseline
        y_label = spec_roi.y_cols[y_index]
        if trim_for_paper:
            x, cropped = crop_arrays_to_x_range(x, (y, baseline, corrected), self._parse_figure_x_range())
            y, baseline, corrected = cropped
        return rec, y_index, y_label, x, y, baseline, corrected, params

    def preview_paper_example(self) -> None:
        self.pending_paper_preview_job = None
        try:
            rec, _y_index, y_label, x, y, baseline, corrected, params = self._current_example_data(trim_for_paper=True)
            self._draw_paper_example_preview(rec, y_label, x, y, baseline, corrected, params)
            self.plot_notebook.select(self.paper_frame)
            self.save_settings(show_message=False)
            self.status_var.set(
                f"Paper example preview: {rec.family_label}, {rec.temperature_label}, {y_label}. "
                "Use Save paper example figures to export the two figures."
            )
        except Exception as exc:
            self.status_var.set(f"Paper example preview failed: {exc}")

    def _draw_empty_plot(self, text: str) -> None:
        self.last_preview_signature = None
        self.hover_annotation = None
        self.fit_hover_lines = []
        self.hover_lines = self.fit_hover_lines + self.paper_hover_lines
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_empty_paper_plot(self, text: str) -> None:
        self.paper_hover_lines = []
        self.hover_lines = self.fit_hover_lines + self.paper_hover_lines
        self.paper_fig.clear()
        ax = self.paper_fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        self.paper_fig.tight_layout()
        self.paper_canvas.draw_idle()

    def _draw_preview(
        self,
        rec: SpectrumFileRecord,
        y_label: str,
        x: np.ndarray,
        y: np.ndarray,
        baseline: np.ndarray,
        corrected: np.ndarray,
        params: AlsParams,
        x_range: Tuple[Optional[float], Optional[float]],
    ) -> None:
        self.fig.clear()
        ax1, ax2 = self.fig.subplots(2, 1, sharex=True, height_ratios=[2.0, 1.2])
        line_spectrum, = ax1.plot(x, y, color="#1f77b4", lw=1.0, label="Spike-removed spectrum")
        line_baseline, = ax1.plot(x, baseline, color="#d62728", lw=1.7, label="ALS baseline")
        line_corrected, = ax2.plot(x, corrected, color="#111111", lw=1.0, label="Baseline corrected")
        self.fit_hover_lines = [line_spectrum, line_baseline, line_corrected]
        self.hover_lines = self.fit_hover_lines + self.paper_hover_lines
        self.hover_annotation = None
        ax1.set_ylabel("Intensity (a.u.)")
        ax2.set_ylabel("Corrected")
        ax2.set_xlabel(r"Raman Shift (cm$^{-1}$)")
        ax1.grid(False)
        ax2.grid(False)
        ax1.legend(loc="best", frameon=False)
        ax2.legend(loc="best", frameon=False)
        x_start, x_end = x_range
        if x_start is not None or x_end is not None:
            ax1.set_xlim(x[0] if x_start is None else x_start, x[-1] if x_end is None else x_end)
        title = (
            f"{rec.family_label} | {rec.spike_rel_dir} | {rec.temperature_label} | {y_label}\n"
            f"X={x[0]:.6g} to {x[-1]:.6g} | lambda={params.lam:.3g}, p={params.p:.3g}, iterations={params.niter}"
        )
        ax1.set_title(title, loc="left", fontsize=11)
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92, hspace=0.12)
        self.canvas.draw_idle()

    def _marker_step(self, x: np.ndarray) -> int:
        return 1

    def _plot_family_spectrum_line(self, ax, rec: SpectrumFileRecord, x: np.ndarray, y: np.ndarray, label: str):
        marker_style = sample_marker_style(rec.family_name)
        (line,) = ax.plot(
            x,
            y,
            color="#111111",
            lw=1.15,
            label=label,
            marker=marker_style["marker"],
            markevery=self._marker_step(x),
            markersize=5.0,
            markerfacecolor=marker_style["markerfacecolor"],
            markeredgecolor=marker_style["markeredgecolor"],
            markeredgewidth=marker_style["markeredgewidth"],
            alpha=0.98,
            zorder=3,
        )
        return line

    def _style_paper_axis(self, ax, title: str) -> None:
        ax.set_title(title, loc="left", fontsize=18, pad=10)
        ax.set_xlabel(r"Raman Shift (cm$^{-1}$)", fontsize=20)
        ax.set_ylabel("Intensity (a.u.)", fontsize=20)
        ax.grid(False)
        ax.tick_params(axis="x", which="both", direction="out", length=4, width=0.9, labelsize=16)
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color("#666666")
        ax.margins(x=0.005)

    def _draw_paper_original_axis(
        self,
        ax,
        rec: SpectrumFileRecord,
        y_label: str,
        x: np.ndarray,
        y: np.ndarray,
        baseline: np.ndarray,
        params: AlsParams,
    ) -> None:
        spectrum_line = self._plot_family_spectrum_line(ax, rec, x, y, rec.family_label)
        (baseline_line,) = ax.plot(x, baseline, color="#d62728", lw=1.55, label="ALS baseline", zorder=4)
        self._style_paper_axis(ax, "Spike-removed spectrum with ALS baseline")
        ax.legend(handles=[spectrum_line, baseline_line], frameon=False, loc="upper right", fontsize=12)
        ax.text(
            0.015,
            0.965,
            f"{rec.temperature_label} | {y_label} | lambda={params.lam:.3g}, p={params.p:.3g}, n={params.niter}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )

    def _draw_paper_corrected_axis(
        self,
        ax,
        rec: SpectrumFileRecord,
        y_label: str,
        x: np.ndarray,
        corrected: np.ndarray,
        params: AlsParams,
    ) -> None:
        spectrum_line = self._plot_family_spectrum_line(ax, rec, x, corrected, rec.family_label)
        ax.axhline(0, color="#888888", lw=0.75, alpha=0.65, zorder=1)
        self._style_paper_axis(ax, "ALS baseline-corrected spectrum")
        ax.legend(handles=[spectrum_line], frameon=False, loc="upper right", fontsize=12)
        ax.text(
            0.015,
            0.965,
            f"{rec.temperature_label} | {y_label} | lambda={params.lam:.3g}, p={params.p:.3g}, n={params.niter}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )

    def _draw_paper_example_preview(
        self,
        rec: SpectrumFileRecord,
        y_label: str,
        x: np.ndarray,
        y: np.ndarray,
        baseline: np.ndarray,
        corrected: np.ndarray,
        params: AlsParams,
    ) -> None:
        self.last_preview_signature = None
        self.hover_annotation = None
        self.paper_fig.clear()
        ax1, ax2 = self.paper_fig.subplots(2, 1, sharex=True, height_ratios=[1.0, 1.0])
        self._draw_paper_original_axis(ax1, rec, y_label, x, y, baseline, params)
        self._draw_paper_corrected_axis(ax2, rec, y_label, x, corrected, params)
        self.paper_hover_lines = list(ax1.lines) + list(ax2.lines)
        self.hover_lines = self.fit_hover_lines + self.paper_hover_lines
        self.paper_fig.subplots_adjust(left=0.08, right=0.985, bottom=0.105, top=0.92, hspace=0.32)
        self.paper_canvas.draw_idle()

    def _make_paper_figure_original(
        self,
        rec: SpectrumFileRecord,
        y_label: str,
        x: np.ndarray,
        y: np.ndarray,
        baseline: np.ndarray,
        params: AlsParams,
    ) -> Figure:
        fig = Figure(figsize=(12.5, 5.2), dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        self._draw_paper_original_axis(ax, rec, y_label, x, y, baseline, params)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.88)
        return fig

    def _make_paper_figure_corrected(
        self,
        rec: SpectrumFileRecord,
        y_label: str,
        x: np.ndarray,
        corrected: np.ndarray,
        params: AlsParams,
    ) -> Figure:
        fig = Figure(figsize=(12.5, 5.2), dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        self._draw_paper_corrected_axis(ax, rec, y_label, x, corrected, params)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.88)
        return fig

    def _hide_hover_annotation(self) -> None:
        if self.hover_annotation is not None:
            try:
                self.hover_annotation.set_visible(False)
                if self.hover_canvas is not None:
                    self.hover_canvas.draw_idle()
                else:
                    self.canvas.draw_idle()
            except Exception:
                pass

    def _format_hover_value(self, value: float) -> str:
        if not np.isfinite(value):
            return "nan"
        abs_value = abs(float(value))
        if abs_value >= 10000 or (0 < abs_value < 0.001):
            return f"{value:.4e}"
        return f"{value:.6g}"

    def _nearest_line_point(self, event) -> Optional[Tuple[object, float, float, float]]:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        best: Optional[Tuple[object, float, float, float]] = None
        for line in self.hover_lines:
            if line.axes is not event.inaxes:
                continue
            if str(line.get_label()).startswith("_"):
                continue
            xdata = np.asarray(line.get_xdata(orig=False), dtype=float)
            ydata = np.asarray(line.get_ydata(orig=False), dtype=float)
            finite = np.isfinite(xdata) & np.isfinite(ydata)
            if not np.any(finite):
                continue
            xdata = xdata[finite]
            ydata = ydata[finite]
            idx = int(np.searchsorted(xdata, float(event.xdata)))
            candidate_indices = range(max(0, idx - 3), min(xdata.size, idx + 4))
            for candidate in candidate_indices:
                px, py = event.inaxes.transData.transform((xdata[candidate], ydata[candidate]))
                dist = math.hypot(px - event.x, py - event.y)
                if best is None or dist < best[3]:
                    best = (line, float(xdata[candidate]), float(ydata[candidate]), float(dist))
        if best is None or best[3] > 18.0:
            return None
        return best

    def _on_plot_hover(self, event) -> None:
        self.hover_canvas = getattr(event, "canvas", None)
        nearest = self._nearest_line_point(event)
        if nearest is None:
            self._hide_hover_annotation()
            return
        line, x_value, y_value, _dist = nearest
        ax = line.axes
        if self.hover_annotation is None or self.hover_annotation.axes is not ax:
            if self.hover_annotation is not None:
                self.hover_annotation.set_visible(False)
            self.hover_annotation = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 14),
                textcoords="offset points",
                fontsize=9,
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#666666", lw=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                zorder=20,
            )
        label = line.get_label()
        self.hover_annotation.xy = (x_value, y_value)
        self.hover_annotation.set_text(
            f"{label}\n"
            f"x = {self._format_hover_value(x_value)} cm^-1\n"
            f"y = {self._format_hover_value(y_value)}"
        )
        self.hover_annotation.set_visible(True)
        self.status_var.set(
            f"{label}: x = {self._format_hover_value(x_value)} cm^-1, y = {self._format_hover_value(y_value)}"
        )
        if self.hover_canvas is not None:
            self.hover_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_paper_example_figures(self) -> None:
        if self.root_folder is None:
            messagebox.showinfo("No root", "Select an analysis root first.")
            return
        try:
            rec, y_index, y_label, x, y, baseline, corrected, params = self._current_example_data(trim_for_paper=True)
        except Exception as exc:
            messagebox.showerror("Cannot prepare paper example", str(exc))
            return

        out_dir = unique_output_dir(self.root_folder / PAPER_FIGURE_DIRNAME)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = short_paper_figure_stem(rec, y_index)
        fig_original = self._make_paper_figure_original(rec, y_label, x, y, baseline, params)
        fig_corrected = self._make_paper_figure_corrected(rec, y_label, x, corrected, params)
        saved: List[Path] = []
        try:
            for ext in FIGURE_FORMATS:
                path1 = out_dir / f"{stem}_with_baseline.{ext}"
                path2 = out_dir / f"{stem}_baseline_corrected.{ext}"
                fig_original.savefig(path1, dpi=300, bbox_inches="tight", facecolor="white")
                fig_corrected.savefig(path2, dpi=300, bbox_inches="tight", facecolor="white")
                saved.extend([path1, path2])
            manifest = pd.DataFrame(
                [
                    {
                        "family": rec.family_name,
                        "family_label": rec.family_label,
                        "spike_folder": rec.spike_rel_dir,
                        "input_path": str(rec.path),
                        "y_column_index": y_index + 1,
                        "y_column_label": y_label,
                        "temperature_K": rec.temperature,
                        "temperature_label": rec.temperature_label,
                        "x_start_saved": float(x[0]) if x.size else "",
                        "x_end_saved": float(x[-1]) if x.size else "",
                        "x_points": int(x.size),
                        "lambda": params.lam,
                        "p": params.p,
                        "iterations": params.niter,
                    }
                ]
            )
            manifest.to_csv(out_dir / f"{stem}_paper_example_manifest.csv", index=False)
            self._draw_paper_example_preview(rec, y_label, x, y, baseline, corrected, params)
            self.save_settings(show_message=False)
            messagebox.showinfo(
                "Saved paper example figures",
                f"Saved {len(saved)} figure file(s) plus manifest into:\n{out_dir}",
            )
            self.status_var.set(f"Saved paper example figures into {out_dir}")
        except Exception as exc:
            messagebox.showerror("Save paper example failed", f"{exc}\n\n{traceback.format_exc(limit=2)}")

    def save_corrected_family(self) -> None:
        family = self._current_family_name()
        if family is None:
            messagebox.showinfo("No family selected", "Scan GOOD_SPECTRA and select a family first.")
            return
        records = self.records_by_family.get(family, [])
        if not records:
            messagebox.showinfo("No files", "No spike-removed files are available for this family.")
            return
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A save job is already running.")
            return
        try:
            params = self._parse_params()
            x_range = self._parse_x_range()
        except Exception as exc:
            messagebox.showerror("Bad ALS / X-range settings", str(exc))
            return
        self._store_params_for_family(family)
        if self.root_folder is None:
            messagebox.showinfo("No root", "Select an analysis root first.")
            return

        out_root = unique_output_dir(self.root_folder / OUTPUT_DIRNAME)
        family_label = FAMILY_LABELS.get(family, family)
        ok = messagebox.askyesno(
            "Save corrected family",
            f"Baseline-correct all {len(records)} spike-removed file(s) for {family_label}?\n\n"
            f"Only the selected X range will be saved: {x_range[0] if x_range[0] is not None else '-inf'} "
            f"to {x_range[1] if x_range[1] is not None else 'inf'}.\n"
            f"Input GOOD_SPECTRA files will not be changed.\n"
            f"New output folder:\n{out_root}",
        )
        if not ok:
            return

        self.progress_var.set(0.0)
        self.status_var.set(f"Saving corrected spectra for {family_label}...")
        decimal = self.decimal_var.get().strip() or "."
        save_baselines = bool(self.save_baselines_var.get())
        self.worker_thread = threading.Thread(
            target=self._save_worker,
            args=(records, params, x_range, out_root, decimal, save_baselines),
            daemon=True,
        )
        self.worker_thread.start()
        self.after(100, self._poll_worker_queue)

    def _save_worker(
        self,
        records: Sequence[SpectrumFileRecord],
        params: AlsParams,
        x_range: Tuple[Optional[float], Optional[float]],
        out_root: Path,
        decimal: str,
        save_baselines: bool,
    ) -> None:
        manifest_rows: List[Dict[str, object]] = []
        errors: List[str] = []
        total = len(records)
        try:
            for idx, rec in enumerate(records, start=1):
                try:
                    spec = read_spectrum_file(rec.path, decimal=decimal)
                    spec_roi = crop_spectrum_to_x_range(spec, x_range)
                    corrected, baselines = correct_spectrum_matrix(spec_roi, params)
                    corrected_path = out_root / rec.family_name / rec.spike_rel_dir / rec.path.name
                    write_spectrum_file(corrected_path, spec_roi.x_col, spec_roi.y_cols, spec_roi.x, corrected, decimal=decimal)

                    baseline_path = ""
                    if save_baselines:
                        baseline_file = rec.path.with_name(f"{rec.path.stem}_ALS_baseline{rec.path.suffix}")
                        baseline_out = out_root / rec.family_name / "ALS_BASELINES" / rec.spike_rel_dir / baseline_file.name
                        write_spectrum_file(baseline_out, spec_roi.x_col, spec_roi.y_cols, spec_roi.x, baselines, decimal=decimal)
                        baseline_path = str(baseline_out)

                    manifest_rows.append(
                        {
                            "family": rec.family_name,
                            "family_label": rec.family_label,
                            "spike_folder": rec.spike_rel_dir,
                            "input_path": str(rec.path),
                            "corrected_output_path": str(corrected_path),
                            "baseline_output_path": baseline_path,
                            "temperature_K": rec.temperature,
                            "temperature_label": rec.temperature_label,
                            "x_column": spec_roi.x_col,
                            "x_start_requested": x_range[0],
                            "x_end_requested": x_range[1],
                            "x_start_saved": float(spec_roi.x[0]) if spec_roi.x.size else "",
                            "x_end_saved": float(spec_roi.x[-1]) if spec_roi.x.size else "",
                            "x_points_saved": int(spec_roi.x.size),
                            "y_column_count": spec_roi.y_count,
                            "y_columns": ";".join(spec_roi.y_cols),
                            "lambda": params.lam,
                            "p": params.p,
                            "iterations": params.niter,
                        }
                    )
                except Exception as exc:
                    errors.append(f"{rec.path.name}: {exc}")
                self.worker_queue.put(("progress", idx, total, rec.path.name))

            out_root.mkdir(parents=True, exist_ok=True)
            manifest_path = out_root / "als_baseline_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            self.worker_queue.put(("done", out_root, len(manifest_rows), errors))
        except Exception as exc:
            self.worker_queue.put(("error", str(exc), traceback.format_exc(limit=3)))

    def _poll_worker_queue(self) -> None:
        keep_polling = False
        while True:
            try:
                item = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "progress":
                _, idx, total, name = item
                pct = 100.0 * float(idx) / max(1.0, float(total))
                self.progress_var.set(pct)
                self.status_var.set(f"Processed {idx}/{total}: {name}")
                keep_polling = True
            elif kind == "done":
                _, out_root, saved_count, errors = item
                self.progress_var.set(100.0)
                detail = ""
                if errors:
                    detail = "\n\nWarnings:\n" + "\n".join(errors[:12])
                    if len(errors) > 12:
                        detail += f"\n... and {len(errors) - 12} more warning(s)."
                messagebox.showinfo(
                    "ALS baseline correction complete",
                    f"Saved {saved_count} corrected file(s) into:\n{out_root}\n\n"
                    "Original GOOD_SPECTRA files were not modified."
                    + detail,
                )
                self.status_var.set(f"Saved {saved_count} corrected file(s) into {out_root}")
                keep_polling = False
            elif kind == "error":
                _, message, trace = item
                self.progress_var.set(0.0)
                messagebox.showerror("Save failed", f"{message}\n\n{trace}")
                self.status_var.set(f"Save failed: {message}")
                keep_polling = False

        if keep_polling or (self.worker_thread is not None and self.worker_thread.is_alive()):
            self.after(100, self._poll_worker_queue)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _load_settings_file(self) -> Dict[str, object]:
        try:
            if self.settings_path.exists():
                with self.settings_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _settings_text(self, key: str, default: str = "") -> str:
        value = self.settings.get(key, default)
        return str(value) if value is not None else default

    def _settings_dict(self, key: str) -> Dict[str, Dict[str, object]]:
        value = self.settings.get(key, {})
        if not isinstance(value, dict):
            return {}
        out: Dict[str, Dict[str, object]] = {}
        for family_name, family_value in value.items():
            if isinstance(family_value, dict):
                out[str(family_name)] = dict(family_value)
        return out

    def save_settings(self, show_message: bool = False) -> None:
        self._store_params_for_family(self._current_family_name())
        data: Dict[str, object] = {
            "last_root_folder": str(self.root_folder) if self.root_folder else "",
            "selected_family": self._current_family_name() or "",
            "family_settings": self.family_settings,
        }
        try:
            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.settings = data
            if show_message:
                messagebox.showinfo("Settings saved", f"Saved settings to:\n{self.settings_path}")
                self.status_var.set(f"Saved settings to {self.settings_path}")
        except Exception as exc:
            if show_message:
                messagebox.showerror("Save settings failed", str(exc))
            else:
                self.status_var.set(f"Could not save settings: {exc}")

    def _on_close(self) -> None:
        self.save_settings(show_message=False)
        self.destroy()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _try_zoom(self) -> None:
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass


def main() -> None:
    app = GoodSpectraAlsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
