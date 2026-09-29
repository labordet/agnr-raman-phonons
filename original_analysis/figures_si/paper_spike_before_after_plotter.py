"""
Paper Raman spectra: before/after spike-removal filter + cascade plotter.

Purpose
-------
This app scans selected sample-family folders, pairs RAW TXT spectra with the
corresponding spike-removed TXT spectra, lets you exclude bad spectra once, and
then makes two paper-style cascade figures:

    1. RAW spectra before spike removal
    2. Spike-removed spectra after spike removal

The same excluded row is removed from both figures and from the exported
analysis-ready data. Spike-removed files are exported while preserving their
relative spike-folder structure, which is useful when a sample has several
"up / down / up" spike-removal folders.

Install
-------
    python -m pip install -r requirements.txt

or directly:
    python -m pip install numpy pandas matplotlib openpyxl

Run
---
    python paper_spike_before_after_plotter.py

Notes
-----
- The app intentionally uses the standard tkinter/ttk GUI instead of
  ttkbootstrap, so it is easier to run on a new PC.
- TXT/CSV/TSV/DAT/ASC/XY files are supported.
- For each spectrum file, the first numeric column is used as X and every
  remaining numeric column is plotted as a spectrum.
- Temperature is extracted from the number immediately before "K" in the file
  name, e.g. "..._100K.txt" -> 100 K.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _dependency_message(missing: Sequence[str]) -> str:
    return (
        "Missing required Python package(s): " + ", ".join(missing) + "\n\n"
        "Install them with:\n"
        "    python -m pip install numpy pandas matplotlib openpyxl\n\n"
        "If you are using Anaconda, first activate the environment you want, "
        "then run the install command there."
    )


def _check_required_dependencies() -> None:
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
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

# Prefer TkAgg for the GUI. In headless test environments this backend cannot
# be loaded, so keep the default backend instead of failing during import.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, StrMethodFormatter

try:
    from matplotlib import colormaps as mpl_colormaps
except Exception:  # pragma: no cover - old Matplotlib fallback
    mpl_colormaps = None


# ---------------------------------------------------------------------------
# User-facing constants
# ---------------------------------------------------------------------------

APP_TITLE = "Paper spectra before/after spike-removal plotter - FAMILY FILTER OVERLAY"
SETTINGS_FILENAME = "paper_spike_plotter_settings.json"
SUPPORTED_EXTENSIONS = (".txt", ".csv", ".tsv", ".dat", ".asc", ".xy")

TARGET_SAMPLE_FOLDERS = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "Aligned_RO_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
]

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

SHAPES = {"3A": "o", "8A": "D"}

FAMILY_PLOT_ORDER = [
    "aligned_au_3a",
    "aligned_au_8a",
    "unaligned_au_8a",
    "unaligned_ro_8a",
    "aligned_ro_8a",
]

WORK_ALL_FAMILIES = "All families"
SPIKE_FOLDER_ALL = "All spike folders"
WORK_FAMILY_CHOICES = [SAMPLE_LABELS[key] for key in FAMILY_PLOT_ORDER] + [WORK_ALL_FAMILIES]
FAMILY_FILE_STEMS = {
    "aligned_au_3a": "Aligned_Au_3A",
    "aligned_au_8a": "Aligned_Au_8A",
    "aligned_ro_8a": "Aligned_RO_8A",
    "unaligned_au_8a": "Unaligned_Au_8A",
    "unaligned_ro_8a": "Unaligned_RO_8A",
}

PLOT_NORMALIZATIONS = ["None", "Max = 1", "Min-max", "Z-score"]
PLOT_MODES = ["Overlay", "Stacked"]
SORT_MODES = ["Temperature, then family", "Family, then temperature", "Load order"]
SAVE_FORMATS = ["png", "pdf", "svg"]
PAIR_VALUE_MATCH_MIN_FRACTION = 0.70

TEMPERATURE_RE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*K", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def safe_filename_part(text: object) -> str:
    s = str(text).strip()
    s = re.sub(r"[<>:\"/\\|?*]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._ ") or "UNKNOWN"


def normalized_name(text: object) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def extract_temperature_from_name(name: object) -> Tuple[Optional[float], Optional[str]]:
    """Return the last number immediately before K in the filename."""
    matches = list(TEMPERATURE_RE.finditer(str(name)))
    if not matches:
        return None, None
    raw = matches[-1].group(1).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None, None
    if abs(value - round(value)) < 1e-9:
        label = f"{int(round(value))}K"
    else:
        label = f"{value:.12g}K"
    return value, label


def temperatures_match(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return True
    return abs(float(a) - float(b)) <= tol


def infer_family_key(folder_name: object) -> Optional[str]:
    """Infer one of the paper sample families from a folder name.

    This is deliberately conservative. The exact target folders are preferred,
    but this function lets the app still work if the user selects a sample folder
    manually.
    """
    n = normalized_name(folder_name)

    # Exact folder aliases first.
    aliases = {
        "aligned_au_3a": "aligned_au_3a",
        "aligned_au_8a": "aligned_au_8a",
        "aligned_ro_8a": "aligned_ro_8a",
        "mira_au_unaligned_8a": "unaligned_au_8a",
        "mira_ro_unaligned_8a": "unaligned_ro_8a",
        "au_unaligned_8a": "unaligned_au_8a",
        "ro_unaligned_8a": "unaligned_ro_8a",
    }
    if n in aliases:
        return aliases[n]

    # Check unaligned before aligned because "unaligned" contains "aligned".
    if "unaligned" in n and "au" in n and "8a" in n:
        return "unaligned_au_8a"
    if "unaligned" in n and "ro" in n and "8a" in n:
        return "unaligned_ro_8a"
    if "aligned" in n and "au" in n and "3a" in n:
        return "aligned_au_3a"
    if "aligned" in n and "au" in n and "8a" in n:
        return "aligned_au_8a"
    if "aligned" in n and "ro" in n and "8a" in n:
        return "aligned_ro_8a"
    return None


def thickness_from_family(family_key: str) -> str:
    return "3A" if family_key.endswith("3a") else "8A"


def family_label(family_key: str) -> str:
    return SAMPLE_LABELS.get(family_key, family_key)


def work_family_key_from_label(label: object) -> Optional[str]:
    text = str(label or "").strip()
    if not text or text == WORK_ALL_FAMILIES or normalized_name(text) in {"all", "all_families"}:
        return None
    if text in SAMPLE_LABELS:
        return text
    for key, sample_label in SAMPLE_LABELS.items():
        if text == sample_label or normalized_name(text) == normalized_name(sample_label):
            return key
    return None


def work_family_label_from_setting(value: object) -> str:
    key = work_family_key_from_label(value)
    if key is not None:
        return family_label(key)
    return WORK_ALL_FAMILIES


def sample_marker_style(family_key: str) -> Dict[str, object]:
    color = SAMPLE_COLORS.get(family_key, "#707070")
    marker = SHAPES.get(thickness_from_family(family_key), "o")
    style: Dict[str, object] = {
        "marker": marker,
        "facecolors": color,
        "edgecolors": color,
        "linewidths": 0.8,
        "markerfacecolor": color,
        "markeredgecolor": color,
        "markeredgewidth": 0.8,
        "color": color,
        "hollow": False,
    }

    if family_key == "aligned_au_3a":
        style["marker"] = "o"
    elif family_key in {"aligned_au_8a", "aligned_ro_8a"}:
        style["marker"] = "D"
    elif family_key in {"unaligned_au_8a", "unaligned_ro_8a"}:
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


def has_spike_part(path: Path, relative_to: Path) -> bool:
    try:
        parts = path.relative_to(relative_to).parts[:-1]
    except Exception:
        parts = path.parts[:-1]
    return any("spike" in part.lower() for part in parts)


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def normalized_stem_for_matching(path: Path) -> str:
    s = normalized_name(path.stem)
    s = re.sub(r"[-+]?\d+(?:_\d+)?\s*k", "temp", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(spikes?|removed|corrected|clean|cleaned|raw|processed|spectrum|spectra)\b", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def copy_file_unique(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    # If the existing file is the same source copied earlier, overwriting is OK.
    try:
        if src.resolve() == dst.resolve():
            return dst
    except Exception:
        pass

    stem = dst.stem
    suffix = dst.suffix
    for i in range(2, 10000):
        candidate = dst.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise FileExistsError(f"Could not make a unique output name for {dst}")


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
    raise FileExistsError(f"Could not make a unique output folder for {base}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpectrumRecord:
    uid: str
    load_order: int
    family_key: str
    sample_folder: Path
    raw_path: Optional[Path]
    clean_path: Optional[Path]
    clean_rel_dir: str
    temperature: Optional[float]
    temperature_label: str
    include: bool
    status: str
    match_score: float = 0.0
    value_match_fraction: Optional[float] = None
    rejected_y_indices: set[int] = field(default_factory=set)
    y_column_labels: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def thickness(self) -> str:
        return thickness_from_family(self.family_key)

    @property
    def family_label(self) -> str:
        return family_label(self.family_key)

    @property
    def raw_name(self) -> str:
        return self.raw_path.name if self.raw_path is not None else ""

    @property
    def clean_name(self) -> str:
        return self.clean_path.name if self.clean_path is not None else ""

    @property
    def has_pair(self) -> bool:
        return self.raw_path is not None and self.clean_path is not None

    def to_manifest_row(self) -> Dict[str, object]:
        return {
            "include": self.include,
            "uid": self.uid,
            "load_order": self.load_order,
            "family_key": self.family_key,
            "family_label": self.family_label,
            "thickness": self.thickness,
            "temperature_K": self.temperature,
            "temperature_label": self.temperature_label,
            "sample_folder": str(self.sample_folder),
            "raw_path": str(self.raw_path) if self.raw_path else "",
            "clean_path": str(self.clean_path) if self.clean_path else "",
            "clean_relative_folder": self.clean_rel_dir,
            "status": self.status,
            "match_score": self.match_score,
            "value_match_fraction": self.value_match_fraction,
            "rejected_y_indices": ";".join(str(i + 1) for i in sorted(self.rejected_y_indices)),
            "y_column_count": len(self.y_column_labels),
            "y_column_labels": ";".join(self.y_column_labels),
            "note": self.note,
        }


@dataclass
class SpectrumColumnData:
    x: np.ndarray
    y: np.ndarray
    y_col: str


@dataclass
class SpectrumData:
    x_col: str
    y_cols: List[str]
    curves: List[SpectrumColumnData]


@dataclass
class PlotCurve:
    record: SpectrumRecord
    x: np.ndarray
    y: np.ndarray
    y_offset: np.ndarray
    offset_index: int
    y_col: str
    y_index: int


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------


def _read_sample_lines(path: Path, max_lines: int = 30) -> List[str]:
    lines: List[str] = []
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    lines.append(line)
                    if len(lines) >= max_lines:
                        return lines
            return lines
        except Exception as exc:
            last_exc = exc
            lines = []
    if last_exc is not None:
        raise last_exc
    return lines


def _split_line(line: str, sep: str) -> List[str]:
    if sep == "WHITESPACE":
        return re.split(r"\s+", line.strip())
    return [part.strip() for part in line.split(sep)]


def _numeric_fraction(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    ok = 0
    total = 0
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        total += 1
        s_try = s.replace(",", ".")
        try:
            float(s_try)
            ok += 1
        except Exception:
            pass
    return ok / max(total, 1)


def _guess_separator(path: Path) -> str:
    lines = _read_sample_lines(path, max_lines=20)
    if not lines:
        return "WHITESPACE"
    candidates = ["\t", ";", ",", "|", "WHITESPACE"]
    best = "WHITESPACE"
    best_score = -1.0
    for sep in candidates:
        counts = []
        for line in lines:
            tokens = _split_line(line, sep)
            if len(tokens) > 1:
                counts.append(len(tokens))
        if not counts:
            score = 0.0
        else:
            median = float(np.median(counts))
            consistency = counts.count(int(round(median))) / len(counts)
            score = median * consistency
        if score > best_score:
            best_score = score
            best = sep
    return best


def _guess_header(path: Path, sep: str) -> Optional[int]:
    lines = _read_sample_lines(path, max_lines=2)
    if not lines:
        return None
    tokens = _split_line(lines[0], sep)
    return None if _numeric_fraction(tokens) >= 0.60 else 0


def _coerce_numeric_columns(df: pd.DataFrame, decimal: str = ".") -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].astype(str).str.strip()
        if decimal == ",":
            # Decimal comma with optional thousands dots.
            s_num = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            # Dot decimal with optional thousands commas.
            s_num = s.str.replace(",", "", regex=False)
        converted = pd.to_numeric(s_num, errors="coerce")
        if converted.notna().mean() >= 0.55:
            df[col] = converted
    return df


def _read_table(path: Path, decimal: str = ".") -> pd.DataFrame:
    sep = _guess_separator(path)
    header = _guess_header(path, sep)
    read_sep = r"\s+" if sep == "WHITESPACE" else sep
    kwargs = dict(
        sep=read_sep,
        header=header,
        engine="python",
        comment="#",
        decimal=decimal,
        on_bad_lines="skip",
    )
    last_exc: Optional[Exception] = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            break
        except Exception as exc:
            last_exc = exc
    else:
        assert last_exc is not None
        raise last_exc

    if df.empty:
        raise ValueError(f"{path.name}: the file loaded as an empty table")

    new_cols = []
    seen: Dict[str, int] = {}
    for i, col in enumerate(df.columns, start=1):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed") or name.isdigit():
            name = f"Column {i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        new_cols.append(name)
    df.columns = new_cols
    df = _coerce_numeric_columns(df, decimal=decimal)
    return df


def _choose_xy_columns(df: pd.DataFrame) -> Tuple[str, List[str]]:
    numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found")

    if len(numeric_cols) == 1:
        return "__index__", [numeric_cols[0]]

    x_col = numeric_cols[0]
    y_cols = [c for c in numeric_cols[1:] if c != x_col]
    if not y_cols:
        raise ValueError("No numeric Y columns found after the first X column")
    return x_col, y_cols


def read_spectrum_file(path: Path, decimal: str = ".") -> SpectrumData:
    df = _read_table(path, decimal=decimal)
    x_col, y_cols = _choose_xy_columns(df)
    if x_col == "__index__":
        base_x = np.arange(len(df), dtype=float)
    else:
        base_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)

    curves: List[SpectrumColumnData] = []
    for y_col in y_cols:
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(base_x) & np.isfinite(y)
        x = base_x[mask]
        y = y[mask]
        if x.size < 2:
            continue
        order = np.argsort(x)
        curves.append(SpectrumColumnData(x=x[order], y=y[order], y_col=str(y_col)))
    if not curves:
        raise ValueError(f"{path.name}: fewer than 2 numeric X/Y points in every Y column")
    return SpectrumData(x_col=x_col, y_cols=[c.y_col for c in curves], curves=curves)


def average_spectrum_columns(spec: SpectrumData) -> SpectrumColumnData:
    return average_spectrum_curve_list(spec.curves)


def average_spectrum_curve_list(curves: Sequence[SpectrumColumnData]) -> SpectrumColumnData:
    if not curves:
        raise ValueError("No Y columns available to average")
    if len(curves) == 1:
        curve = curves[0]
        return SpectrumColumnData(x=curve.x, y=curve.y, y_col=curve.y_col)

    ref_x = curves[0].x
    same_grid = all(
        curve.x.shape == ref_x.shape and np.allclose(curve.x, ref_x, rtol=1e-9, atol=1e-9)
        for curve in curves
    )
    if same_grid:
        y_stack = np.vstack([curve.y for curve in curves])
        return SpectrumColumnData(x=ref_x, y=np.nanmean(y_stack, axis=0), y_col=f"Average of {len(curves)} kept Y columns")

    lo = max(float(np.nanmin(curve.x)) for curve in curves if curve.x.size)
    hi = min(float(np.nanmax(curve.x)) for curve in curves if curve.x.size)
    common_x = ref_x[(ref_x >= lo) & (ref_x <= hi)]
    if common_x.size < 2:
        raise ValueError("Could not average Y columns because their X ranges do not overlap")

    y_arrays = []
    for curve in curves:
        unique_x, unique_idx = np.unique(curve.x, return_index=True)
        if unique_x.size < 2:
            continue
        y_arrays.append(np.interp(common_x, unique_x, curve.y[unique_idx]))
    if not y_arrays:
        raise ValueError("Could not average Y columns because no valid Y arrays were found")
    return SpectrumColumnData(x=common_x, y=np.nanmean(np.vstack(y_arrays), axis=0), y_col=f"Average of {len(y_arrays)} kept Y columns")


def spectrum_output_separator(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    return "\t"


def write_filtered_spectrum_file(src: Path, dst: Path, rejected_y_indices: set[int], decimal: str = ".") -> Tuple[Path, int, int]:
    df = _read_table(src, decimal=decimal)
    x_col, y_cols = _choose_xy_columns(df)
    if x_col == "__index__":
        raise ValueError(f"{src.name}: cannot save filtered copy because no numeric X column was found")
    kept_y_cols = [col for idx, col in enumerate(y_cols) if idx not in rejected_y_indices]
    if not kept_y_cols:
        raise ValueError(f"{src.name}: all Y columns were rejected")
    out_df = df[[x_col] + kept_y_cols].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_path = dst
    if out_path.exists():
        stem = out_path.stem
        suffix = out_path.suffix
        for i in range(2, 10000):
            candidate = out_path.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                out_path = candidate
                break
        else:
            raise FileExistsError(f"Could not make a unique output name for {dst}")
    out_df.to_csv(out_path, sep=spectrum_output_separator(src), index=False)
    return out_path, len(kept_y_cols), len(y_cols)


def y_column_labels_for_file(path: Optional[Path], decimal: str = ".") -> List[str]:
    if path is None:
        return []
    try:
        return read_spectrum_file(path, decimal=decimal).y_cols
    except Exception:
        return []


def raw_clean_value_match_fraction(raw_path: Path, clean_path: Path, decimal: str = ".") -> Tuple[float, str]:
    raw = read_spectrum_file(raw_path, decimal=decimal)
    clean = read_spectrum_file(clean_path, decimal=decimal)
    if len(raw.curves) != len(clean.curves):
        return 0.0, f"Y column count mismatch: RAW={len(raw.curves)}, spike-removed={len(clean.curves)}"

    same_values = 0
    total_values = 0
    for idx, (raw_curve, clean_curve) in enumerate(zip(raw.curves, clean.curves), start=1):
        if raw_curve.x.shape != clean_curve.x.shape:
            return 0.0, f"X point count mismatch in spectrum column {idx}"
        if not np.allclose(raw_curve.x, clean_curve.x, rtol=1e-9, atol=1e-9, equal_nan=False):
            return 0.0, f"X values mismatch in spectrum column {idx}"
        if raw_curve.y.shape != clean_curve.y.shape:
            return 0.0, f"Y point count mismatch in spectrum column {idx}"
        equal_y = np.isclose(raw_curve.y, clean_curve.y, rtol=1e-9, atol=1e-12, equal_nan=False)
        same_values += int(np.count_nonzero(equal_y))
        total_values += int(equal_y.size)

    if total_values <= 0:
        return 0.0, "No comparable Y values"
    fraction = same_values / total_values
    return fraction, f"{100.0 * fraction:.1f}% unchanged Y values"


# ---------------------------------------------------------------------------
# Folder scanning and pairing
# ---------------------------------------------------------------------------


def find_target_sample_folders(root: Path) -> List[Path]:
    children = [p for p in root.iterdir() if p.is_dir()]
    by_norm = {normalized_name(p.name): p for p in children}
    targets: List[Path] = []
    for name in TARGET_SAMPLE_FOLDERS:
        p = by_norm.get(normalized_name(name))
        if p is not None:
            targets.append(p)
    return targets


def scan_sample_folder(sample_folder: Path, start_order: int = 0, decimal: str = ".") -> Tuple[List[SpectrumRecord], List[str]]:
    warnings: List[str] = []
    family_key = infer_family_key(sample_folder.name)
    if family_key is None:
        family_key = normalized_name(sample_folder.name) or "unknown_family"
        warnings.append(f"Could not infer family from {sample_folder.name}; using {family_key}.")

    raw_files = sorted(
        [p for p in sample_folder.iterdir() if is_supported_data_file(p)],
        key=lambda p: p.name.lower(),
    )
    clean_files = sorted(
        [p for p in sample_folder.rglob("*") if is_supported_data_file(p) and has_spike_part(p, sample_folder)],
        key=lambda p: str(p.relative_to(sample_folder)).lower(),
    )

    if not raw_files:
        warnings.append(f"No immediate RAW TXT/data files found in {sample_folder}.")
    if not clean_files:
        warnings.append(f"No spike-removed TXT/data files found below folders containing 'Spikes' in {sample_folder}.")

    raw_info = []
    for p in raw_files:
        temp, label = extract_temperature_from_name(p.name)
        raw_info.append(
            {
                "path": p,
                "temp": temp,
                "label": label,
                "stem_key": normalized_stem_for_matching(p),
            }
        )

    exact_stem_map: Dict[str, List[dict]] = {}
    by_temp: Dict[float, List[dict]] = {}
    for info in raw_info:
        exact_stem_map.setdefault(info["stem_key"], []).append(info)
        if info["temp"] is not None:
            by_temp.setdefault(round(float(info["temp"]), 6), []).append(info)

    used_by_group: Dict[str, set[Path]] = {}
    value_match_cache: Dict[Tuple[Path, Path], Tuple[float, str]] = {}
    records: List[SpectrumRecord] = []
    order = start_order

    def pick_raw_for_clean(clean_path: Path, clean_group: str, clean_temp: Optional[float]) -> Tuple[Optional[dict], float, str]:
        clean_key = normalized_stem_for_matching(clean_path)
        used = used_by_group.setdefault(clean_group, set())

        def value_match(info: dict) -> Tuple[float, str]:
            key = (info["path"], clean_path)
            if key not in value_match_cache:
                value_match_cache[key] = raw_clean_value_match_fraction(info["path"], clean_path, decimal=decimal)
            return value_match_cache[key]

        def choose_by_value(candidates: Sequence[dict], note: str) -> Tuple[Optional[dict], float, str]:
            scored = []
            for info in candidates:
                similarity = text_similarity(clean_key, info["stem_key"])
                try:
                    fraction, value_note = value_match(info)
                except Exception as exc:
                    fraction = -1.0
                    value_note = f"value check failed: {exc}"
                scored.append((fraction, similarity, info, value_note))
            if not scored:
                return None, 0.0, note
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            fraction, similarity, info, value_note = scored[0]
            score = float(fraction if fraction >= 0 else similarity)
            return info, score, f"{note}; {value_note}"

        # 1) Same normalized stem, not yet used within this spike folder.
        exact_candidates = [
            info
            for info in exact_stem_map.get(clean_key, [])
            if info["path"] not in used and temperatures_match(clean_temp, info["temp"])
        ]
        if exact_candidates:
            picked, score, note = choose_by_value(exact_candidates, "exact filename/stem match")
            if picked is not None:
                return picked, score, note

        # 2) Same temperature; choose the RAW file with the strongest value match.
        if clean_temp is not None:
            temp_candidates = [info for info in by_temp.get(round(float(clean_temp), 6), []) if info["path"] not in used]
            if temp_candidates:
                picked, score, note = choose_by_value(temp_candidates, "same temperature value match")
                if picked is not None:
                    return picked, score, note

        # 3) Closest stem globally, if reasonable.
        remaining = [info for info in raw_info if info["path"] not in used and temperatures_match(clean_temp, info["temp"])]
        if remaining and clean_temp is None:
            picked, score, note = choose_by_value(remaining, "closest filename/value match")
            if picked is not None and (score >= PAIR_VALUE_MATCH_MIN_FRACTION or text_similarity(clean_key, picked["stem_key"]) >= 0.55):
                return picked, score, note

        # 4) Reuse same-temp raw as a last resort, useful when several spike folders
        # contain independent cleaned copies of the same raw set.
        if clean_temp is not None:
            temp_candidates = by_temp.get(round(float(clean_temp), 6), [])
            if temp_candidates:
                picked, score, note = choose_by_value(temp_candidates, "same temperature reused raw value match")
                if picked is not None:
                    return picked, score, note

        return None, 0.0, "no RAW match at the same detected temperature"

    for clean_path in clean_files:
        clean_temp, clean_label = extract_temperature_from_name(clean_path.name)
        try:
            clean_group = str(clean_path.parent.relative_to(sample_folder))
        except Exception:
            clean_group = clean_path.parent.name
        raw_match, score, match_note = pick_raw_for_clean(clean_path, clean_group, clean_temp)
        raw_path: Optional[Path] = raw_match["path"] if raw_match else None
        if raw_match is not None:
            used_by_group.setdefault(clean_group, set()).add(raw_match["path"])
        raw_temp = raw_match["temp"] if raw_match else None
        raw_label = raw_match["label"] if raw_match else None
        temp = raw_temp if raw_temp is not None else clean_temp
        label = raw_label if raw_label is not None else clean_label
        if temp is None:
            label = "UNKNOWN"
        status = "PAIRED" if raw_path is not None else "MISSING RAW"
        include = raw_path is not None
        value_match_fraction: Optional[float] = None
        y_column_labels = y_column_labels_for_file(raw_path or clean_path, decimal=decimal)
        if raw_path is not None and raw_temp is not None:
            match_note = f"{match_note}; temperature from RAW filename"
        elif clean_temp is not None:
            match_note = f"{match_note}; temperature from spike-removed filename"
        if raw_path is not None:
            try:
                cache_key = (raw_path, clean_path)
                if cache_key in value_match_cache:
                    value_match_fraction, value_note = value_match_cache[cache_key]
                else:
                    value_match_fraction, value_note = raw_clean_value_match_fraction(raw_path, clean_path, decimal=decimal)
                    value_match_cache[cache_key] = (value_match_fraction, value_note)
                if value_note not in match_note:
                    match_note = f"{match_note}; {value_note}"
                if value_match_fraction < PAIR_VALUE_MATCH_MIN_FRACTION:
                    status = "PAIR CHECK WARN"
                    warnings.append(
                        f"{clean_path.name}: pair-check warning against {raw_path.name} "
                        f"({100.0 * value_match_fraction:.1f}% unchanged Y values); kept for inspection."
                    )
            except Exception as exc:
                status = "PAIR CHECK WARN"
                match_note = f"{match_note}; pair value check failed: {exc}"
                warnings.append(f"{clean_path.name}: pair value check failed against {raw_path.name}: {exc}; kept for inspection.")
        uid = "|".join(
            [
                family_key,
                safe_filename_part(clean_group),
                safe_filename_part(clean_path.stem),
                safe_filename_part(raw_path.stem if raw_path else "no_raw"),
                str(order),
            ]
        )
        records.append(
            SpectrumRecord(
                uid=uid,
                load_order=order,
                family_key=family_key,
                sample_folder=sample_folder,
                raw_path=raw_path,
                clean_path=clean_path,
                clean_rel_dir=clean_group,
                temperature=temp,
                temperature_label=label or "UNKNOWN",
                include=include,
                status=status,
                match_score=score,
                value_match_fraction=value_match_fraction,
                y_column_labels=y_column_labels,
                note=match_note,
            )
        )
        order += 1

    # Add unmatched/raw-only records for visibility. They are included only if no
    # spike-removed data exist at all for this sample folder.
    used_any = {info["path"] for used in used_by_group.values() for info_path in []}
    for used in used_by_group.values():
        used_any.update(used)
    raw_only_default_include = len(clean_files) == 0
    for info in raw_info:
        if info["path"] in used_any and clean_files:
            continue
        temp = info["temp"]
        label = info["label"] or "UNKNOWN"
        uid = "|".join([family_key, "RAW_ONLY", safe_filename_part(info["path"].stem), str(order)])
        records.append(
            SpectrumRecord(
                uid=uid,
                load_order=order,
                family_key=family_key,
                sample_folder=sample_folder,
                raw_path=info["path"],
                clean_path=None,
                clean_rel_dir="",
                temperature=temp,
                temperature_label=label,
                include=raw_only_default_include,
                status="RAW ONLY",
                match_score=0.0,
                value_match_fraction=None,
                y_column_labels=y_column_labels_for_file(info["path"], decimal=decimal),
                note="No matching spike-removed file found",
            )
        )
        order += 1

    return records, warnings


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def apply_normalization(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "None":
        return y.copy()
    if mode == "Max = 1":
        denom = np.nanmax(np.abs(y)) if y.size else np.nan
        if not np.isfinite(denom) or denom == 0:
            return y.copy()
        return y / denom
    if mode == "Min-max":
        lo = np.nanmin(y)
        hi = np.nanmax(y)
        denom = hi - lo
        if not np.isfinite(denom) or denom == 0:
            return y.copy()
        return (y - lo) / denom
    if mode == "Z-score":
        mean = np.nanmean(y)
        sd = np.nanstd(y)
        if not np.isfinite(sd) or sd == 0:
            return y.copy()
        return (y - mean) / sd
    return y.copy()


def get_temperature_cmap(name: str):
    if mpl_colormaps is not None:
        try:
            return mpl_colormaps.get_cmap(name)
        except Exception:
            pass
        try:
            return mpl_colormaps.get_cmap("turbo")
        except Exception:
            pass
    return plt.cm.get_cmap("turbo")


def parse_breaks(text: str) -> List[Tuple[float, float]]:
    text = text.strip()
    if not text:
        return []
    breaks: List[Tuple[float, float]] = []
    for part in re.split(r"[,;]", text):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*[-:]\s*([-+]?\d+(?:\.\d+)?)\s*$", part)
        if not m:
            raise ValueError(f"Could not parse break '{part}'. Use e.g. 350-1180, 1370-1540")
        lo = float(m.group(1))
        hi = float(m.group(2))
        if hi <= lo:
            raise ValueError(f"Bad break '{part}': end must be larger than start")
        breaks.append((lo, hi))
    return sorted(breaks)


def make_segments(x_start: float, x_end: float, breaks: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if x_end <= x_start:
        raise ValueError("X end must be larger than X start")
    segments: List[Tuple[float, float]] = []
    current = float(x_start)
    for lo, hi in breaks:
        lo = max(float(lo), x_start)
        hi = min(float(hi), x_end)
        if lo > current:
            segments.append((current, lo))
        current = max(current, hi)
    if current < x_end:
        segments.append((current, x_end))
    if not segments:
        segments = [(x_start, x_end)]
    return segments


def _make_formatter(fmt: str):
    if "{" in fmt and "}" in fmt:
        return StrMethodFormatter(fmt)
    return FormatStrFormatter(fmt)


def format_axis_ticks(ax, is_first_panel: bool, x_fmt: str = "{x:.0f}", y_fmt: str = "{x:.2f}") -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(_make_formatter(x_fmt))
    ax.tick_params(which="both", direction="out", length=4, width=0.8)
    ax.set_yticks([])
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)


def setup_paper_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans L"],
            "axes.titlesize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class PaperSpikePlotterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        setup_paper_fonts()

        self.title(APP_TITLE)
        self.geometry("1660x980")
        self.minsize(1180, 760)
        self.settings_path = Path(__file__).with_name(SETTINGS_FILENAME)
        self.settings = self._load_settings_file()

        self.records: List[SpectrumRecord] = []
        self.uid_to_record: Dict[str, SpectrumRecord] = {}
        self.tree_iid_to_uid: Dict[str, str] = {}
        self.tree_iid_to_y_index: Dict[str, Optional[int]] = {}
        saved_root = self._settings_text("last_root_folder", "")
        self.root_folder: Optional[Path] = Path(saved_root) if saved_root else None
        self.last_warning_lines: List[str] = []
        self.last_plotted_uids: Dict[str, set[str]] = {"raw": set(), "clean": set()}
        self.active_plot_uid: Optional[str] = None
        self.active_plot_y_index: Optional[int] = None
        self.last_output_dir: Optional[Path] = None

        # User controls.
        self.root_var = tk.StringVar(value=str(self.root_folder) if self.root_folder else "No root folder selected")
        self.status_var = tk.StringVar(value="Select the folder that contains the five sample-family folders, then click Scan.")
        self.summary_var = tk.StringVar(value="No spectra loaded.")
        self.work_family_var = tk.StringVar(value=work_family_label_from_setting(self.settings.get("selected_work_family", "")))
        self.spike_folder_var = tk.StringVar(value=SPIKE_FOLDER_ALL)
        self.show_rejected_var = tk.BooleanVar(value=self._settings_bool("show_rejected_rows", True))
        self.normalization_var = tk.StringVar(value=self._settings_text("normalization", "Max = 1"))
        self.average_y_columns_var = tk.BooleanVar(value=self._settings_bool("average_y_columns", False))
        self.plot_mode_var = tk.StringVar(value=self._settings_text("plot_mode", "Overlay"))
        self.sort_mode_var = tk.StringVar(value=self._settings_text("sort_mode", SORT_MODES[0]))
        self.cap_k_var = tk.StringVar(value=self._settings_text("cap_k", "300"))
        self.x_start_var = tk.StringVar(value=self._settings_text("x_start", "270"))
        self.x_end_var = tk.StringVar(value=self._settings_text("x_end", "1630"))
        self.breaks_var = tk.StringVar(value=self._settings_text("breaks", ""))
        self.use_x_breaks_var = tk.BooleanVar(value=self._settings_bool("use_x_breaks", False))
        self.offset_var = tk.StringVar(value=self._settings_text("offset", "0"))
        self.auto_offset_var = tk.BooleanVar(value=self._settings_bool("auto_offset", False))
        self.line_width_var = tk.StringVar(value=self._settings_text("line_width", "1.15"))
        self.marker_size_var = tk.StringVar(value=self._settings_text("scatter_size", "120"))
        self.scatter_step_var = tk.StringVar(value=self._settings_text("point_step", "1"))
        self.cmap_var = tk.StringVar(value=self._settings_text("cmap", "turbo"))
        self.decimal_var = tk.StringVar(value=self._settings_text("decimal", "."))
        self.legend_var = tk.BooleanVar(value=self._settings_bool("family_legend", True))
        self.plot_title_prefix_var = tk.StringVar(value="")
        self.raw_title_var = tk.StringVar(value=self._settings_text("raw_title", "RAW spectra before spike removal"))
        self.clean_title_var = tk.StringVar(value=self._settings_text("spike_title", "Spike-removed spectra"))
        self.include_raw_only_var = tk.BooleanVar(value=False)
        self.require_pair_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._try_zoom)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._build_top_bar()

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 6))

        left = ttk.Frame(main, width=620)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_left_panel(left)
        self._build_plot_panel(right)

        status = ttk.Label(self, textvariable=self.status_var, anchor=tk.W)
        status.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _build_top_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=10, pady=(10, 4))

        ttk.Button(bar, text="Select root folder", command=self.select_root_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Scan target folders", command=self.scan_root_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Add sample folder(s)", command=self.add_sample_folders).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(bar, text="Work on:").pack(side=tk.LEFT, padx=(8, 4))
        self.work_family_combo = ttk.Combobox(
            bar,
            textvariable=self.work_family_var,
            values=WORK_FAMILY_CHOICES,
            state="readonly",
            width=21,
        )
        self.work_family_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.work_family_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_work_family_changed())
        ttk.Label(bar, text="Spike folder:").pack(side=tk.LEFT, padx=(0, 4))
        self.spike_folder_combo = ttk.Combobox(
            bar,
            textvariable=self.spike_folder_var,
            values=[SPIKE_FOLDER_ALL],
            state="readonly",
            width=24,
        )
        self.spike_folder_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.spike_folder_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_spike_folder_changed())
        ttk.Button(bar, text="Plot RAW + SPIKE", command=self.plot_both).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Save figures", command=self.save_figures).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Save GOOD spectra", command=self.save_good_spectra).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Save settings", command=lambda: self.save_settings(show_message=True)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Clear", command=self.clear_records).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(bar, textvariable=self.root_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Scan + filter spectra")
        controls.pack(fill=tk.X, padx=(0, 8), pady=(0, 8))
        c = ttk.Frame(controls)
        c.pack(fill=tk.X, padx=8, pady=8)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="Reject selected", command=self.reject_selected_rows).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row, text="Keep selected", command=self.keep_selected_rows).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row, text="Keep all visible", command=self.keep_all_visible_rows).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row, text="Invert visible", command=self.invert_visible_rows).pack(side=tk.LEFT, padx=(0, 4))

        row2 = ttk.Frame(c)
        row2.pack(fill=tk.X, pady=(6, 2))
        ttk.Checkbutton(row2, text="Require RAW + spike pair", variable=self.require_pair_var, command=self.apply_filter_policy).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(row2, text="Allow RAW-only rows", variable=self.include_raw_only_var, command=self.apply_filter_policy).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(row2, text="Show rejected rows", variable=self.show_rejected_var, command=self._on_show_rejected_changed).pack(side=tk.LEFT, padx=(0, 12))

        tree_box = ttk.LabelFrame(parent, text="Loaded spectra pairs - expand a file to keep/reject individual Y spectra")
        tree_box.pack(fill=tk.BOTH, expand=True, padx=(0, 8), pady=(0, 8))

        columns = ("include", "family", "temp", "raw", "clean", "spike_folder", "status")
        self.records_tree = ttk.Treeview(tree_box, columns=columns, show="tree headings", selectmode="extended", height=16)
        self.records_tree.heading("#0", text="Spectrum")
        self.records_tree.column("#0", width=155, minwidth=95, anchor=tk.W, stretch=False)
        headings = {
            "include": "Keep",
            "family": "Family",
            "temp": "T",
            "raw": "RAW file",
            "clean": "Spike-removed file",
            "spike_folder": "Spike folder",
            "status": "Status",
        }
        widths = {
            "include": 44,
            "family": 94,
            "temp": 62,
            "raw": 145,
            "clean": 145,
            "spike_folder": 100,
            "status": 90,
        }
        for col in columns:
            self.records_tree.heading(col, text=headings[col])
            anchor = tk.CENTER if col in ("include", "temp", "status") else tk.W
            self.records_tree.column(col, width=widths[col], minwidth=40, anchor=anchor, stretch=(col in ("raw", "clean", "spike_folder")))

        vsb = ttk.Scrollbar(tree_box, orient=tk.VERTICAL, command=self.records_tree.yview)
        hsb = ttk.Scrollbar(tree_box, orient=tk.HORIZONTAL, command=self.records_tree.xview)
        self.records_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.records_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_box.rowconfigure(0, weight=1)
        tree_box.columnconfigure(0, weight=1)
        self.records_tree.bind("<Double-1>", lambda _e: self.toggle_selected_rows())
        self.records_tree.bind("<Delete>", lambda _e: self.reject_selected_rows())
        self.records_tree.bind("<<TreeviewSelect>>", lambda _e: self._update_status_from_selection())

        options = ttk.LabelFrame(parent, text="Paper cascade style")
        options.pack(fill=tk.X, padx=(0, 8), pady=(0, 8))
        o = ttk.Frame(options)
        o.pack(fill=tk.X, padx=8, pady=8)

        self._labeled_combobox(o, "Normalize", self.normalization_var, PLOT_NORMALIZATIONS, row=0, col=0, width=12)
        self._labeled_combobox(o, "Sort", self.sort_mode_var, SORT_MODES, row=0, col=2, width=22)
        self._labeled_entry(o, "Cap K", self.cap_k_var, row=1, col=0, width=8)
        self._labeled_entry(o, "Decimal", self.decimal_var, row=1, col=2, width=8)
        self._labeled_entry(o, "X start", self.x_start_var, row=2, col=0, width=8)
        self._labeled_entry(o, "X end", self.x_end_var, row=2, col=2, width=8)
        self._labeled_entry(o, "Breaks", self.breaks_var, row=3, col=0, width=22)
        ttk.Checkbutton(o, text="Use X breaks", variable=self.use_x_breaks_var).grid(row=3, column=2, sticky="w", padx=4, pady=3)
        self._labeled_combobox(o, "Plot mode", self.plot_mode_var, PLOT_MODES, row=4, col=0, width=12)
        ttk.Checkbutton(o, text="Auto offset", variable=self.auto_offset_var).grid(row=4, column=2, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(o, text="Average Y columns per file", variable=self.average_y_columns_var).grid(row=4, column=3, sticky="w", padx=4, pady=3)
        self._labeled_entry(o, "Offset", self.offset_var, row=5, col=0, width=8)
        self._labeled_entry(o, "LW", self.line_width_var, row=5, col=2, width=8)
        self._labeled_entry(o, "Scatter", self.marker_size_var, row=6, col=0, width=8)
        self._labeled_entry(o, "Point step", self.scatter_step_var, row=6, col=2, width=8)
        self._labeled_entry(o, "Cmap", self.cmap_var, row=7, col=0, width=10)
        ttk.Checkbutton(o, text="Family legend", variable=self.legend_var).grid(row=7, column=2, sticky="w", padx=4, pady=3)
        ttk.Label(o, text="Raw title").grid(row=8, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(o, textvariable=self.raw_title_var, width=42).grid(row=8, column=1, columnspan=3, sticky="ew", padx=4, pady=3)
        ttk.Label(o, text="Spike title").grid(row=9, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(o, textvariable=self.clean_title_var, width=42).grid(row=9, column=1, columnspan=3, sticky="ew", padx=4, pady=3)
        o.columnconfigure(1, weight=1)
        o.columnconfigure(3, weight=1)

        summary_box = ttk.LabelFrame(parent, text="Summary")
        summary_box.pack(fill=tk.X, padx=(0, 8), pady=(0, 0))
        ttk.Label(summary_box, textvariable=self.summary_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=8)

    def _labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, col: int, width: int = 10, colspan: int = 1) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=col + 1, columnspan=colspan, sticky="ew", padx=4, pady=3)

    def _labeled_combobox(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: Sequence[str], row: int, col: int, width: int = 12) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
        ttk.Combobox(parent, textvariable=variable, values=list(values), state="readonly", width=width).grid(row=row, column=col + 1, sticky="ew", padx=4, pady=3)

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.raw_frame = ttk.Frame(self.notebook)
        self.clean_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_frame, text="FIG 1 — RAW before spike removal")
        self.notebook.add(self.clean_frame, text="FIG 2 — Spike removed")

        self.raw_fig = Figure(figsize=(12.5, 6), dpi=100, facecolor="white")
        self.clean_fig = Figure(figsize=(12.5, 6), dpi=100, facecolor="white")
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=self.raw_frame)
        self.clean_canvas = FigureCanvasTkAgg(self.clean_fig, master=self.clean_frame)

        for frame, fig, canvas, source in (
            (self.raw_frame, self.raw_fig, self.raw_canvas, "raw"),
            (self.clean_frame, self.clean_fig, self.clean_canvas, "clean"),
        ):
            toolbar_host = ttk.Frame(frame)
            toolbar_host.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_host, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            canvas_widget.bind("<Delete>", lambda _e: self.reject_active_plot_curve())
            canvas_widget.bind("<Return>", lambda _e: self.toggle_active_plot_curve())
            canvas_widget.bind("<space>", lambda _e: self.toggle_active_plot_curve())
            canvas.mpl_connect("pick_event", lambda event, src=source: self._on_plot_pick(event, src))

        self._draw_empty(self.raw_fig, self.raw_canvas, "Load spectra and click Plot RAW + SPIKE")
        self._draw_empty(self.clean_fig, self.clean_canvas, "Load spectra and click Plot RAW + SPIKE")

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def select_root_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder containing Aligned_Au_3A, Aligned_Au_8A, ...")
        if not folder:
            return
        self.root_folder = Path(folder)
        self.root_var.set(str(self.root_folder))
        self.status_var.set("Root folder selected. Click 'Scan target folders'.")

    def scan_root_folder(self) -> None:
        if self.root_folder is None:
            self.select_root_folder()
            if self.root_folder is None:
                return
        target_folders = find_target_sample_folders(self.root_folder)
        if not target_folders:
            messagebox.showwarning(
                "No target folders",
                "I did not find the exact target sample folders:\n\n"
                + "\n".join(TARGET_SAMPLE_FOLDERS)
                + "\n\nUse 'Add sample folder(s)' if the folder names are different.",
            )
            return
        self._load_sample_folders(target_folders, replace=True)

    def add_sample_folders(self) -> None:
        # Tk does not have a native multi-directory picker. Let the user repeat.
        added: List[Path] = []
        while True:
            folder = filedialog.askdirectory(title="Select one sample-family folder; Cancel when done")
            if not folder:
                break
            path = Path(folder)
            if path not in added:
                added.append(path)
            more = messagebox.askyesno("Add another?", "Add another sample-family folder?")
            if not more:
                break
        if added:
            self._load_sample_folders(added, replace=False)

    def _load_sample_folders(self, folders: Sequence[Path], replace: bool) -> None:
        if replace:
            self.records = []
            self.uid_to_record = {}
            self.last_warning_lines = []
        start_order = max([r.load_order for r in self.records], default=-1) + 1
        all_new: List[SpectrumRecord] = []
        warnings: List[str] = []
        decimal = self.decimal_var.get().strip() or "."
        for folder in folders:
            try:
                recs, warns = scan_sample_folder(folder, start_order=start_order, decimal=decimal)
                start_order += len(recs)
                all_new.extend(recs)
                warnings.extend(warns)
            except Exception as exc:
                warnings.append(f"Failed to scan {folder}: {exc}")
        # Avoid UID duplicates when adding folders twice.
        existing_uids = {r.uid for r in self.records}
        for rec in all_new:
            if rec.uid in existing_uids:
                rec.uid = rec.uid + "_dup"
            existing_uids.add(rec.uid)
            self.records.append(rec)
        self.uid_to_record = {r.uid: r for r in self.records}
        self.last_warning_lines.extend(warnings)
        self._select_default_work_family_after_load()
        self._update_spike_folder_choices(reset_to_all=True)
        self._refresh_tree()
        self._update_summary()
        n_paired = sum(1 for r in self.records if r.has_pair)
        self.status_var.set(f"Loaded {len(self.records)} row(s), {n_paired} paired RAW/spike spectra. Double-click rows to exclude bad spectra.")
        if warnings:
            self._show_warnings_short(warnings)
        if self.records:
            self.plot_both(show_errors=False)

    def _show_warnings_short(self, warnings: Sequence[str]) -> None:
        short = "\n".join(warnings[:10])
        if len(warnings) > 10:
            short += f"\n... and {len(warnings) - 10} more warning(s)."
        messagebox.showwarning("Scan warnings", short)

    def clear_records(self) -> None:
        self.records = []
        self.uid_to_record = {}
        self.last_warning_lines = []
        self.active_plot_uid = None
        self.active_plot_y_index = None
        self.work_family_var.set(WORK_ALL_FAMILIES)
        self.spike_folder_var.set(SPIKE_FOLDER_ALL)
        self._update_spike_folder_choices(reset_to_all=True)
        self._refresh_tree()
        self._update_summary()
        self._draw_empty(self.raw_fig, self.raw_canvas, "Load spectra and click Plot RAW + SPIKE")
        self._draw_empty(self.clean_fig, self.clean_canvas, "Load spectra and click Plot RAW + SPIKE")
        self.status_var.set("Cleared loaded spectra.")

    # ------------------------------------------------------------------
    # Table and filtering
    # ------------------------------------------------------------------
    def _current_work_family_key(self) -> Optional[str]:
        return work_family_key_from_label(self.work_family_var.get())

    def _available_family_keys(self) -> List[str]:
        present = {r.family_key for r in self.records}
        ordered = [key for key in FAMILY_PLOT_ORDER if key in present]
        extras = sorted(key for key in present if key not in FAMILY_PLOT_ORDER)
        return ordered + extras

    def _select_default_work_family_after_load(self) -> None:
        available = self._available_family_keys()
        current_key = self._current_work_family_key()
        if current_key in available:
            return
        if available:
            self.work_family_var.set(family_label(available[0]))
        else:
            self.work_family_var.set(WORK_ALL_FAMILIES)

    def _records_for_work_family(self, records: Sequence[SpectrumRecord]) -> List[SpectrumRecord]:
        family_key = self._current_work_family_key()
        if family_key is None:
            return list(records)
        return [r for r in records if r.family_key == family_key]

    def _available_spike_folders_for_current_family(self) -> List[str]:
        records = self._records_for_work_family(self.records)
        folders = sorted({r.clean_rel_dir for r in records if r.clean_rel_dir}, key=lambda text: normalized_name(text))
        return [SPIKE_FOLDER_ALL] + folders

    def _update_spike_folder_choices(self, reset_to_all: bool = False) -> None:
        values = self._available_spike_folders_for_current_family()
        current = self.spike_folder_var.get()
        if reset_to_all or current not in values:
            self.spike_folder_var.set(SPIKE_FOLDER_ALL)
        if hasattr(self, "spike_folder_combo"):
            self.spike_folder_combo.configure(values=values)

    def _current_spike_folder(self) -> Optional[str]:
        text = str(self.spike_folder_var.get() or "").strip()
        if not text or text == SPIKE_FOLDER_ALL:
            return None
        return text

    def _records_for_spike_folder_value(self, records: Sequence[SpectrumRecord], spike_folder: Optional[str]) -> List[SpectrumRecord]:
        if spike_folder is None:
            return list(records)
        return [r for r in records if r.clean_rel_dir == spike_folder]

    def _records_for_spike_folder(self, records: Sequence[SpectrumRecord]) -> List[SpectrumRecord]:
        return self._records_for_spike_folder_value(records, self._current_spike_folder())

    def _records_for_current_view(self, records: Sequence[SpectrumRecord]) -> List[SpectrumRecord]:
        return self._records_for_spike_folder(self._records_for_work_family(records))

    def _save_scope_family_keys(self) -> List[str]:
        family_key = self._current_work_family_key()
        if family_key is not None:
            return [family_key]
        return self._available_family_keys()

    def _save_scope_records(self) -> List[SpectrumRecord]:
        family_keys = set(self._save_scope_family_keys())
        return [r for r in self.records if r.family_key in family_keys]

    def _save_scope_label(self) -> str:
        family_key = self._current_work_family_key()
        return family_label(family_key) if family_key is not None else WORK_ALL_FAMILIES

    def _raw_group_key(self, rec: SpectrumRecord) -> str:
        if rec.raw_path is not None:
            try:
                return f"raw::{rec.raw_path.resolve()}"
            except Exception:
                return f"raw::{rec.raw_path}"
        return f"row::{rec.uid}"

    def _linked_raw_records(self, rec: SpectrumRecord, records: Optional[Sequence[SpectrumRecord]] = None) -> List[SpectrumRecord]:
        key = self._raw_group_key(rec)
        pool = self.records if records is None else records
        return [r for r in pool if self._raw_group_key(r) == key]

    def _set_y_index_for_raw_group(self, rec: SpectrumRecord, y_index: int, rejected: bool) -> None:
        for linked in self._linked_raw_records(rec):
            if rejected:
                linked.rejected_y_indices.add(y_index)
            else:
                linked.rejected_y_indices.discard(y_index)

    def _sync_rejected_columns_for_raw_groups(self, records: Sequence[SpectrumRecord]) -> None:
        by_raw: Dict[str, set[int]] = {}
        for rec in records:
            by_raw.setdefault(self._raw_group_key(rec), set()).update(rec.rejected_y_indices)
        for rec in records:
            rec.rejected_y_indices = set(by_raw.get(self._raw_group_key(rec), set()))

    def _good_spectra_records_for_scope(self, scope_records: Sequence[SpectrumRecord]) -> List[SpectrumRecord]:
        included = [r for r in scope_records if r.include]
        included_raw_keys = {self._raw_group_key(r) for r in included if r.raw_path is not None}
        out: List[SpectrumRecord] = []
        seen: set[str] = set()
        for rec in scope_records:
            should_include = rec.include or (rec.clean_path is not None and self._raw_group_key(rec) in included_raw_keys)
            if not should_include or rec.uid in seen:
                continue
            out.append(rec)
            seen.add(rec.uid)
        self._sync_rejected_columns_for_raw_groups(out)
        return out

    def _table_records(self) -> List[SpectrumRecord]:
        records = self._records_for_current_view(self.records)
        if not self.show_rejected_var.get():
            records = [r for r in records if r.include]
        return records

    def _on_work_family_changed(self) -> None:
        self.active_plot_uid = None
        self.active_plot_y_index = None
        self._update_spike_folder_choices(reset_to_all=True)
        self._refresh_tree()
        self._update_summary()
        self.plot_both(show_errors=False)

    def _on_spike_folder_changed(self) -> None:
        self.active_plot_uid = None
        self.active_plot_y_index = None
        self._refresh_tree()
        self._update_summary()
        self.plot_both(show_errors=False)

    def _on_show_rejected_changed(self) -> None:
        self._refresh_tree()
        self._update_summary()

    def _refresh_tree(self) -> None:
        self.records_tree.delete(*self.records_tree.get_children())
        self.tree_iid_to_uid.clear()
        self.tree_iid_to_y_index.clear()
        for i, rec in enumerate(self._table_records()):
            iid = f"row_{i}"
            self.tree_iid_to_uid[iid] = rec.uid
            self.tree_iid_to_y_index[iid] = None
            rejected_count = len(rec.rejected_y_indices)
            total_y = len(rec.y_column_labels)
            status = rec.status
            if total_y:
                status = f"{status}; {total_y - rejected_count}/{total_y} Y"
            self.records_tree.insert(
                "",
                tk.END,
                iid=iid,
                text=rec.raw_name or rec.clean_name or rec.temperature_label,
                open=False,
                values=(
                    "YES" if rec.include else "REJECT",
                    rec.family_label,
                    rec.temperature_label,
                    rec.raw_name,
                    rec.clean_name,
                    rec.clean_rel_dir,
                    status,
                ),
            )
            for y_index, y_label in enumerate(rec.y_column_labels):
                is_rejected = y_index in rec.rejected_y_indices
                if is_rejected and not self.show_rejected_var.get():
                    continue
                child_iid = f"{iid}_y_{y_index}"
                self.tree_iid_to_uid[child_iid] = rec.uid
                self.tree_iid_to_y_index[child_iid] = y_index
                self.records_tree.insert(
                    iid,
                    tk.END,
                    iid=child_iid,
                    text=f"Y{y_index + 1:02d}: {y_label}",
                    values=(
                        "REJECT" if is_rejected else "YES",
                        "",
                        rec.temperature_label,
                        "",
                        "",
                        "",
                        "Y column",
                    ),
                )

    def _selected_records(self) -> List[SpectrumRecord]:
        return [rec for rec, _y_index in self._selected_tree_items()]

    def _selected_tree_items(self) -> List[Tuple[SpectrumRecord, Optional[int]]]:
        selected = list(self.records_tree.selection())
        out: List[Tuple[SpectrumRecord, Optional[int]]] = []
        seen: set[Tuple[str, Optional[int]]] = set()
        for iid in selected:
            uid = self.tree_iid_to_uid.get(iid)
            rec = self.uid_to_record.get(uid or "")
            if rec is None:
                continue
            y_index = self.tree_iid_to_y_index.get(iid)
            key = (rec.uid, y_index)
            if key not in seen:
                out.append((rec, y_index))
                seen.add(key)
        return out

    def _update_status_from_selection(self) -> None:
        selected_items = self._selected_tree_items()
        selected = [rec for rec, _y_index in selected_items]
        if not selected:
            return
        if len(selected_items) == 1:
            r, y_index = selected_items[0]
            if y_index is not None:
                status = "rejected" if y_index in r.rejected_y_indices else "kept"
                y_label = r.y_column_labels[y_index] if y_index < len(r.y_column_labels) else f"Y{y_index + 1}"
                self.status_var.set(
                    f"Selected {r.family_label}, {r.temperature_label}, Y column {y_index + 1}: {y_label} ({status}). "
                    "Reject/Keep selected affects this one spectrum column in both RAW and spike-removed."
                )
                return
            value_match = ""
            if r.value_match_fraction is not None:
                value_match = f" | Y match: {100.0 * r.value_match_fraction:.1f}%"
            rejected_cols = ""
            if r.rejected_y_indices:
                rejected_cols = f" | Rejected Y columns: {', '.join(str(i + 1) for i in sorted(r.rejected_y_indices))}"
            self.status_var.set(
                f"Selected {r.family_label}, {r.temperature_label}. "
                f"RAW: {r.raw_name or 'missing'} | SPIKE: {r.clean_name or 'missing'}{value_match}{rejected_cols} | {r.note}"
            )
        else:
            self.status_var.set(f"Selected {len(selected_items)} item(s). Parent rows affect whole file pairs; child rows affect individual Y columns.")

    def _select_uid_in_tree(self, uid: str, y_index: Optional[int] = None) -> None:
        parent_iid: Optional[str] = None
        target_iid: Optional[str] = None
        for iid, row_uid in self.tree_iid_to_uid.items():
            if row_uid != uid:
                continue
            row_y_index = self.tree_iid_to_y_index.get(iid)
            if row_y_index is None:
                parent_iid = iid
            if row_y_index == y_index:
                target_iid = iid
                break
        if target_iid is None and parent_iid is not None:
            target_iid = parent_iid
        if parent_iid is not None and y_index is not None:
            self.records_tree.item(parent_iid, open=True)
        if target_iid is not None:
            self.records_tree.selection_set(target_iid)
            self.records_tree.focus(target_iid)
            self.records_tree.see(target_iid)
            return
        if y_index is not None and not self.show_rejected_var.get():
            self.show_rejected_var.set(True)
            self._refresh_tree()
            self._select_uid_in_tree(uid, y_index)
            return
        for iid, row_uid in self.tree_iid_to_uid.items():
            if row_uid == uid:
                self.records_tree.selection_set(iid)
                self.records_tree.focus(iid)
                self.records_tree.see(iid)
                break

    def toggle_selected_rows(self) -> None:
        selected_items = self._selected_tree_items()
        if not selected_items:
            return
        selected_records: List[SpectrumRecord] = []
        for rec, y_index in selected_items:
            selected_records.append(rec)
            if y_index is None:
                if rec.include:
                    rec.include = False
                elif self._can_keep_record(rec):
                    rec.include = True
            elif y_index in rec.rejected_y_indices:
                self._set_y_index_for_raw_group(rec, y_index, rejected=False)
                rec.include = True
            else:
                self._set_y_index_for_raw_group(rec, y_index, rejected=True)
        self._refresh_tree_keep_selection(selected_records)
        self._update_summary()
        self.plot_both_preserve_view(show_errors=False)

    def reject_selected_rows(self) -> None:
        selected_items = self._selected_tree_items()
        if not selected_items:
            return
        selected_records: List[SpectrumRecord] = []
        for rec, y_index in selected_items:
            selected_records.append(rec)
            if y_index is None:
                rec.include = False
            else:
                self._set_y_index_for_raw_group(rec, y_index, rejected=True)
        self._refresh_tree_keep_selection(selected_records)
        self._update_summary()
        self.plot_both_preserve_view(show_errors=False)

    def keep_selected_rows(self) -> None:
        selected_items = self._selected_tree_items()
        if not selected_items:
            return
        kept = 0
        selected_records: List[SpectrumRecord] = []
        for rec, y_index in selected_items:
            selected_records.append(rec)
            if y_index is None:
                if self._can_keep_record(rec):
                    rec.include = True
                    kept += 1
            else:
                self._set_y_index_for_raw_group(rec, y_index, rejected=False)
                rec.include = True
                kept += 1
        self._refresh_tree_keep_selection(selected_records)
        self._update_summary()
        self.plot_both_preserve_view(show_errors=False)
        if kept < len(selected_items):
            self.status_var.set(f"Kept {kept} of {len(selected_items)} selected item(s). Rows missing required files stayed rejected.")

    def keep_all_visible_rows(self) -> None:
        records = self._table_records()
        kept = 0
        for rec in records:
            if self._can_keep_record(rec):
                rec.include = True
                if self.show_rejected_var.get():
                    rec.rejected_y_indices.clear()
                kept += 1
        self._refresh_tree()
        self._update_summary()
        self.plot_both(show_errors=False)
        self.status_var.set(f"Kept {kept} visible row(s).")

    def invert_visible_rows(self) -> None:
        for rec in self._table_records():
            if self.show_rejected_var.get() and rec.y_column_labels:
                visible_indices = set(range(len(rec.y_column_labels)))
                rec.rejected_y_indices = visible_indices - rec.rejected_y_indices
                rec.include = True
            elif rec.include:
                rec.include = False
            elif self._can_keep_record(rec):
                rec.include = True
            else:
                rec.include = False
        self._refresh_tree()
        self._update_summary()
        self.plot_both(show_errors=False)

    def _can_keep_record(self, rec: SpectrumRecord) -> bool:
        if self.require_pair_var.get() and not rec.has_pair:
            return False
        if rec.status == "RAW ONLY" and not self.include_raw_only_var.get():
            return False
        return rec.raw_path is not None or rec.clean_path is not None

    def exclude_selected_rows(self) -> None:
        self.reject_selected_rows()

    def include_all_paired(self) -> None:
        self.keep_all_visible_rows()

    def invert_include(self) -> None:
        self.invert_visible_rows()

    def apply_filter_policy(self) -> None:
        for rec in self.records:
            if self.require_pair_var.get() and not rec.has_pair:
                rec.include = False
            if rec.status == "RAW ONLY" and not self.include_raw_only_var.get():
                rec.include = False
        self._refresh_tree()
        self._update_summary()
        self.plot_both(show_errors=False)

    def _refresh_tree_keep_selection(self, selected_records: Sequence[SpectrumRecord]) -> None:
        selected_uids = {r.uid for r in selected_records}
        self._refresh_tree()
        to_select = [iid for iid, uid in self.tree_iid_to_uid.items() if uid in selected_uids]
        if to_select:
            self.records_tree.selection_set(to_select)
            self.records_tree.focus(to_select[0])
            self.records_tree.see(to_select[0])

    def _update_summary(self) -> None:
        total = len(self.records)
        paired = sum(1 for r in self.records if r.has_pair)
        included = sum(1 for r in self.records if r.include)
        included_clean = sum(1 for r in self.records if r.include and r.clean_path is not None)
        work_records = self._records_for_work_family(self.records)
        view_records = self._records_for_current_view(self.records)
        view_included = sum(1 for r in view_records if r.include)
        pair_check_warn = sum(1 for r in view_records if r.status in {"PAIR CHECK WARN", "PAIR CHECK FAIL", "PAIR CHECK ERROR"})
        rejected_y_columns = sum(len(r.rejected_y_indices) for r in view_records)
        families = sorted({r.family_label for r in self.records})
        temps = [r.temperature for r in self.records if r.temperature is not None]
        if temps:
            temp_text = f"{min(temps):.6g}–{max(temps):.6g} K"
        else:
            temp_text = "no temperatures detected"
        work_label = self.work_family_var.get() or WORK_ALL_FAMILIES
        spike_label = self.spike_folder_var.get() or SPIKE_FOLDER_ALL
        self.summary_var.set(
            f"Rows: {total}  |  Paired RAW/spike: {paired}  |  Included: {included}\n"
            f"Work on: {work_label}  |  Spike folder: {spike_label}  |  Family rows: {len(work_records)}  |  Visible rows: {len(view_records)}  |  Included here: {view_included}\n"
            f"Rejected Y columns: {rejected_y_columns}  |  Pair-check warnings visible: {pair_check_warn}\n"
            f"Included with spike-removed file: {included_clean}\n"
            f"Families: {', '.join(families) if families else 'none'}\n"
            f"Temperature range: {temp_text}\n"
            "Tip: double-click a row or right-click a plotted trace to keep/reject it in both figures."
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _draw_empty(self, fig: Figure, canvas: FigureCanvasTkAgg, text: str) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        canvas.draw_idle()

    def _included_records_for_source(
        self,
        source: str,
        family_key: Optional[str] = None,
        use_current_filter: bool = True,
        spike_folder: Optional[str] = None,
    ) -> List[SpectrumRecord]:
        records = [r for r in self.records if r.include]
        if use_current_filter:
            records = self._records_for_current_view(records)
        elif family_key is not None:
            records = [r for r in records if r.family_key == family_key]
        if spike_folder is not None:
            records = self._records_for_spike_folder_value(records, spike_folder)

        if source == "clean":
            records = [r for r in records if r.clean_path is not None]
        elif source == "raw":
            records = [r for r in records if r.raw_path is not None]

        cap = self._float_or_none(self.cap_k_var.get())
        if cap is not None:
            records = [r for r in records if r.temperature is None or r.temperature <= cap]

        mode = self.sort_mode_var.get()
        if mode == "Temperature, then family":
            records.sort(key=lambda r: (math.inf if r.temperature is None else r.temperature, r.family_key, r.load_order))
        elif mode == "Family, then temperature":
            records.sort(key=lambda r: (r.family_key, math.inf if r.temperature is None else r.temperature, r.load_order))
        else:
            records.sort(key=lambda r: r.load_order)
        return records

    def plot_both(self, show_errors: bool = True) -> None:
        shared_ylim = None
        try:
            raw_records = self._included_records_for_source("raw")
            clean_records = self._included_records_for_source("clean")
            shared_ylim = self._shared_ylim_for_record_sets((("raw", raw_records), ("clean", clean_records)))
        except Exception:
            shared_ylim = None
        self.plot_source("raw", show_errors=show_errors, forced_ylim=shared_ylim)
        self.plot_source("clean", show_errors=show_errors, forced_ylim=shared_ylim)
        self._update_summary()
        self.save_settings(show_message=False)

    def _capture_plot_views(self) -> Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        views: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
        for source, fig in (("raw", self.raw_fig), ("clean", self.clean_fig)):
            axes = [ax for ax in fig.axes if not ax.get_label().startswith("<colorbar") and ax.has_data()]
            if axes:
                views[source] = [(tuple(ax.get_xlim()), tuple(ax.get_ylim())) for ax in axes]
        return views

    def _restore_plot_views(self, views: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]]) -> None:
        for source, fig, canvas in (
            ("raw", self.raw_fig, self.raw_canvas),
            ("clean", self.clean_fig, self.clean_canvas),
        ):
            saved = views.get(source)
            if not saved:
                continue
            axes = [ax for ax in fig.axes if not ax.get_label().startswith("<colorbar") and ax.has_data()]
            for ax, (xlim, ylim) in zip(axes, saved):
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            canvas.draw_idle()

    def plot_both_preserve_view(self, show_errors: bool = False) -> None:
        views = self._capture_plot_views()
        self.plot_both(show_errors=show_errors)
        self._restore_plot_views(views)

    def plot_source(self, source: str, show_errors: bool = True, forced_ylim: Optional[Tuple[float, float]] = None) -> None:
        fig = self.raw_fig if source == "raw" else self.clean_fig
        canvas = self.raw_canvas if source == "raw" else self.clean_canvas
        title = self.raw_title_var.get().strip() if source == "raw" else self.clean_title_var.get().strip()
        try:
            records = self._included_records_for_source(source)
            if not records:
                self._draw_empty(fig, canvas, f"No included {'RAW' if source == 'raw' else 'spike-removed'} spectra to plot")
                return
            curve_count = self._draw_cascade(fig=fig, canvas=canvas, records=records, source=source, title=title, forced_ylim=forced_ylim)
            self.status_var.set(f"Plotted {curve_count} {'RAW' if source == 'raw' else 'spike-removed'} curve(s) from {len(records)} kept row(s).")
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Plot failed", f"{exc}\n\n{traceback.format_exc(limit=2)}")
            self.status_var.set(f"Plot failed: {exc}")

    def _current_x_segments(self) -> List[Tuple[float, float]]:
        x_start = float(self.x_start_var.get())
        x_end = float(self.x_end_var.get())
        breaks = parse_breaks(self.breaks_var.get()) if self.use_x_breaks_var.get() else []
        return make_segments(x_start, x_end, breaks)

    def _curve_display_ylim(self, curves: Sequence[PlotCurve], segments: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not curves:
            return None
        offset_step = self._offset_step(curves)
        y_min = math.inf
        y_max = -math.inf
        plotted_any = False
        for idx, curve in enumerate(curves):
            y_offset = curve.y + idx * offset_step
            for lo, hi in segments:
                mask = (curve.x >= lo) & (curve.x <= hi) & np.isfinite(curve.x) & np.isfinite(y_offset)
                if not np.any(mask):
                    continue
                yb = y_offset[mask]
                plotted_any = True
                y_min = min(y_min, float(np.nanmin(yb)))
                y_max = max(y_max, float(np.nanmax(yb)))
        if not plotted_any:
            return None
        y_range = max(1e-12, y_max - y_min)
        return (y_min - 0.03 * y_range, y_max + 0.16 * y_range)

    def _shared_ylim_for_record_sets(self, record_sets: Sequence[Tuple[str, Sequence[SpectrumRecord]]]) -> Optional[Tuple[float, float]]:
        segments = self._current_x_segments()
        y_min = math.inf
        y_max = -math.inf
        found = False
        for source, records in record_sets:
            curves, _read_warnings = self._load_plot_curves(records=records, source=source)
            ylim = self._curve_display_ylim(curves, segments)
            if ylim is None:
                continue
            found = True
            y_min = min(y_min, ylim[0])
            y_max = max(y_max, ylim[1])
        if not found:
            return None
        return (y_min, y_max)

    def _draw_cascade(
        self,
        fig: Figure,
        canvas: Optional[FigureCanvasTkAgg],
        records: Sequence[SpectrumRecord],
        source: str,
        title: str,
        forced_ylim: Optional[Tuple[float, float]] = None,
    ) -> int:
        fig.clear()
        setup_paper_fonts()

        segments = self._current_x_segments()
        widths = [max(1.0, hi - lo) for lo, hi in segments]

        axes = fig.subplots(
            1,
            len(segments),
            sharey=True,
            gridspec_kw={"width_ratios": widths, "wspace": 0.01},
        )
        if len(segments) == 1:
            axes = [axes]
        else:
            axes = list(np.ravel(axes))

        curves, read_warnings = self._load_plot_curves(records=records, source=source)
        if not curves:
            raise RuntimeError("No spectra could be read for plotting. Check delimiter/decimal settings.")

        offset_step = self._offset_step(curves)
        for idx, curve in enumerate(curves):
            curve.offset_index = idx
            curve.y_offset = curve.y + idx * offset_step

        finite_temps = [c.record.temperature for c in curves if c.record.temperature is not None and np.isfinite(c.record.temperature)]
        cmap = get_temperature_cmap(self.cmap_var.get().strip() or "turbo")
        if finite_temps:
            vmin = float(min(finite_temps))
            vmax = float(max(finite_temps))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None

        line_width = self._float_or_default(self.line_width_var.get(), 1.15)
        marker_size = self._float_or_default(self.marker_size_var.get(), 120.0)
        scatter_step = max(1, int(self._float_or_default(self.scatter_step_var.get(), 1)))

        y_min = math.inf
        y_max = -math.inf
        plotted_any = False
        self.last_plotted_uids[source] = set()

        for ax_index, (ax, (lo, hi)) in enumerate(zip(axes, segments)):
            for curve in curves:
                x = curve.x
                y = curve.y_offset
                mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
                if not np.any(mask):
                    continue
                xb = x[mask]
                yb = y[mask]
                plotted_any = True
                y_min = min(y_min, float(np.nanmin(yb)))
                y_max = max(y_max, float(np.nanmax(yb)))

                rec = curve.record
                if rec.temperature is not None and norm is not None:
                    line_color = cmap(norm(float(rec.temperature)))
                else:
                    line_color = "#707070"
                marker_style = sample_marker_style(rec.family_key)
                is_active_pick = (
                    canvas is not None
                    and rec.uid == self.active_plot_uid
                    and curve.y_index == self.active_plot_y_index
                    and curve.y_index >= 0
                )

                (line,) = ax.plot(
                    xb,
                    yb,
                    color=line_color,
                    lw=line_width * (1.85 if is_active_pick else 1.0),
                    alpha=1.0,
                    zorder=4 if is_active_pick else 2,
                    picker=6,
                )
                setattr(line, "_record_uid", rec.uid)
                setattr(line, "_y_index", curve.y_index)
                setattr(line, "_y_col", curve.y_col)
                setattr(line, "_plot_source", source)

                scatter_x = xb[::scatter_step]
                scatter_y = yb[::scatter_step]
                sc = ax.scatter(
                    scatter_x,
                    scatter_y,
                    s=marker_size * (1.25 if is_active_pick else 1.0),
                    marker=marker_style["marker"],
                    facecolors=marker_style["facecolors"],
                    edgecolors=marker_style["edgecolors"],
                    linewidths=marker_style["linewidths"],
                    alpha=0.95,
                    zorder=5 if marker_style.get("hollow") else 3,
                    picker=True,
                )
                setattr(sc, "_record_uid", rec.uid)
                setattr(sc, "_y_index", curve.y_index)
                setattr(sc, "_y_col", curve.y_col)
                setattr(sc, "_plot_source", source)
                self.last_plotted_uids[source].add(rec.uid)

            ax.set_xlim(lo, hi)
            ax.set_xlabel(r"Raman Shift (cm$^{-1}$)")
            ax.grid(False)
            format_axis_ticks(ax, is_first_panel=(ax_index == 0), x_fmt="{x:.0f}", y_fmt="{x:.2f}")

            # Broken-axis diagonal marks.
            d = 0.015
            if ax_index < len(axes) - 1:
                kw = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.9)
                ax.plot((1 - d, 1 + d), (-d, +d), **kw)
                ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
            if ax_index > 0:
                kw = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.9)
                ax.plot((-d, +d), (-d, +d), **kw)
                ax.plot((-d, +d), (1 - d, 1 + d), **kw)

        if not plotted_any:
            raise RuntimeError("No plotted points were inside the selected X range/segments.")

        if forced_ylim is not None:
            for ax in axes:
                ax.set_ylim(*forced_ylim)
        else:
            y_range = max(1e-12, y_max - y_min)
            for ax in axes:
                ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.16 * y_range)
        axes[0].set_ylabel("Intensity (a.u.)")
        if title:
            axes[0].set_title(title, loc="left", pad=8)

        fig.subplots_adjust(wspace=0.01, right=0.90, left=0.08, bottom=0.13, top=0.90)

        if norm is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, location="right", pad=0.012, fraction=0.032, shrink=1.0, aspect=30)
            cbar.set_label("Temperature (K)", size=18)
            cbar.ax.tick_params(labelsize=14)
            ticks = np.asarray(cbar.get_ticks(), dtype=float)
            if ticks.size:
                vmin = float(norm.vmin)
                vmax = float(norm.vmax)
                if not np.any(np.isclose(ticks, vmin)):
                    ticks = np.r_[vmin, ticks]
                if not np.any(np.isclose(ticks, vmax)):
                    ticks = np.r_[ticks, vmax]
                ticks = ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]
                cbar.set_ticks(np.unique(np.round(ticks, 10)))

        if self.legend_var.get():
            used_families = []
            for curve in curves:
                if curve.record.family_key not in used_families:
                    used_families.append(curve.record.family_key)
            handles = []
            for key in used_families:
                marker_style = sample_marker_style(key)
                handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="None",
                        marker=marker_style["marker"],
                        markersize=8.5,
                        markerfacecolor=marker_style["markerfacecolor"],
                        markeredgecolor=marker_style["markeredgecolor"],
                        markeredgewidth=marker_style["markeredgewidth"],
                        label="",
                    )
                )
            if handles:
                axes[0].legend(
                    handles=handles,
                    labels=[""] * len(handles),
                    loc="upper left",
                    bbox_to_anchor=(0.012, 0.988),
                    bbox_transform=axes[0].transAxes,
                    frameon=False,
                    handlelength=1.0,
                    handletextpad=0.0,
                    borderpad=0.1,
                    labelspacing=0.25,
                )

        if canvas is not None:
            canvas.draw_idle()

        if read_warnings:
            self.status_var.set(f"Plotted with {len(read_warnings)} file-read warning(s). First: {read_warnings[0]}")
        return len(curves)

    def _load_plot_curves(self, records: Sequence[SpectrumRecord], source: str) -> Tuple[List[PlotCurve], List[str]]:
        curves: List[PlotCurve] = []
        warnings: List[str] = []
        decimal = self.decimal_var.get().strip() or "."
        norm_mode = self.normalization_var.get()
        for rec in records:
            path = rec.raw_path if source == "raw" else rec.clean_path
            if path is None:
                continue
            try:
                spec = read_spectrum_file(path, decimal=decimal)
                kept_pairs = [(idx, curve) for idx, curve in enumerate(spec.curves) if idx not in rec.rejected_y_indices]
                if not kept_pairs:
                    continue
                if self.average_y_columns_var.get():
                    spectrum_pairs = [(-1, average_spectrum_curve_list([curve for _idx, curve in kept_pairs]))]
                else:
                    spectrum_pairs = kept_pairs
                for y_index, spectrum_curve in spectrum_pairs:
                    y = apply_normalization(spectrum_curve.y, norm_mode)
                    curves.append(
                        PlotCurve(
                            record=rec,
                            x=spectrum_curve.x,
                            y=y,
                            y_offset=y.copy(),
                            offset_index=0,
                            y_col=spectrum_curve.y_col,
                            y_index=y_index,
                        )
                    )
            except Exception as exc:
                warnings.append(f"{path.name}: {exc}")
        return curves, warnings

    def _offset_step(self, curves: Sequence[PlotCurve]) -> float:
        if self.plot_mode_var.get() != "Stacked":
            return 0.0
        if not self.auto_offset_var.get():
            manual_offset = self._float_or_default(self.offset_var.get(), 0.05)
            return manual_offset if abs(manual_offset) > 1e-12 else 0.05
        values = []
        for c in curves:
            finite = c.y[np.isfinite(c.y)]
            if finite.size:
                values.append(finite)
        if not values:
            return 0.05
        joined = np.concatenate(values)
        p5, p95 = np.nanpercentile(joined, [5, 95])
        robust_range = float(p95 - p5)
        if not np.isfinite(robust_range) or robust_range <= 0:
            robust_range = float(np.nanmax(joined) - np.nanmin(joined))
        if not np.isfinite(robust_range) or robust_range <= 0:
            return 0.05
        return 0.08 * robust_range

    def _active_plot_record(self) -> Optional[SpectrumRecord]:
        if self.active_plot_uid is None:
            return None
        return self.uid_to_record.get(self.active_plot_uid)

    def reject_active_plot_curve(self) -> None:
        rec = self._active_plot_record()
        y_index = self.active_plot_y_index
        if rec is None or y_index is None or y_index < 0:
            self.reject_selected_rows()
            return
        self._set_y_index_for_raw_group(rec, y_index, rejected=True)
        self._refresh_tree()
        self._select_uid_in_tree(rec.uid, y_index)
        self._update_summary()
        self.plot_both_preserve_view(show_errors=False)
        self.status_var.set(f"Rejected Y column {y_index + 1} for {rec.family_label} {rec.temperature_label} in both RAW and spike-removed plots.")

    def keep_active_plot_curve(self) -> None:
        rec = self._active_plot_record()
        y_index = self.active_plot_y_index
        if rec is None or y_index is None or y_index < 0:
            self.keep_selected_rows()
            return
        self._set_y_index_for_raw_group(rec, y_index, rejected=False)
        rec.include = True
        self._refresh_tree()
        self._select_uid_in_tree(rec.uid, y_index)
        self._update_summary()
        self.plot_both_preserve_view(show_errors=False)
        self.status_var.set(f"Kept Y column {y_index + 1} for {rec.family_label} {rec.temperature_label}.")

    def toggle_active_plot_curve(self) -> None:
        rec = self._active_plot_record()
        y_index = self.active_plot_y_index
        if rec is None or y_index is None or y_index < 0:
            self.toggle_selected_rows()
            return
        if y_index in rec.rejected_y_indices:
            self.keep_active_plot_curve()
        else:
            self.reject_active_plot_curve()

    def _on_plot_pick(self, event, source: str) -> None:
        artist = getattr(event, "artist", None)
        uid = getattr(artist, "_record_uid", None)
        y_index = getattr(artist, "_y_index", None)
        y_col = getattr(artist, "_y_col", "")
        if not uid:
            return
        rec = self.uid_to_record.get(uid)
        if rec is None:
            return
        mouseevent = getattr(event, "mouseevent", None)
        button = getattr(mouseevent, "button", None)
        try:
            mouseevent.canvas.get_tk_widget().focus_set()
        except Exception:
            pass
        self.active_plot_uid = uid
        self.active_plot_y_index = int(y_index) if y_index is not None else None
        self._select_uid_in_tree(uid, self.active_plot_y_index if self.active_plot_y_index is not None and self.active_plot_y_index >= 0 else None)
        if self.active_plot_y_index is not None and self.active_plot_y_index < 0:
            self.plot_both_preserve_view(show_errors=False)
            self.status_var.set(
                f"Selected averaged curve for {rec.family_label} {rec.temperature_label}. "
                "Turn off 'Average Y columns per file' to reject individual Y spectra."
            )
            return
        if button == 3:
            self.toggle_active_plot_curve()
        else:
            self.plot_both_preserve_view(show_errors=False)
            y_text = f"Y column {self.active_plot_y_index + 1}" if self.active_plot_y_index is not None else "Y column"
            if y_col:
                y_text += f" ({y_col})"
            self.status_var.set(
                f"Selected {y_text} for {rec.family_label} {rec.temperature_label}. "
                "Delete rejects this Y column; Enter/Space toggles it; right-click toggles immediately."
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_figures(self) -> None:
        scope_label = self._save_scope_label()
        scope_records = self._save_scope_records()
        spike_filter = self._current_spike_folder()
        figure_scope_label = scope_label if spike_filter is None else f"{scope_label} / {spike_filter}"
        figure_scope_records = self._records_for_spike_folder_value(scope_records, spike_filter)
        included = [r for r in figure_scope_records if r.include]
        if not included:
            messagebox.showinfo("No included spectra", f"Keep at least one spectrum in {figure_scope_label} before saving figures.")
            return
        folder = filedialog.askdirectory(title="Choose output folder for paper figures")
        if not folder:
            return
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_dir = out_dir

        saved_files = 0
        skipped: List[str] = []
        try:
            for family_key in self._save_scope_family_keys():
                file_stem = FAMILY_FILE_STEMS.get(family_key, safe_filename_part(family_label(family_key)))
                family_raw_records = self._included_records_for_source(
                    "raw",
                    family_key=family_key,
                    use_current_filter=False,
                    spike_folder=spike_filter,
                )
                family_clean_records = self._included_records_for_source(
                    "clean",
                    family_key=family_key,
                    use_current_filter=False,
                    spike_folder=spike_filter,
                )
                shared_ylim = self._shared_ylim_for_record_sets((("raw", family_raw_records), ("clean", family_clean_records)))
                for source, suffix, title in (
                    ("raw", "RAW", self.raw_title_var.get().strip()),
                    ("clean", "Spikes_Removed", self.clean_title_var.get().strip()),
                ):
                    records = family_raw_records if source == "raw" else family_clean_records
                    if not records:
                        skipped.append(f"{file_stem} {suffix}: no included {'RAW' if source == 'raw' else 'spike-removed'} spectra")
                        continue
                    fig = Figure(figsize=(12.5, 6), dpi=100, facecolor="white")
                    try:
                        self._draw_cascade(fig=fig, canvas=None, records=records, source=source, title=title, forced_ylim=shared_ylim)
                    except Exception as exc:
                        skipped.append(f"{file_stem} {suffix}: {exc}")
                        plt.close(fig)
                        continue
                    base = out_dir / f"{file_stem}_{suffix}"
                    for ext in SAVE_FORMATS:
                        fig.savefig(base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
                        saved_files += 1
                    plt.close(fig)

            self._write_manifest(out_dir / "plot_manifest_all_rows.csv", figure_scope_records)
            self._write_manifest(out_dir / "plot_manifest_included_rows.csv", included)
            self.save_settings(show_message=False)
            if saved_files:
                detail = ""
                if skipped:
                    detail = "\n\nSkipped:\n" + "\n".join(skipped[:12])
                    if len(skipped) > 12:
                        detail += f"\n... and {len(skipped) - 12} more skip(s)."
                messagebox.showinfo(
                    "Saved figures",
                    f"Saved {saved_files} figure file(s) for {figure_scope_label} plus manifests into:\n{out_dir}{detail}",
                )
                self.status_var.set(f"Saved {saved_files} figure file(s) for {figure_scope_label} into {out_dir}")
            else:
                messagebox.showwarning("No figures saved", f"No included spectra in {figure_scope_label} were available for figure export.")
                self.status_var.set(f"No figures saved: no included spectra were available for {figure_scope_label}.")
        except Exception as exc:
            messagebox.showerror("Save figures failed", str(exc))
        finally:
            self.plot_both(show_errors=False)

    def save_good_spectra(self) -> None:
        scope_label = self._save_scope_label()
        scope_records = self._save_scope_records()
        included_rows = [r for r in scope_records if r.include]
        if not included_rows:
            messagebox.showinfo("No included spectra", f"Keep at least one spectrum in {scope_label} before saving GOOD spectra.")
            return
        selected = self._good_spectra_records_for_scope(scope_records)
        folder = filedialog.askdirectory(title="Choose parent folder for GOOD_SPECTRA")
        if not folder:
            return
        parent_dir = Path(folder)
        base_dir = parent_dir.parent if parent_dir.name == "GOOD_SPECTRA" else parent_dir
        out_dir = unique_output_dir(base_dir / "GOOD_SPECTRA")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_dir = out_dir
        copied = []
        copied_pairs: set[Tuple[str, str]] = set()
        skipped: List[str] = []
        decimal = self.decimal_var.get().strip() or "."
        try:
            for rec in selected:
                family_dir = out_dir / safe_filename_part(rec.sample_folder.name)
                if rec.raw_path is not None:
                    raw_dst = family_dir / "RAW" / rec.raw_path.name
                    key = (str(rec.raw_path.resolve()), str(raw_dst.resolve()), ",".join(str(i) for i in sorted(rec.rejected_y_indices)))
                    if key not in copied_pairs:
                        try:
                            copied_path, kept_cols, total_cols = write_filtered_spectrum_file(
                                rec.raw_path,
                                raw_dst,
                                rec.rejected_y_indices,
                                decimal=decimal,
                            )
                            copied.append(copied_path)
                            copied_pairs.add(key)
                            if kept_cols < total_cols:
                                skipped.append(f"{rec.raw_path.name}: saved {kept_cols}/{total_cols} Y columns")
                        except Exception as exc:
                            skipped.append(f"{rec.raw_path.name}: {exc}")
                if rec.clean_path is not None:
                    clean_rel = Path(rec.clean_rel_dir) if rec.clean_rel_dir else Path("SPIKE_REMOVED")
                    clean_dst = family_dir / clean_rel / rec.clean_path.name
                    key = (str(rec.clean_path.resolve()), str(clean_dst.resolve()), ",".join(str(i) for i in sorted(rec.rejected_y_indices)))
                    if key not in copied_pairs:
                        try:
                            copied_path, kept_cols, total_cols = write_filtered_spectrum_file(
                                rec.clean_path,
                                clean_dst,
                                rec.rejected_y_indices,
                                decimal=decimal,
                            )
                            copied.append(copied_path)
                            copied_pairs.add(key)
                            if kept_cols < total_cols:
                                skipped.append(f"{rec.clean_path.name}: saved {kept_cols}/{total_cols} Y columns")
                        except Exception as exc:
                            skipped.append(f"{rec.clean_path.name}: {exc}")

            self._write_manifest(out_dir / "manifest_all_rows.csv", scope_records)
            self._write_manifest(out_dir / "manifest_included_rows.csv", selected)
            self._write_wide_csv(out_dir / "combined_RAW_selected.csv", selected, source="raw")
            self._write_wide_csv(out_dir / "combined_SPIKE_REMOVED_selected.csv", selected, source="clean")
            self._try_write_wide_xlsx(out_dir / "combined_SELECTED_spectra.xlsx", selected)
            self.save_settings(show_message=False)

            messagebox.showinfo(
                "Saved GOOD spectra",
                f"Saved {len(copied)} filtered file(s) for {scope_label} and wrote manifests/combined tables into:\n{out_dir}\n\n"
                f"Kept RAW rows: {len(included_rows)}. Spike-folder records carried forward: {len(selected)}.\n"
                "RAW files are under each family/RAW folder. Spike-removed files keep their original spike-folder relative path.\n"
                "Each saved file keeps X plus only the included individual Y columns."
                + (("\n\nColumn notes:\n" + "\n".join(skipped[:12])) if skipped else "")
            )
            self.status_var.set(f"Saved GOOD spectra for {scope_label} into {out_dir}")
        except Exception as exc:
            messagebox.showerror("Save GOOD spectra failed", f"{exc}\n\n{traceback.format_exc(limit=2)}")

    def save_filtered_data(self) -> None:
        self.save_good_spectra()

    def _write_manifest(self, path: Path, records: Sequence[SpectrumRecord]) -> None:
        rows = [r.to_manifest_row() for r in records]
        pd.DataFrame(rows).to_csv(path, index=False)

    def _wide_dataframe(self, records: Sequence[SpectrumRecord], source: str) -> pd.DataFrame:
        decimal = self.decimal_var.get().strip() or "."
        columns: Dict[str, pd.Series] = {}
        seen_inputs: set[Tuple[str, str]] = set()
        for i, rec in enumerate(records, start=1):
            path = rec.raw_path if source == "raw" else rec.clean_path
            if path is None:
                continue
            try:
                path_key = str(path.resolve())
            except Exception:
                path_key = str(path)
            rejected_key = ",".join(str(idx) for idx in sorted(rec.rejected_y_indices))
            input_key = (path_key, rejected_key)
            if input_key in seen_inputs:
                continue
            seen_inputs.add(input_key)
            try:
                spec = read_spectrum_file(path, decimal=decimal)
            except Exception:
                continue
            kept_curves = [(idx, curve) for idx, curve in enumerate(spec.curves) if idx not in rec.rejected_y_indices]
            for idx, spectrum_curve in kept_curves:
                tag = safe_filename_part(f"{i:03d}_{idx + 1:02d}_{rec.family_key}_{rec.temperature_label}_{Path(path).stem}_{spectrum_curve.y_col}")[:70]
                columns[f"X_{tag}"] = pd.Series(spectrum_curve.x)
                columns[f"Y_{tag}"] = pd.Series(spectrum_curve.y)
        return pd.DataFrame(columns)

    def _write_wide_csv(self, path: Path, records: Sequence[SpectrumRecord], source: str) -> None:
        df = self._wide_dataframe(records, source=source)
        if df.empty:
            return
        df.to_csv(path, index=False)

    def _try_write_wide_xlsx(self, path: Path, records: Sequence[SpectrumRecord]) -> None:
        if importlib.util.find_spec("openpyxl") is None:
            return
        raw_df = self._wide_dataframe(records, source="raw")
        clean_df = self._wide_dataframe(records, source="clean")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame([r.to_manifest_row() for r in records]).to_excel(writer, index=False, sheet_name="manifest_included")
            if not raw_df.empty:
                raw_df.to_excel(writer, index=False, sheet_name="RAW_selected")
            if not clean_df.empty:
                clean_df.to_excel(writer, index=False, sheet_name="SPIKE_REMOVED_selected")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _load_settings_file(self) -> Dict[str, object]:
        try:
            if self.settings_path.exists():
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _settings_text(self, key: str, default: str) -> str:
        value = self.settings.get(key, default)
        if value is None:
            return default
        return str(value)

    def _settings_bool(self, key: str, default: bool) -> bool:
        value = self.settings.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def save_settings(self, show_message: bool = False) -> None:
        data: Dict[str, object] = {
            "last_root_folder": str(self.root_folder) if self.root_folder else "",
            "selected_work_family": self._current_work_family_key() or WORK_ALL_FAMILIES,
            "normalization": self.normalization_var.get(),
            "average_y_columns": bool(self.average_y_columns_var.get()),
            "plot_mode": self.plot_mode_var.get(),
            "cap_k": self.cap_k_var.get(),
            "x_start": self.x_start_var.get(),
            "x_end": self.x_end_var.get(),
            "breaks": self.breaks_var.get(),
            "use_x_breaks": bool(self.use_x_breaks_var.get()),
            "offset": self.offset_var.get(),
            "auto_offset": bool(self.auto_offset_var.get()),
            "line_width": self.line_width_var.get(),
            "scatter_size": self.marker_size_var.get(),
            "point_step": self.scatter_step_var.get(),
            "cmap": self.cmap_var.get(),
            "decimal": self.decimal_var.get(),
            "family_legend": bool(self.legend_var.get()),
            "show_rejected_rows": bool(self.show_rejected_var.get()),
            "raw_title": self.raw_title_var.get(),
            "spike_title": self.clean_title_var.get(),
            "sort_mode": self.sort_mode_var.get(),
        }
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
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
    @staticmethod
    def _float_or_default(text: str, default: float) -> float:
        try:
            return float(str(text).strip())
        except Exception:
            return default

    @staticmethod
    def _float_or_none(text: str) -> Optional[float]:
        s = str(text).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _try_zoom(self) -> None:
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = PaperSpikePlotterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
