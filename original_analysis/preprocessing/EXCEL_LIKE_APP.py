"""
EXCEL_LIKE_APP_GITHUB_READY.py

A polished Excel-like multi-TXT/CSV/XLSX column plotter with robust Windows multi-select file loading.

Main features
-------------
- Open many TXT/CSV/TSV/DAT/ASC/XY/XLSX files at once using askopenfilenames
- Auto-detect delimiter and header for every file
- Preview the active file in an Excel-like grid
- Choose one X column and many Y columns per file
- Plot all selected traces from all files together
- Plot only the averaged curve of each TXT file
- Plot a grand average across TXT files using interpolation to a common X grid
- Save plots and export selected/averaged data

Install:
    pip install ttkbootstrap pandas matplotlib openpyxl numpy

Run:
    python EXCEL_LIKE_APP.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _dependency_help_message(missing: Sequence[str]) -> str:
    missing_text = ", ".join(missing)
    return (
        "Missing required Python package(s): "
        f"{missing_text}\n\n"
        "Install the full app environment with:\n"
        "    python -m pip install -r requirements.txt\n\n"
        "or directly with:\n"
        "    python -m pip install ttkbootstrap pandas matplotlib openpyxl numpy\n\n"
        "If you are using Anaconda, first activate the environment you want, then run the install command there."
    )


def _check_required_dependencies() -> None:
    """Fail early with a useful message instead of a long traceback.

    This keeps the script friendlier for new PCs and makes the project more
    suitable for GitHub/paper supplementary-material release.
    """
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("ttkbootstrap", "ttkbootstrap"),
        ("openpyxl", "openpyxl"),
    ]

    missing = [package for package, module in required if importlib.util.find_spec(module) is None]

    # tkinter is part of the Python installation, not pip, but missing Tk support
    # is common on some Linux/Python builds. Check it separately.
    if importlib.util.find_spec("tkinter") is None:
        missing.append("tkinter / Python Tk support")

    if missing:
        message = _dependency_help_message(missing)
        print("\n" + "=" * 78)
        print(message)
        print("=" * 78 + "\n")
        try:
            import tkinter as _tk
            from tkinter import messagebox as _messagebox

            root = _tk.Tk()
            root.withdraw()
            _messagebox.showerror("Missing dependencies", message)
            root.destroy()
        except Exception:
            pass
        raise SystemExit(1)


_check_required_dependencies()

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

try:
    from matplotlib import colormaps as mpl_colormaps
except Exception:  # Older Matplotlib fallback; avoid deprecated colormap APIs.
    mpl_colormaps = None


APP_TITLE = "Excel-like Raman Multi-TXT Plotter — GitHub-ready toolbar-fixed spike editor"
SUPPORTED_EXTENSIONS = (".txt", ".csv", ".tsv", ".dat", ".asc", ".xy", ".xlsx", ".xls")

# Simple Tk-compatible file type patterns. The important part for multi-select
# is using filedialog.askopenfilenames, not askopenfilename.
FILE_DIALOG_TYPES = [
    ("TXT files", "*.txt"),
    ("Text/data files", "*.txt *.csv *.tsv *.dat *.asc *.xy"),
    ("Excel files", "*.xlsx *.xls"),
    ("All supported files", "*.txt *.csv *.tsv *.dat *.asc *.xy *.xlsx *.xls"),
    ("All files", "*.*"),
]

DELIMITER_OPTIONS = ["Auto", "Whitespace", "Tab", "Comma", "Semicolon", "Pipe"]
HEADER_OPTIONS = ["Auto", "No header", "First row is header"]
DECIMAL_OPTIONS = [".", ","]
NORMALIZE_OPTIONS = ["None", "Max = 1", "Min-max", "Z-score"]
PLOT_MODES = [
    "All traces from all files",
    "Average Y columns in each file",
    "Grand average of file averages",
    "Temperature waterfall average per file",
]

TEMPERATURE_RE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*K")


def extract_temperature_from_filename(path_or_name: object) -> Tuple[Optional[float], Optional[str]]:
    """Return Raman temperature metadata from the number immediately before capital K."""
    name = Path(str(path_or_name)).name
    matches = list(TEMPERATURE_RE.finditer(name))
    if not matches:
        return None, None
    raw = matches[-1].group(1).replace(",", ".")
    try:
        temperature = float(raw)
    except ValueError:
        return None, None
    label = f"{temperature:.12g}K"
    return temperature, label


@dataclass
class LoadedDataFile:
    path: Path
    df: pd.DataFrame
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    numeric_cols: List[str] = field(default_factory=list)
    x_col: Optional[str] = None
    y_cols: List[str] = field(default_factory=list)
    temperature: Optional[float] = None
    temperature_label: Optional[str] = None
    load_order: int = 0
    spike_corrections_applied: int = 0
    last_corrected_save_path: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem


class MultiTxtPlotter(tb.Window):
    def __init__(self) -> None:
        super().__init__(themename="flatly")

        self.title(APP_TITLE)
        self.geometry("1760x1020")
        self.minsize(1320, 780)

        self.files: List[LoadedDataFile] = []
        self.active_index: Optional[int] = None
        self._next_load_order = 0

        self.file_iid_to_index: Dict[str, int] = {}
        self.preview_id_to_col: Dict[str, str] = {}
        self.role_id_to_col: Dict[str, str] = {}
        self.col_to_role_id: Dict[str, str] = {}

        self.status_var = tk.StringVar(
            value="Open one or more TXT files. In the file dialog, use Ctrl-click or Shift-click to select many files; first numeric column = X, remaining numeric columns = Y."
        )
        self.file_label_var = tk.StringVar(value="No files loaded")
        self.summary_var = tk.StringVar(value="No data loaded yet.")
        self.active_file_var = tk.StringVar(value="No active file")

        # Parsing controls
        self.delimiter_var = tk.StringVar(value="Auto")
        self.header_var = tk.StringVar(value="Auto")
        self.decimal_var = tk.StringVar(value=".")
        self.skiprows_var = tk.StringVar(value="0")
        self.preview_rows_var = tk.StringVar(value="500")
        self.preview_cols_var = tk.StringVar(value="250")

        # Column role controls
        self.x_combo_var = tk.StringVar(value="")

        # Plot controls
        self.plot_mode_var = tk.StringVar(value=PLOT_MODES[0])
        self.normalize_var = tk.StringVar(value="None")
        self.theme_var = tk.StringVar(value="flatly")
        self.sort_x_var = tk.BooleanVar(value=True)
        self.sort_temp_var = tk.BooleanVar(value=True)
        self.color_by_temp_var = tk.BooleanVar(value=True)
        self.fast_display_var = tk.BooleanVar(value=True)
        self.grid_var = tk.BooleanVar(value=True)
        self.markers_var = tk.BooleanVar(value=False)
        self.legend_var = tk.BooleanVar(value=True)
        self.logy_var = tk.BooleanVar(value=False)
        self.offset_var = tk.BooleanVar(value=False)
        self.auto_waterfall_offset_var = tk.BooleanVar(value=True)
        self.std_band_var = tk.BooleanVar(value=False)
        self.offset_value_var = tk.StringVar(value="0")
        self.max_plot_points_var = tk.StringVar(value="2500")
        self.linewidth_var = tk.StringVar(value="1.3")
        self.title_var = tk.StringVar(value="")
        self.xlabel_var = tk.StringVar(value="")
        self.ylabel_var = tk.StringVar(value="Intensity")

        # Used by export
        self.last_plot_payload: Dict[str, object] = {}
        self._current_colorbar = None

        # Spike editor state
        self.spike_window: Optional[tk.Toplevel] = None
        self.spike_file_var = tk.StringVar(value="")
        self.spike_y_var = tk.StringVar(value="")
        self.spike_count_var = tk.StringVar(value="Selected spikes: 0")
        self.spike_visible_count_var = tk.StringVar(value="Visible spectra: 0")
        self.spike_status_var = tk.StringVar(value="Open files, choose a spectrum, then click or drag-box spike points.")
        self.spike_view_mode_var = tk.StringVar(value="Overlay")
        self.spike_show_all_var = tk.BooleanVar(value=True)
        self.spike_same_y_only_var = tk.BooleanVar(value=False)
        self.max_spike_display_points_var = tk.StringVar(value="3500")
        self.max_spike_context_var = tk.StringVar(value="1200")
        self.spike_stack_visible_var = tk.StringVar(value="6")
        self.spike_stack_offset_var = tk.StringVar(value="Auto")
        self.spike_stack_position_var = tk.DoubleVar(value=0.0)
        self.spike_file_label_to_index: Dict[str, int] = {}
        self.spike_selected_positions: set[Tuple[int, str, int]] = set()
        self.spike_selection_order: List[Tuple[int, str, int]] = []
        self.spike_apply_history: List[Dict[str, object]] = []
        self.spike_curve_positions = np.array([], dtype=int)
        self.spike_curve_x = np.array([], dtype=float)
        self.spike_curve_y = np.array([], dtype=float)
        self.spike_visible_curves: List[Dict[str, object]] = []
        self.spike_context_hidden_count = 0
        self.spike_canvas: Optional[FigureCanvasTkAgg] = None
        self.spike_toolbar: Optional[NavigationToolbar2Tk] = None
        self.spike_toolbar_host = None
        self.spike_file_combo = None
        self.spike_y_combo = None
        self.spike_figure: Optional[Figure] = None
        self.spike_ax = None
        self.spike_listbox: Optional[tk.Listbox] = None
        self.spike_rect_selector = None
        self.spike_stack_scale = None
        self.spike_stack_vscale = None

        self._build_ui()
        self.sort_temp_var.trace_add("write", lambda *_args: self._on_sort_temperature_changed())
        self.after(100, self._maximize_main_window)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._build_top_toolbar()

        main = tb.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 6))

        left = tb.Frame(main)
        center = tb.Frame(main)
        right = tb.Frame(main)
        left.configure(width=360)
        right.configure(width=405)
        left.pack_propagate(False)
        right.pack_propagate(False)

        main.add(left)
        main.add(center)
        main.add(right)

        self._build_left_panel(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

        status = tb.Label(self, textvariable=self.status_var, anchor=tk.W)
        status.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.after(250, lambda: self._safe_set_sashes(main))

    def _safe_set_sashes(self, pane: tb.Panedwindow) -> None:
        try:
            pane.sashpos(0, 365)
            pane.sashpos(1, max(930, self.winfo_width() - 430))
        except Exception:
            pass

    def _build_top_toolbar(self) -> None:
        toolbar = tb.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 4))

        tb.Button(toolbar, text="Open MANY files — Shift/Ctrl select", bootstyle="primary", command=self.open_files).pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(toolbar, text="Add many files", bootstyle="secondary", command=self.add_files).pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(toolbar, text="Open folder", bootstyle="secondary", command=self.open_folder).pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(toolbar, text="Reload all", bootstyle="secondary", command=self.reload_all_files).pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(toolbar, text="Plot", bootstyle="success", command=self.plot_selected).pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(toolbar, text="Save plot", bootstyle="secondary", command=self.save_plot).pack(side=tk.LEFT, padx=(0, 6))
        tb.Button(toolbar, text="Export plotted data", bootstyle="secondary", command=self.export_plotted_data).pack(side=tk.LEFT, padx=(0, 12))
        tb.Button(toolbar, text="Full-screen spike editor", bootstyle="warning", command=self.open_spike_editor).pack(side=tk.LEFT, padx=(0, 12))

        tb.Label(toolbar, textvariable=self.file_label_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _label_frame_inner(self, parent: tk.Widget, title: str, fill: str = tk.X, expand: bool = False) -> tb.Frame:
        box = tb.LabelFrame(parent, text=title)
        box.pack(fill=fill, expand=expand, padx=0, pady=(0, 10))
        inner = tb.Frame(box)
        inner.pack(fill=tk.BOTH, expand=True, padx=9, pady=8)
        return inner

    def _row(self, parent: tk.Widget, label: str, widget: tk.Widget, label_width: int = 13) -> None:
        row = tb.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        tb.Label(row, text=label, width=label_width, anchor=tk.W).pack(side=tk.LEFT)
        widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_left_panel(self, parent: tk.Widget) -> None:
        files_box = self._label_frame_inner(parent, "1. Loaded files", fill=tk.BOTH, expand=True)

        file_btns = tb.Frame(files_box)
        file_btns.pack(fill=tk.X, pady=(0, 7))
        tb.Button(file_btns, text="Open MANY", bootstyle="primary", command=self.open_files).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        tb.Button(file_btns, text="Add many", bootstyle="secondary", command=self.add_files).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        tb.Button(file_btns, text="Open folder", bootstyle="secondary", command=self.open_folder).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        tb.Button(file_btns, text="Clear", bootstyle="secondary", command=self.clear_files).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        tb.Button(file_btns, text="Remove active", bootstyle="secondary", command=self.remove_active_file).grid(row=2, column=0, columnspan=2, sticky="ew", padx=0, pady=2)
        file_btns.columnconfigure(0, weight=1)
        file_btns.columnconfigure(1, weight=1)

        tree_frame = tb.Frame(files_box)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.files_tree = tb.Treeview(
            tree_frame,
            columns=("file", "temp", "rows", "cols", "y"),
            show="headings",
            selectmode="browse",
            height=8,
        )
        self.files_tree.heading("file", text="File")
        self.files_tree.heading("temp", text="Temp")
        self.files_tree.heading("rows", text="Rows")
        self.files_tree.heading("cols", text="Cols")
        self.files_tree.heading("y", text="Y")
        self.files_tree.column("file", width=170, stretch=True, anchor=tk.W)
        self.files_tree.column("temp", width=66, stretch=False, anchor=tk.E)
        self.files_tree.column("rows", width=62, stretch=False, anchor=tk.E)
        self.files_tree.column("cols", width=55, stretch=False, anchor=tk.E)
        self.files_tree.column("y", width=45, stretch=False, anchor=tk.E)
        yscroll = tb.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=yscroll.set)
        self.files_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.files_tree.bind("<<TreeviewSelect>>", self._on_active_file_selected)

        parse = self._label_frame_inner(parent, "2. File parsing")
        self._row(parse, "Delimiter", tb.Combobox(parse, textvariable=self.delimiter_var, values=DELIMITER_OPTIONS, state="readonly", width=16))
        self._row(parse, "Header", tb.Combobox(parse, textvariable=self.header_var, values=HEADER_OPTIONS, state="readonly", width=16))
        self._row(parse, "Decimal", tb.Combobox(parse, textvariable=self.decimal_var, values=DECIMAL_OPTIONS, state="readonly", width=16))
        self._row(parse, "Skip rows", tb.Entry(parse, textvariable=self.skiprows_var, width=8))
        self._row(parse, "Preview rows", tb.Entry(parse, textvariable=self.preview_rows_var, width=8))
        self._row(parse, "Preview cols", tb.Entry(parse, textvariable=self.preview_cols_var, width=8))
        tb.Button(parse, text="Reload all with these settings", bootstyle="secondary", command=self.reload_all_files).pack(fill=tk.X, pady=(8, 0))

        plot = self._label_frame_inner(parent, "3. Plot options")
        self._row(
            plot,
            "Mode",
            tb.Combobox(plot, textvariable=self.plot_mode_var, values=PLOT_MODES, state="readonly", width=36),
        )
        tb.Checkbutton(plot, text="Sort by detected temperature", variable=self.sort_temp_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Color by temperature", variable=self.color_by_temp_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Fast display for large data", variable=self.fast_display_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Auto waterfall offset", variable=self.auto_waterfall_offset_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Sort by X before plotting", variable=self.sort_x_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Show grid", variable=self.grid_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Markers", variable=self.markers_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Legend", variable=self.legend_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Log Y axis", variable=self.logy_var).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(plot, text="Mean ± std band for averaged plots", variable=self.std_band_var).pack(anchor=tk.W, pady=2)

        self._row(plot, "Normalize", tb.Combobox(plot, textvariable=self.normalize_var, values=NORMALIZE_OPTIONS, state="readonly", width=16))
        self._row(plot, "Max plot pts", tb.Entry(plot, textvariable=self.max_plot_points_var, width=8))
        self._row(plot, "Line width", tb.Entry(plot, textvariable=self.linewidth_var, width=8))

        offset_row = tb.Frame(plot)
        offset_row.pack(fill=tk.X, pady=3)
        tb.Checkbutton(offset_row, text="Offset traces", variable=self.offset_var).pack(side=tk.LEFT)
        tb.Entry(offset_row, textvariable=self.offset_value_var, width=8).pack(side=tk.RIGHT)

        tb.Label(plot, text="Title").pack(anchor=tk.W, pady=(8, 2))
        tb.Entry(plot, textvariable=self.title_var).pack(fill=tk.X)
        tb.Label(plot, text="X label").pack(anchor=tk.W, pady=(8, 2))
        tb.Entry(plot, textvariable=self.xlabel_var).pack(fill=tk.X)
        tb.Label(plot, text="Y label").pack(anchor=tk.W, pady=(8, 2))
        tb.Entry(plot, textvariable=self.ylabel_var).pack(fill=tk.X)
        tb.Button(plot, text="Plot", bootstyle="success", command=self.plot_selected).pack(fill=tk.X, pady=(10, 0))

        theme = self._label_frame_inner(parent, "4. Appearance")
        themes = sorted(self.style.theme_names())
        theme_cb = tb.Combobox(theme, textvariable=self.theme_var, values=themes, state="readonly", width=16)
        theme_cb.pack(fill=tk.X)
        theme_cb.bind("<<ComboboxSelected>>", lambda _e: self._change_theme())

    def _build_center_panel(self, parent: tk.Widget) -> None:
        vertical = tb.Panedwindow(parent, orient=tk.VERTICAL)
        vertical.pack(fill=tk.BOTH, expand=True)

        data_area = tb.Frame(vertical)
        plot_area = tb.Frame(vertical)
        vertical.add(data_area)
        vertical.add(plot_area)
        self.after(300, lambda: self._safe_set_vertical_sash(vertical))

        self._build_data_preview(data_area)
        self._build_plot_area(plot_area)

    def _safe_set_vertical_sash(self, pane: tb.Panedwindow) -> None:
        try:
            pane.sashpos(0, 430)
        except Exception:
            pass

    def _build_data_preview(self, parent: tk.Widget) -> None:
        header = tb.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 5))
        tb.Label(header, text="Active file preview", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        tb.Label(header, textvariable=self.active_file_var, anchor=tk.W).pack(side=tk.LEFT, padx=(14, 0), fill=tk.X, expand=True)
        tb.Label(header, text="Click a preview header to highlight that column on the right.").pack(side=tk.RIGHT)

        table_frame = tb.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_tree = tb.Treeview(table_frame, show="headings", selectmode="browse")
        yscroll = tb.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        xscroll = tb.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _build_plot_area(self, parent: tk.Widget) -> None:
        header = tb.Frame(parent)
        header.pack(fill=tk.X, pady=(5, 5))
        tb.Label(header, text="Plot", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        tb.Button(header, text="Clear", bootstyle="secondary", command=self.clear_plot).pack(side=tk.RIGHT)

        self.figure = Figure(figsize=(8.8, 4.7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self._setup_default_axes()

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tb.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.nav_toolbar.update()
        self.nav_toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_right_panel(self, parent: tk.Widget) -> None:
        top = tb.LabelFrame(parent, text="Column roles for active file")
        top.pack(fill=tk.BOTH, expand=True)
        inner = tb.Frame(top)
        inner.pack(fill=tk.BOTH, expand=True, padx=9, pady=8)

        tb.Label(inner, text="Each file has its own X/Y selection. Use the apply button if all TXT files have the same layout.", wraplength=370, anchor=tk.W).pack(fill=tk.X, pady=(0, 8))

        xrow = tb.Frame(inner)
        xrow.pack(fill=tk.X, pady=(0, 8))
        tb.Label(xrow, text="X column", width=9, anchor=tk.W).pack(side=tk.LEFT)
        self.x_combo = tb.Combobox(xrow, textvariable=self.x_combo_var, values=[], state="readonly")
        self.x_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.x_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_x_combo_changed())

        button_grid = tb.Frame(inner)
        button_grid.pack(fill=tk.X, pady=(0, 8))
        tb.Button(button_grid, text="Highlighted → X", bootstyle="primary", command=self.set_highlighted_as_x).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        tb.Button(button_grid, text="Toggle highlighted Y", bootstyle="secondary", command=self.toggle_highlighted_y).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        tb.Button(button_grid, text="All numeric except X → Y", bootstyle="secondary", command=self.select_all_numeric_y).grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        tb.Button(button_grid, text="Clear Y", bootstyle="secondary", command=self.clear_y_selection).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=2)
        tb.Button(button_grid, text="Only highlighted Y", bootstyle="secondary", command=self.only_highlighted_y).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=2)
        tb.Button(button_grid, text="Apply active roles to all files by column number", bootstyle="warning", command=self.apply_active_roles_to_all_files).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        tb.Button(button_grid, text="Reset auto roles for all files", bootstyle="secondary", command=self.reset_auto_roles_all_files).grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)
        button_grid.columnconfigure(0, weight=1)
        button_grid.columnconfigure(1, weight=1)

        tb.Label(inner, text="Double-click a column below to toggle Y.", anchor=tk.W).pack(fill=tk.X, pady=(0, 4))

        tree_frame = tb.Frame(inner)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.roles_tree = tb.Treeview(
            tree_frame,
            columns=("role", "name", "type", "nonempty", "first"),
            show="headings",
            selectmode="extended",
            height=18,
        )
        self.roles_tree.heading("role", text="Role")
        self.roles_tree.heading("name", text="Column")
        self.roles_tree.heading("type", text="Type")
        self.roles_tree.heading("nonempty", text="Non-empty")
        self.roles_tree.heading("first", text="First value")
        self.roles_tree.column("role", width=58, stretch=False, anchor=tk.CENTER)
        self.roles_tree.column("name", width=150, stretch=True, anchor=tk.W)
        self.roles_tree.column("type", width=78, stretch=False, anchor=tk.W)
        self.roles_tree.column("nonempty", width=82, stretch=False, anchor=tk.CENTER)
        self.roles_tree.column("first", width=90, stretch=False, anchor=tk.E)
        yscroll = tb.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.roles_tree.yview)
        xscroll = tb.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.roles_tree.xview)
        self.roles_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.roles_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.roles_tree.bind("<Double-1>", lambda _e: self.toggle_highlighted_y())

        summary_box = tb.LabelFrame(parent, text="Summary")
        summary_box.pack(fill=tk.X, pady=(10, 0))
        summary_inner = tb.Frame(summary_box)
        summary_inner.pack(fill=tk.X, padx=9, pady=8)
        tb.Label(summary_inner, textvariable=self.summary_var, justify=tk.LEFT, anchor=tk.W).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _ask_many_files(self, title: str) -> List[Path]:
        """Open the real multi-file dialog.

        This uses tkinter.filedialog.askopenfilenames directly. On Windows this is
        the function that enables Ctrl-click and Shift-click range selection.
        """
        dialog_title = "SELECT MANY TXT FILES — Shift-click range or Ctrl-click individual files"

        raw = filedialog.askopenfilenames(
            parent=self,
            title=dialog_title,
            filetypes=FILE_DIALOG_TYPES,
        )

        if not raw:
            return []

        # Tk usually returns a tuple. Some builds return a Tcl-list string.
        if isinstance(raw, (tuple, list)):
            items = [str(p) for p in raw if str(p).strip()]
        else:
            try:
                items = [str(p) for p in self.tk.splitlist(str(raw)) if str(p).strip()]
            except Exception:
                items = [str(raw)]

        seen = set()
        paths: List[Path] = []
        for item in items:
            p = Path(item)
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                seen.add(key)
                paths.append(p)

        # This is only an instruction, not an error. It helps diagnose whether the
        # user accidentally opened the old single-file version or clicked only one file.
        if len(paths) == 1:
            self.status_var.set(
                "Only 1 file selected. In the dialog use Shift-click to select a range, or use Open folder."
            )

        return paths

    def open_files(self) -> None:
        paths = self._ask_many_files("Open one or more TXT/CSV/XLSX data files")
        if not paths:
            return
        self._load_paths(paths, replace=True)

    def add_files(self) -> None:
        paths = self._ask_many_files("Add one or more TXT/CSV/XLSX data files")
        if not paths:
            return
        self._load_paths(paths, replace=False)

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self,
            title="Open a folder containing TXT/CSV/XLSX data files",
        )
        if not folder:
            return
        root = Path(folder)
        paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not paths:
            messagebox.showwarning(
                "No data files",
                "No supported data files were found in that folder.\n\nSupported: " + ", ".join(SUPPORTED_EXTENSIONS),
            )
            return
        self._load_paths(paths, replace=True)

    def _load_paths(self, paths: Sequence[Path], replace: bool) -> None:
        loaded: List[LoadedDataFile] = []
        errors: List[str] = []

        if replace:
            self._next_load_order = 0

        for path in paths:
            try:
                lf = self._load_one_file(path)
                lf.load_order = self._next_load_order
                self._next_load_order += 1
                loaded.append(lf)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        if replace:
            self.files = loaded
            self.active_index = None
        else:
            existing = {str(f.path.resolve()) for f in self.files if f.path.exists()}
            for lf in loaded:
                key = str(lf.path.resolve()) if lf.path.exists() else str(lf.path)
                if key not in existing:
                    self.files.append(lf)
                    existing.add(key)

        self._apply_file_ordering()

        if self.files:
            self.active_index = 0 if self.active_index is None or self.active_index >= len(self.files) else self.active_index
        else:
            self.active_index = None

        self._refresh_all_views()
        self._update_title_and_labels()

        if errors:
            messagebox.showwarning("Some files could not be loaded", "Loaded the valid files, but these failed:\n\n" + "\n".join(errors[:12]))

        detected = sum(1 for f in self.files if f.temperature is not None)
        self.status_var.set(
            f"Loaded {len(self.files)} file(s); detected temperatures in {detected}/{len(self.files)}. Choose a plot mode or open the spike editor."
        )
        if self.files:
            # Auto-plot if the number of traces is not too huge.
            total_y = sum(len(f.y_cols) for f in self.files)
            if total_y <= 80:
                self.plot_selected(show_popup_on_error=False)

    def _load_one_file(self, path: Path) -> LoadedDataFile:
        df = self._read_file(path)
        df = self._make_columns_unique_and_readable(df)
        df = self._coerce_numeric_columns(df)
        if df.empty:
            raise ValueError("Loaded table is empty. Try delimiter/header/skip rows.")

        numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        temperature, temperature_label = extract_temperature_from_filename(path.name)
        lf = LoadedDataFile(
            path=path,
            df=df.copy(deep=True),
            raw_df=df.copy(deep=True),
            numeric_cols=numeric_cols,
            temperature=temperature,
            temperature_label=temperature_label,
        )
        self._auto_assign_columns(lf)
        return lf

    def reload_all_files(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "Open one or more files first.")
            return

        old_files = self.files[:]
        old_active = self.active_index
        old_active_key = self._file_identity(old_files[old_active]) if old_active is not None and 0 <= old_active < len(old_files) else None
        reloaded: List[LoadedDataFile] = []
        errors: List[str] = []

        for old in old_files:
            try:
                new = self._load_one_file(old.path)
                # Try to preserve manually edited roles by column name.
                if old.x_col in new.df.columns:
                    new.x_col = old.x_col
                if old.y_cols:
                    new.y_cols = [c for c in old.y_cols if c in new.df.columns and c != new.x_col]
                new.load_order = old.load_order
                reloaded.append(new)
            except Exception as exc:
                errors.append(f"{old.path.name}: {exc}")

        self.files = reloaded
        self.active_index = None
        self._apply_file_ordering()
        if self.files:
            if old_active_key is not None:
                for idx, lf in enumerate(self.files):
                    if self._file_identity(lf) == old_active_key:
                        self.active_index = idx
                        break
            if self.active_index is None:
                self.active_index = min(old_active if old_active is not None else 0, len(self.files) - 1)
        else:
            self.active_index = None

        self._refresh_all_views()
        self._update_title_and_labels()

        if errors:
            messagebox.showwarning("Some files could not be reloaded", "These files failed:\n\n" + "\n".join(errors[:12]))
        self.status_var.set(f"Reloaded {len(self.files)} file(s).")

    def remove_active_file(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        removed = lf.name
        idx = self.active_index or 0
        del self.files[idx]
        if not self.files:
            self.active_index = None
        else:
            self.active_index = min(idx, len(self.files) - 1)
        self._refresh_all_views()
        self._update_title_and_labels()
        self.status_var.set(f"Removed {removed}.")

    def clear_files(self) -> None:
        self.files = []
        self.active_index = None
        self.clear_plot()
        self._refresh_all_views()
        self._update_title_and_labels()
        self.status_var.set("All files cleared.")

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _read_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        skiprows = self._safe_int(self.skiprows_var.get(), default=0, minimum=0)
        header = self._determine_header(path, skiprows)

        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path, header=header, skiprows=skiprows)

        sep = self._determine_separator(path, skiprows)
        decimal = self.decimal_var.get()
        kwargs = dict(
            header=header,
            skiprows=skiprows,
            decimal=decimal,
            engine="python",
            on_bad_lines="warn",
        )
        if sep == "WHITESPACE":
            return pd.read_csv(path, sep=r"\s+", **kwargs)
        return pd.read_csv(path, sep=sep, **kwargs)

    def _determine_separator(self, path: Path, skiprows: int) -> str:
        choice = self.delimiter_var.get()
        mapping = {
            "Tab": "\t",
            "Comma": ",",
            "Semicolon": ";",
            "Pipe": "|",
            "Whitespace": "WHITESPACE",
        }
        if choice in mapping:
            return mapping[choice]
        return self._guess_separator(path, skiprows)

    def _determine_header(self, path: Path, skiprows: int) -> Optional[int]:
        choice = self.header_var.get()
        if choice == "No header":
            return None
        if choice == "First row is header":
            return 0
        return self._guess_header(path, skiprows)

    def _guess_separator(self, path: Path, skiprows: int) -> str:
        lines = self._read_sample_lines(path, skiprows=skiprows, max_lines=25)
        if not lines:
            return "WHITESPACE"

        candidates: List[Tuple[str, str]] = [
            ("\t", "Tab"),
            (";", "Semicolon"),
            (",", "Comma"),
            ("|", "Pipe"),
            ("WHITESPACE", "Whitespace"),
        ]

        best_sep = "WHITESPACE"
        best_score = -1.0
        for sep, _name in candidates:
            counts = []
            for line in lines:
                tokens = self._split_line(line, sep)
                if len(tokens) > 1:
                    counts.append(len(tokens))
            if not counts:
                score = 0.0
            else:
                median_count = float(np.median(counts))
                consistency = counts.count(int(round(median_count))) / max(len(counts), 1)
                score = median_count * consistency
            if score > best_score:
                best_score = score
                best_sep = sep
        return best_sep

    def _guess_header(self, path: Path, skiprows: int) -> Optional[int]:
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            try:
                sample = pd.read_excel(path, header=None, skiprows=skiprows, nrows=2)
                if sample.empty:
                    return None
                first_values = [str(v).strip() for v in sample.iloc[0].tolist()]
                numeric_fraction = self._numeric_fraction(first_values)
                return None if numeric_fraction >= 0.60 else 0
            except Exception:
                return None

        sep = self._determine_separator(path, skiprows)
        lines = self._read_sample_lines(path, skiprows=skiprows, max_lines=2)
        if not lines:
            return None
        first_tokens = self._split_line(lines[0], sep)
        if not first_tokens:
            return None
        numeric_fraction = self._numeric_fraction(first_tokens)
        return None if numeric_fraction >= 0.60 else 0

    def _read_sample_lines(self, path: Path, skiprows: int, max_lines: int) -> List[str]:
        lines: List[str] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < skiprows:
                    continue
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                if len(lines) >= max_lines:
                    break
        return lines

    @staticmethod
    def _split_line(line: str, sep: str) -> List[str]:
        line = line.strip()
        if not line:
            return []
        if sep == "WHITESPACE":
            return re.split(r"\s+", line)
        return [part.strip() for part in line.split(sep)]

    def _numeric_fraction(self, values: Sequence[str]) -> float:
        if not values:
            return 0.0
        ok = 0
        for value in values:
            text = str(value).strip()
            if text == "":
                continue
            text_dot = text.replace(",", ".")
            try:
                float(text_dot)
                ok += 1
            except Exception:
                pass
        return ok / max(len(values), 1)

    def _make_columns_unique_and_readable(self, df: pd.DataFrame) -> pd.DataFrame:
        old_cols = list(df.columns)
        new_cols: List[str] = []
        seen: Dict[str, int] = {}
        header_is_auto_or_none = self.header_var.get() in ("Auto", "No header")

        for idx, col in enumerate(old_cols, start=1):
            if isinstance(col, int) or (header_is_auto_or_none and str(col).strip().isdigit()):
                name = f"Column {idx}"
            else:
                name = str(col).strip()
                if not name or name.lower().startswith("unnamed"):
                    name = f"Column {idx}"

            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            new_cols.append(name)

        df = df.copy()
        df.columns = new_cols
        return df

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        decimal = self.decimal_var.get()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            s = df[col].astype(str).str.strip()
            if decimal == ",":
                s_num = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            else:
                # Conservative: remove commas as thousands separators for dot-decimal files.
                s_num = s.str.replace(",", "", regex=False)
            converted = pd.to_numeric(s_num, errors="coerce")
            fraction = converted.notna().mean() if len(converted) else 0
            if fraction >= 0.70:
                df[col] = converted
        return df

    def _auto_assign_columns(self, lf: LoadedDataFile) -> None:
        if lf.df.empty:
            lf.x_col = None
            lf.y_cols = []
            return
        if lf.numeric_cols:
            lf.x_col = lf.numeric_cols[0]
            lf.y_cols = [c for c in lf.numeric_cols[1:] if c != lf.x_col]
        else:
            lf.x_col = str(lf.df.columns[0])
            lf.y_cols = []

    def _file_identity(self, lf: Optional[LoadedDataFile]) -> Optional[str]:
        if lf is None:
            return None
        try:
            return str(lf.path.resolve())
        except Exception:
            return str(lf.path)

    @staticmethod
    def _temperature_sort_key(lf: LoadedDataFile) -> Tuple[int, float, int, str]:
        if lf.temperature is None or not np.isfinite(lf.temperature):
            return (1, float("inf"), lf.load_order, lf.name.lower())
        return (0, float(lf.temperature), lf.load_order, lf.name.lower())

    def _apply_file_ordering(self) -> None:
        if not self.files:
            return
        active_key = self._file_identity(self._active_file())
        if self.sort_temp_var.get():
            self.files.sort(key=self._temperature_sort_key)
        else:
            self.files.sort(key=lambda lf: (lf.load_order, lf.name.lower()))
        if active_key is not None:
            for idx, lf in enumerate(self.files):
                if self._file_identity(lf) == active_key:
                    self.active_index = idx
                    break

    def _on_sort_temperature_changed(self) -> None:
        if not self.files:
            return
        self._apply_file_ordering()
        self._refresh_all_views()
        self._update_title_and_labels()
        if self.spike_window is not None:
            self._refresh_spike_file_choices()
            self._refresh_spike_plot()
        self.status_var.set(
            "Loaded files are sorted by detected temperature." if self.sort_temp_var.get() else "Loaded files restored to original load order."
        )

    # ------------------------------------------------------------------
    # View updates
    # ------------------------------------------------------------------
    def _refresh_all_views(self) -> None:
        self._update_files_tree()
        self._update_active_file_views()
        self._update_summary()

    def _active_file(self) -> Optional[LoadedDataFile]:
        if self.active_index is None:
            return None
        if 0 <= self.active_index < len(self.files):
            return self.files[self.active_index]
        return None

    def _update_files_tree(self) -> None:
        tree = self.files_tree
        tree.delete(*tree.get_children())
        self.file_iid_to_index.clear()
        for idx, lf in enumerate(self.files):
            iid = f"file_{idx}"
            self.file_iid_to_index[iid] = idx
            rows, cols = lf.df.shape
            temp_text = lf.temperature_label or "UNKNOWN"
            tree.insert("", tk.END, iid=iid, values=(lf.name, temp_text, f"{rows:,}", f"{cols:,}", f"{len(lf.y_cols):,}"))
        if self.active_index is not None and self.files:
            iid = f"file_{self.active_index}"
            if iid in self.file_iid_to_index:
                tree.selection_set(iid)
                tree.focus(iid)
                tree.see(iid)

    def _on_active_file_selected(self, _event: object = None) -> None:
        selected = list(self.files_tree.selection())
        if not selected:
            return
        iid = selected[0]
        idx = self.file_iid_to_index.get(iid)
        if idx is None:
            return
        if idx != self.active_index:
            self.active_index = idx
            self._update_active_file_views()
            self._update_title_and_labels()
            self._update_summary()
            self.status_var.set(f"Active file: {self.files[idx].name}")

    def _update_active_file_views(self) -> None:
        self._update_x_combo()
        self._update_role_tree()
        self._update_preview_tree()

    def _update_title_and_labels(self) -> None:
        if not self.files:
            self.file_label_var.set("No files loaded")
            self.active_file_var.set("No active file")
            self.title_var.set("")
            self.xlabel_var.set("")
            return

        detected = sum(1 for f in self.files if f.temperature is not None)
        self.file_label_var.set(f"{len(self.files)} file(s) loaded; temperatures detected: {detected}/{len(self.files)}")
        lf = self._active_file()
        if lf is not None:
            self.active_file_var.set(str(lf.path))
            self.xlabel_var.set(lf.x_col or "")
        if len(self.files) == 1:
            self.title_var.set(self.files[0].stem)
        else:
            self.title_var.set(f"{len(self.files)} files")

    def _update_x_combo(self) -> None:
        lf = self._active_file()
        if lf is None:
            self.x_combo.configure(values=[])
            self.x_combo_var.set("")
            return
        values = [str(c) for c in lf.df.columns]
        self.x_combo.configure(values=values)
        self.x_combo_var.set(lf.x_col or "")

    def _role_for_col(self, lf: LoadedDataFile, col: str) -> str:
        if col == lf.x_col:
            return "X"
        if col in lf.y_cols:
            return "Y"
        return "—"

    def _short(self, text: str, max_len: int = 28) -> str:
        text = str(text)
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _fmt_cell(self, value: object) -> str:
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            val = float(value)
            if abs(val) >= 1e5 or (0 < abs(val) < 1e-3):
                return f"{val:.5E}"
            return f"{val:.7g}"
        return str(value)

    def _update_preview_tree(self) -> None:
        tree = self.preview_tree
        tree.delete(*tree.get_children())
        tree["columns"] = []
        self.preview_id_to_col.clear()

        lf = self._active_file()
        if lf is None:
            return

        df = lf.df
        preview_rows = self._safe_int(self.preview_rows_var.get(), default=500, minimum=1)
        preview_cols = self._safe_int(self.preview_cols_var.get(), default=250, minimum=1)
        preview = df.iloc[:preview_rows, :preview_cols]

        col_ids = [f"C{i}" for i in range(len(preview.columns))]
        tree["columns"] = col_ids

        for col_id, col in zip(col_ids, preview.columns):
            col = str(col)
            self.preview_id_to_col[col_id] = col
            role = self._role_for_col(lf, col)
            label = f"{role}: {col}" if role in ("X", "Y") else col
            tree.heading(col_id, text=self._short(label, 26), command=lambda c=col: self.highlight_column(c))
            series = df[col]
            numeric = pd.api.types.is_numeric_dtype(series)
            tree.column(col_id, width=118 if numeric else 150, minwidth=70, stretch=True, anchor=tk.E if numeric else tk.W)

        for idx, row in preview.iterrows():
            values = [self._fmt_cell(row[col]) for col in preview.columns]
            tree.insert("", tk.END, iid=f"row_{idx}", values=values)

    def _update_role_tree(self) -> None:
        tree = self.roles_tree
        tree.delete(*tree.get_children())
        self.role_id_to_col.clear()
        self.col_to_role_id.clear()

        lf = self._active_file()
        if lf is None:
            return

        rows, _cols = lf.df.shape
        for idx, col in enumerate(lf.df.columns):
            col = str(col)
            role = self._role_for_col(lf, col)
            dtype = "numeric" if col in lf.numeric_cols else "text"
            nonempty = int(lf.df[col].notna().sum())
            first_value = ""
            non_null = lf.df[col].dropna()
            if not non_null.empty:
                first_value = self._fmt_cell(non_null.iloc[0])
            iid = f"col_{idx}"
            self.role_id_to_col[iid] = col
            self.col_to_role_id[col] = iid
            tree.insert("", tk.END, iid=iid, values=(role, col, dtype, f"{nonempty}/{rows}", first_value))

    def _update_summary(self) -> None:
        if not self.files:
            self.summary_var.set("No data loaded yet.")
            return
        lf = self._active_file()
        total_rows = sum(f.df.shape[0] for f in self.files)
        total_y = sum(len(f.y_cols) for f in self.files)
        total_available_y = sum(len(self._all_numeric_y_columns_for_file(f)) for f in self.files)
        detected = sum(1 for f in self.files if f.temperature is not None)
        if lf is None:
            active_text = "None"
        else:
            rows, cols = lf.df.shape
            temp_text = lf.temperature_label or "UNKNOWN"
            active_available_y = len(self._all_numeric_y_columns_for_file(lf))
            active_text = (
                f"{lf.name}\n"
                f"Temperature: {temp_text}\n"
                f"Rows: {rows:,}; Columns: {cols:,}\n"
                f"Numeric columns: {len(lf.numeric_cols):,}\n"
                f"X: {lf.x_col or 'None'}\n"
                f"Y selected in active file: {len(lf.y_cols):,}\n"
                f"Available numeric Y spectra: {active_available_y:,}"
            )
        self.summary_var.set(
            f"Files: {len(self.files):,}\n"
            f"Total rows: {total_rows:,}\n"
            f"Total selected Y traces: {total_y:,}\n\n"
            f"Total numeric Y spectra available: {total_available_y:,}\n\n"
            f"Detected temperatures: {detected:,}/{len(self.files):,}\n"
            f"Temperature sort: {'on' if self.sort_temp_var.get() else 'off'}\n\n"
            f"Active:\n{active_text}"
        )

    def highlight_column(self, col: str) -> None:
        iid = self.col_to_role_id.get(col)
        if not iid:
            return
        self.roles_tree.selection_set(iid)
        self.roles_tree.focus(iid)
        self.roles_tree.see(iid)
        self.status_var.set(f"Highlighted column: {col}. Use 'Highlighted → X' or 'Toggle highlighted Y'.")

    # ------------------------------------------------------------------
    # Column role operations
    # ------------------------------------------------------------------
    def _selected_role_columns(self) -> List[str]:
        selected = list(self.roles_tree.selection())
        return [self.role_id_to_col[iid] for iid in selected if iid in self.role_id_to_col]

    def _on_x_combo_changed(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        col = self.x_combo_var.get()
        if not col:
            return
        lf.x_col = col
        lf.y_cols = [c for c in lf.y_cols if c != col]
        self.xlabel_var.set(col)
        self._refresh_all_views()
        self.status_var.set(f"X column set to {col} for {lf.name}.")

    def set_highlighted_as_x(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        cols = self._selected_role_columns()
        if not cols:
            messagebox.showinfo("No column selected", "Highlight a column in the right panel first.")
            return
        col = cols[0]
        lf.x_col = col
        lf.y_cols = [c for c in lf.y_cols if c != col]
        self.x_combo_var.set(col)
        self.xlabel_var.set(col)
        self._refresh_all_views()
        self.status_var.set(f"X column set to {col} for {lf.name}.")

    def toggle_highlighted_y(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        cols = self._selected_role_columns()
        if not cols:
            return
        for col in cols:
            if col == lf.x_col:
                continue
            if col in lf.y_cols:
                lf.y_cols.remove(col)
            else:
                lf.y_cols.append(col)
        self._refresh_all_views()
        self.status_var.set(f"Y columns selected in {lf.name}: {len(lf.y_cols)}.")

    def only_highlighted_y(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        lf.y_cols = [c for c in self._selected_role_columns() if c != lf.x_col]
        self._refresh_all_views()
        self.status_var.set(f"Y columns selected in {lf.name}: {len(lf.y_cols)}.")

    def select_all_numeric_y(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        lf.y_cols = [c for c in lf.numeric_cols if c != lf.x_col]
        self._refresh_all_views()
        self.status_var.set(f"All numeric columns except X selected as Y in {lf.name}: {len(lf.y_cols)}.")

    def clear_y_selection(self) -> None:
        lf = self._active_file()
        if lf is None:
            return
        lf.y_cols = []
        self._refresh_all_views()
        self.status_var.set(f"Y selection cleared for {lf.name}.")

    def apply_active_roles_to_all_files(self) -> None:
        active = self._active_file()
        if active is None:
            return
        if active.x_col is None:
            messagebox.showwarning("No X column", "Set an X column in the active file first.")
            return

        active_cols = list(active.df.columns)
        try:
            x_idx = active_cols.index(active.x_col)
        except ValueError:
            messagebox.showerror("Bad X column", "The active X column is not in the active file.")
            return
        y_indices = [active_cols.index(c) for c in active.y_cols if c in active_cols]

        changed = 0
        skipped = 0
        for lf in self.files:
            cols = list(lf.df.columns)
            if x_idx >= len(cols):
                skipped += 1
                continue
            lf.x_col = str(cols[x_idx])
            lf.y_cols = [str(cols[i]) for i in y_indices if i < len(cols) and str(cols[i]) != lf.x_col]
            changed += 1

        self._refresh_all_views()
        self.status_var.set(f"Applied active column positions to {changed} file(s). Skipped {skipped} file(s).")

    def reset_auto_roles_all_files(self) -> None:
        for lf in self.files:
            self._auto_assign_columns(lf)
        self._refresh_all_views()
        self.status_var.set("Reset auto roles for all files: first numeric column = X, remaining numeric columns = Y.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _setup_default_axes(self) -> None:
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self._apply_grid_setting(default_on=True)

    def _reset_main_plot_axes(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._current_colorbar = None
        self._setup_default_axes()

    def _apply_grid_setting(self, default_on: bool = False) -> None:
        enabled = bool(self.grid_var.get()) if not default_on else True
        if enabled:
            self.ax.grid(True, alpha=0.3)
        else:
            self.ax.grid(False)

    def _files_for_plot(self, force_temperature_sort: bool = False) -> List[LoadedDataFile]:
        files = list(self.files)
        if force_temperature_sort or self.sort_temp_var.get():
            return sorted(files, key=self._temperature_sort_key)
        return sorted(files, key=lambda lf: (lf.load_order, lf.name.lower()))

    def _get_temperature_cmap(self):
        if mpl_colormaps is not None:
            for name in ("turbo", "plasma"):
                try:
                    return mpl_colormaps.get_cmap(name)
                except Exception:
                    pass
        return LinearSegmentedColormap.from_list(
            "raman_temperature",
            ["#243b9f", "#6121a8", "#c73e84", "#ef7d1a", "#d7191c"],
        )

    def _temperature_norm_for_files(self, files: Sequence[LoadedDataFile]) -> Optional[Normalize]:
        temps = [float(lf.temperature) for lf in files if lf.temperature is not None and np.isfinite(lf.temperature)]
        if not temps:
            return None
        vmin = min(temps)
        vmax = max(temps)
        if vmin == vmax:
            pad = max(abs(vmin) * 0.02, 1.0)
            vmin -= pad
            vmax += pad
        return Normalize(vmin=vmin, vmax=vmax)

    def _temperature_color_context(
        self,
        files: Sequence[LoadedDataFile],
        force: bool = False,
    ) -> Tuple[object, Optional[Normalize]]:
        if not force and not self.color_by_temp_var.get():
            return None, None
        norm = self._temperature_norm_for_files(files)
        if norm is None:
            return None, None
        return self._get_temperature_cmap(), norm

    def _color_for_file(self, lf: LoadedDataFile, cmap: object, norm: Optional[Normalize]) -> Optional[object]:
        if cmap is None or norm is None:
            return None
        if lf.temperature is None or not np.isfinite(lf.temperature):
            return "#8c8c8c"
        return cmap(norm(float(lf.temperature)))

    def _add_temperature_colorbar(self, cmap: object, norm: Optional[Normalize]) -> None:
        if cmap is None or norm is None:
            return
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        self._current_colorbar = self.figure.colorbar(scalar, ax=self.ax, pad=0.018, fraction=0.045)
        self._current_colorbar.set_label("Temperature (K)")

    def _manual_offset_value(self) -> float:
        try:
            return float(self.offset_value_var.get())
        except Exception:
            return 0.0

    def _auto_waterfall_offset(self, y_arrays: Sequence[np.ndarray]) -> float:
        values = []
        for y in y_arrays:
            arr = np.asarray(y, dtype=float)
            values.append(arr[np.isfinite(arr)])
        if not values:
            return 1.0
        finite_values = [arr for arr in values if arr.size]
        if not finite_values:
            return 1.0
        joined = np.concatenate(finite_values)
        if joined.size == 0:
            return 1.0
        p5, p95 = np.nanpercentile(joined, [5, 95])
        robust_range = float(p95 - p5)
        ymin = float(np.nanmin(joined))
        ymax = float(np.nanmax(joined))
        full_range = ymax - ymin
        base_range = robust_range if np.isfinite(robust_range) and robust_range > 0 else full_range
        if not np.isfinite(base_range) or base_range <= 0:
            base_range = 1.0
        return 0.08 * base_range

    def _max_plot_points(self) -> int:
        return self._safe_int(self.max_plot_points_var.get(), default=2500, minimum=200)

    def _max_spike_display_points(self) -> int:
        return self._safe_int(self.max_spike_display_points_var.get(), default=3500, minimum=300)

    def _max_spike_context_curves(self) -> int:
        return self._safe_int(self.max_spike_context_var.get(), default=300, minimum=1)

    def _spike_stack_visible_count(self) -> int:
        return self._safe_int(self.spike_stack_visible_var.get(), default=6, minimum=1)

    def _spike_is_stacked(self) -> bool:
        return self.spike_view_mode_var.get() == "Stacked"

    def _display_indices_for_curve(self, x: np.ndarray, y: np.ndarray, max_points: int) -> np.ndarray:
        n = int(len(x))
        if n == 0:
            return np.array([], dtype=int)
        if not self.fast_display_var.get() or max_points <= 0 or n <= max_points:
            return np.arange(n, dtype=int)

        bucket_count = max(1, int(max_points) // 2)
        edges = np.linspace(0, n, bucket_count + 1, dtype=int)
        chosen: List[int] = [0, n - 1]
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            segment = np.asarray(y[start:end], dtype=float)
            finite = np.isfinite(segment)
            if not np.any(finite):
                chosen.append(start)
                continue
            local = np.flatnonzero(finite)
            finite_values = segment[finite]
            min_idx = int(start + local[int(np.argmin(finite_values))])
            max_idx = int(start + local[int(np.argmax(finite_values))])
            chosen.extend((min_idx, max_idx))

        indices = np.array(sorted(set(chosen)), dtype=int)
        if indices.size > max_points:
            keep = np.linspace(0, indices.size - 1, max_points, dtype=int)
            indices = indices[keep]
        return indices

    def _display_xy(self, x: np.ndarray, y: np.ndarray, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        max_points = self._max_plot_points() if max_points is None else int(max_points)
        idx = self._display_indices_for_curve(x, y, max_points)
        return x[idx], y[idx]

    def plot_selected(self, show_popup_on_error: bool = True) -> None:
        if not self.files:
            if show_popup_on_error:
                messagebox.showinfo("No data", "Open one or more data files first.")
            return

        mode = self.plot_mode_var.get()
        self._reset_main_plot_axes()
        self.last_plot_payload = {"mode": mode, "curves": []}

        try:
            if mode == "All traces from all files":
                plotted, any_nonpositive = self._plot_all_traces()
            elif mode == "Average Y columns in each file":
                plotted, any_nonpositive = self._plot_average_each_file()
            elif mode == "Grand average of file averages":
                plotted, any_nonpositive = self._plot_grand_average()
            elif mode == "Temperature waterfall average per file":
                plotted, any_nonpositive = self._plot_temperature_waterfall_average()
            else:
                plotted, any_nonpositive = self._plot_all_traces()
        except Exception as exc:
            if show_popup_on_error:
                messagebox.showerror("Plot failed", str(exc))
            self.status_var.set(f"Plot failed: {exc}")
            return

        if plotted == 0:
            if show_popup_on_error:
                messagebox.showwarning("Nothing plotted", "No selected Y column had enough numeric data.")
            self.clear_plot()
            return

        self._finish_axes(plotted=plotted, any_nonpositive=any_nonpositive)

        if mode == "Average Y columns in each file":
            self.status_var.set(f"Plotted only averaged curves: one average per file ({plotted} file average(s)).")
        elif mode == "Grand average of file averages":
            self.status_var.set(f"Plotted grand average across {plotted} file average(s).")
        elif mode == "Temperature waterfall average per file":
            self.status_var.set(f"Plotted temperature waterfall averages for {plotted} file(s).")
        else:
            self.status_var.set(f"Plotted {plotted} traces from {len(self.files)} file(s): one trace per selected Y column.")

    def _common_plot_options(self) -> Tuple[float, float, Optional[str]]:
        try:
            linewidth = float(self.linewidth_var.get())
        except Exception:
            linewidth = 1.3
        try:
            offset_step = float(self.offset_value_var.get()) if self.offset_var.get() else 0.0
        except Exception:
            offset_step = 0.0
        marker = "o" if self.markers_var.get() else None
        return linewidth, offset_step, marker

    def _plot_all_traces(self) -> Tuple[int, bool]:
        linewidth, offset_step, marker = self._common_plot_options()
        plotted = 0
        any_nonpositive = False
        files = self._files_for_plot()
        cmap, norm = self._temperature_color_context(files)

        for lf in files:
            if not lf.x_col or not lf.y_cols:
                continue
            file_color = self._color_for_file(lf, cmap, norm)
            for y_col in lf.y_cols:
                curve = self._single_y_curve(lf, y_col)
                if curve is None:
                    continue
                x = curve["x"].to_numpy(dtype=float)
                y = curve["y"].to_numpy(dtype=float)
                if offset_step:
                    y = y + plotted * offset_step
                if np.any(y <= 0):
                    any_nonpositive = True
                label = f"{lf.stem}: {y_col}" if len(self.files) > 1 else str(y_col)
                plot_kwargs = {"linewidth": linewidth, "marker": marker, "markersize": 3, "label": label}
                if file_color is not None:
                    plot_kwargs["color"] = file_color
                x_plot, y_plot = self._display_xy(x, y)
                self.ax.plot(x_plot, y_plot, **plot_kwargs)
                self.last_plot_payload["curves"].append({"name": label, "x": x, "y": y, "temperature": lf.temperature})
                plotted += 1
        if plotted:
            self._add_temperature_colorbar(cmap, norm)
        return plotted, any_nonpositive

    def _plot_average_each_file(self) -> Tuple[int, bool]:
        linewidth, offset_step, marker = self._common_plot_options()
        plotted = 0
        any_nonpositive = False
        files = self._files_for_plot()
        cmap, norm = self._temperature_color_context(files)

        for lf in files:
            curve = self._average_curve_for_file(lf)
            if curve is None:
                continue
            x = curve["x"].to_numpy(dtype=float)
            y = curve["mean"].to_numpy(dtype=float)
            std = curve["std"].to_numpy(dtype=float) if "std" in curve else np.full_like(y, np.nan)
            if offset_step:
                y = y + plotted * offset_step
            if np.any(y <= 0):
                any_nonpositive = True

            label = f"{lf.stem} mean"
            plot_kwargs = {"linewidth": linewidth, "marker": marker, "markersize": 3, "label": label}
            file_color = self._color_for_file(lf, cmap, norm)
            if file_color is not None:
                plot_kwargs["color"] = file_color
            display_idx = self._display_indices_for_curve(x, y, self._max_plot_points())
            x_plot = x[display_idx]
            y_plot = y[display_idx]
            line = self.ax.plot(x_plot, y_plot, **plot_kwargs)[0]
            if self.std_band_var.get() and np.isfinite(std).any() and not self.logy_var.get():
                color = line.get_color()
                std_plot = std[display_idx]
                self.ax.fill_between(x_plot, y_plot - std_plot, y_plot + std_plot, alpha=0.18, color=color, linewidth=0)
            self.last_plot_payload["curves"].append({"name": label, "x": x, "y": y, "std": std, "temperature": lf.temperature})
            plotted += 1
        if plotted:
            self._add_temperature_colorbar(cmap, norm)
        return plotted, any_nonpositive

    def _plot_grand_average(self) -> Tuple[int, bool]:
        curves: List[Tuple[LoadedDataFile, pd.DataFrame]] = []
        for lf in self._files_for_plot():
            curve = self._average_curve_for_file(lf)
            if curve is not None:
                curves.append((lf, curve))

        if not curves:
            return 0, False
        if len(curves) == 1:
            # With one file, grand average is just its file average.
            lf, curve = curves[0]
            x = curve["x"].to_numpy(dtype=float)
            y = curve["mean"].to_numpy(dtype=float)
            std = curve["std"].to_numpy(dtype=float) if "std" in curve else np.full_like(y, np.nan)
            linewidth, _offset_step, marker = self._common_plot_options()
            display_idx = self._display_indices_for_curve(x, y, self._max_plot_points())
            x_plot = x[display_idx]
            y_plot = y[display_idx]
            line = self.ax.plot(x_plot, y_plot, linewidth=linewidth, marker=marker, markersize=3, label=f"{lf.stem} mean")[0]
            if self.std_band_var.get() and np.isfinite(std).any() and not self.logy_var.get():
                std_plot = std[display_idx]
                self.ax.fill_between(x_plot, y_plot - std_plot, y_plot + std_plot, alpha=0.18, color=line.get_color(), linewidth=0)
            self.last_plot_payload["curves"].append({"name": f"{lf.stem} mean", "x": x, "y": y, "std": std})
            return 1, bool(np.any(y <= 0))

        sorted_curves = []
        for lf, curve in curves:
            tmp = curve[["x", "mean"]].dropna().sort_values("x")
            tmp = tmp.groupby("x", as_index=False).mean(numeric_only=True)
            if tmp.shape[0] >= 2:
                sorted_curves.append((lf, tmp))

        if not sorted_curves:
            return 0, False

        xmin = max(float(tmp["x"].min()) for _lf, tmp in sorted_curves)
        xmax = min(float(tmp["x"].max()) for _lf, tmp in sorted_curves)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            raise ValueError("The X ranges of the files do not overlap, so a grand average cannot be calculated.")

        # Use the densest file's X grid, clipped to the common overlap range.
        densest_lf, densest = max(sorted_curves, key=lambda pair: pair[1].shape[0])
        base_x = densest["x"].to_numpy(dtype=float)
        base_x = base_x[(base_x >= xmin) & (base_x <= xmax)]
        if base_x.size < 2:
            min_points = max(2, min(tmp.shape[0] for _lf, tmp in sorted_curves))
            base_x = np.linspace(xmin, xmax, min_points)

        interpolated = []
        names = []
        for lf, tmp in sorted_curves:
            x = tmp["x"].to_numpy(dtype=float)
            y = tmp["mean"].to_numpy(dtype=float)
            interp_y = np.interp(base_x, x, y)
            interpolated.append(interp_y)
            names.append(lf.stem)

        stack = np.vstack(interpolated)
        grand_mean = np.nanmean(stack, axis=0)
        grand_std = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(grand_mean)

        linewidth, _offset_step, marker = self._common_plot_options()
        display_idx = self._display_indices_for_curve(base_x, grand_mean, self._max_plot_points())
        base_x_plot = base_x[display_idx]
        grand_mean_plot = grand_mean[display_idx]
        line = self.ax.plot(base_x_plot, grand_mean_plot, linewidth=linewidth, marker=marker, markersize=3, label=f"Grand average ({len(interpolated)} files)")[0]
        if self.std_band_var.get() and not self.logy_var.get():
            grand_std_plot = grand_std[display_idx]
            self.ax.fill_between(base_x_plot, grand_mean_plot - grand_std_plot, grand_mean_plot + grand_std_plot, alpha=0.18, color=line.get_color(), linewidth=0)

        self.last_plot_payload["curves"].append({"name": "grand_average", "x": base_x, "y": grand_mean, "std": grand_std})
        self.last_plot_payload["file_names"] = names
        self.last_plot_payload["common_x_source"] = densest_lf.name
        return len(interpolated), bool(np.any(grand_mean <= 0))

    def _plot_temperature_waterfall_average(self) -> Tuple[int, bool]:
        linewidth, _offset_step, marker = self._common_plot_options()
        curve_items: List[Tuple[LoadedDataFile, np.ndarray, np.ndarray, np.ndarray]] = []

        for lf in self._files_for_plot(force_temperature_sort=True):
            curve = self._average_curve_for_file(lf)
            if curve is None:
                continue
            x = curve["x"].to_numpy(dtype=float)
            y = curve["mean"].to_numpy(dtype=float)
            std = curve["std"].to_numpy(dtype=float) if "std" in curve else np.full_like(y, np.nan)
            curve_items.append((lf, x, y, std))

        if not curve_items:
            return 0, False

        if self.auto_waterfall_offset_var.get():
            offset_step = self._auto_waterfall_offset([item[2] for item in curve_items])
        else:
            offset_step = self._manual_offset_value()

        files = [item[0] for item in curve_items]
        cmap, norm = self._temperature_color_context(files, force=True)
        any_nonpositive = False
        plotted = 0

        for idx, (lf, x, y, std) in enumerate(curve_items):
            y_plot = y + idx * offset_step
            if np.any(y_plot <= 0):
                any_nonpositive = True

            temp_label = lf.temperature_label or "UNKNOWN"
            label = f"{temp_label}  {lf.stem}"
            plot_kwargs = {"linewidth": linewidth, "marker": marker, "markersize": 3, "label": label}
            file_color = self._color_for_file(lf, cmap, norm)
            if file_color is not None:
                plot_kwargs["color"] = file_color
            display_idx = self._display_indices_for_curve(x, y_plot, self._max_plot_points())
            x_display = x[display_idx]
            y_display = y_plot[display_idx]
            line = self.ax.plot(x_display, y_display, **plot_kwargs)[0]

            if self.std_band_var.get() and np.isfinite(std).any() and not self.logy_var.get():
                color = line.get_color()
                std_display = std[display_idx]
                self.ax.fill_between(x_display, y_display - std_display, y_display + std_display, alpha=0.14, color=color, linewidth=0)

            self.last_plot_payload["curves"].append(
                {
                    "name": label,
                    "x": x,
                    "y": y_plot,
                    "std": std,
                    "temperature": lf.temperature,
                    "waterfall_offset": idx * offset_step,
                }
            )
            plotted += 1

        self.last_plot_payload["suppress_legend"] = True
        self.last_plot_payload["waterfall_offset_step"] = offset_step
        self._add_temperature_colorbar(cmap, norm)
        return plotted, any_nonpositive

    def _finish_axes(self, plotted: int, any_nonpositive: bool) -> None:
        lf = self._active_file()
        xlabel = self.xlabel_var.get().strip() or (lf.x_col if lf is not None and lf.x_col else "X")
        ylabel = self.ylabel_var.get().strip() or "Y"
        title = self.title_var.get().strip()

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self._apply_grid_setting()

        if self.logy_var.get():
            if any_nonpositive:
                self.status_var.set("Plot done, but log Y was not applied because at least one plotted value is ≤ 0.")
            else:
                self.ax.set_yscale("log")

        legend_limit = 10 if self.last_plot_payload.get("suppress_legend") else 30
        if self.legend_var.get() and plotted <= legend_limit:
            self.ax.legend(loc="best", fontsize=8)
        elif self.legend_var.get() and plotted > legend_limit:
            # Keep the plot readable instead of making a legend wall.
            pass

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _single_y_curve(self, lf: LoadedDataFile, y_col: str) -> Optional[pd.DataFrame]:
        if lf.x_col is None or lf.x_col not in lf.df.columns or y_col not in lf.df.columns:
            return None
        x = pd.to_numeric(lf.df[lf.x_col], errors="coerce")
        y = pd.to_numeric(lf.df[y_col], errors="coerce")
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if tmp.shape[0] < 2:
            return None
        if self.sort_x_var.get():
            tmp = tmp.sort_values("x")
        tmp["y"] = self._apply_normalization(tmp["y"].to_numpy(dtype=float))
        return tmp

    def _average_curve_for_file(self, lf: LoadedDataFile) -> Optional[pd.DataFrame]:
        if lf.x_col is None or lf.x_col not in lf.df.columns or not lf.y_cols:
            return None
        use_cols = [c for c in lf.y_cols if c in lf.df.columns]
        if not use_cols:
            return None

        data = pd.DataFrame({"x": pd.to_numeric(lf.df[lf.x_col], errors="coerce")})
        for col in use_cols:
            data[col] = pd.to_numeric(lf.df[col], errors="coerce")
        data = data.dropna(subset=["x"])
        if data.shape[0] < 2:
            return None

        ymat = data[use_cols]
        data["mean"] = ymat.mean(axis=1, skipna=True)
        data["std"] = ymat.std(axis=1, skipna=True)
        data["n"] = ymat.notna().sum(axis=1)
        data = data[data["n"] > 0][["x", "mean", "std", "n"]].dropna(subset=["mean"])
        if data.shape[0] < 2:
            return None

        # If X has duplicated values, combine them so interpolation/plotting is clean.
        grouped = data.groupby("x", as_index=False).agg({"mean": "mean", "std": "mean", "n": "sum"})
        if self.sort_x_var.get():
            grouped = grouped.sort_values("x")

        y = grouped["mean"].to_numpy(dtype=float)
        std = grouped["std"].to_numpy(dtype=float)
        y_norm, std_norm = self._apply_normalization_with_std(y, std)
        grouped["mean"] = y_norm
        grouped["std"] = std_norm
        return grouped

    def _apply_normalization(self, y: np.ndarray) -> np.ndarray:
        y_norm, _std = self._apply_normalization_with_std(y, None)
        return y_norm

    def _apply_normalization_with_std(self, y: np.ndarray, std: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        mode = self.normalize_var.get()
        y = y.astype(float)
        std_out = np.full_like(y, np.nan) if std is None else std.astype(float)

        if mode == "None":
            return y, std_out
        if mode == "Max = 1":
            max_abs = np.nanmax(np.abs(y)) if len(y) else np.nan
            if not np.isfinite(max_abs) or max_abs == 0:
                return y, std_out
            return y / max_abs, std_out / max_abs
        if mode == "Min-max":
            ymin = np.nanmin(y)
            ymax = np.nanmax(y)
            denom = ymax - ymin
            if not np.isfinite(ymin) or not np.isfinite(ymax) or denom == 0:
                return y, std_out
            return (y - ymin) / denom, std_out / denom
        if mode == "Z-score":
            mean = np.nanmean(y)
            sd = np.nanstd(y)
            if not np.isfinite(sd) or sd == 0:
                return y, std_out
            return (y - mean) / sd, std_out / sd
        return y, std_out

    def clear_plot(self) -> None:
        self._reset_main_plot_axes()
        self.canvas.draw_idle()
        self.last_plot_payload = {}

    # ------------------------------------------------------------------
    # Full-screen spike editor
    # ------------------------------------------------------------------
    def open_spike_editor(self) -> None:
        if not self.files:
            messagebox.showinfo("No data", "Open one or more data files first.")
            return

        if self.spike_window is not None:
            try:
                if self.spike_window.winfo_exists():
                    self.spike_window.lift()
                    self.spike_window.focus_force()
                    self._refresh_spike_file_choices()
                    self._refresh_spike_plot()
                    return
            except Exception:
                pass

        self.spike_view_mode_var.set("Overlay")
        self.spike_show_all_var.set(True)
        self.spike_same_y_only_var.set(False)
        self.spike_stack_position_var.set(0.0)
        self.max_spike_context_var.set("1200")

        win = tb.Toplevel(self)
        self.spike_window = win
        win.title("Full-screen spike editor")
        win.geometry("1500x900")
        win.minsize(1100, 700)
        win.protocol("WM_DELETE_WINDOW", self._close_spike_editor)

        try:
            win.state("zoomed")
        except Exception:
            try:
                win.geometry(f"{win.winfo_screenwidth()}x{win.winfo_screenheight()}+0+0")
            except Exception:
                pass

        top = tb.Frame(win)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))

        tb.Label(top, text="File").pack(side=tk.LEFT, padx=(0, 5))
        self.spike_file_combo = tb.Combobox(top, textvariable=self.spike_file_var, values=[], state="readonly", width=64)
        self.spike_file_combo.pack(side=tk.LEFT, padx=(0, 12), fill=tk.X, expand=True)
        self.spike_file_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_spike_file_changed())

        tb.Label(top, text="Y spectrum").pack(side=tk.LEFT, padx=(0, 5))
        self.spike_y_combo = tb.Combobox(top, textvariable=self.spike_y_var, values=[], state="readonly", width=34, height=25)
        self.spike_y_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.spike_y_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_spike_y_changed())

        tb.Label(top, text="View").pack(side=tk.LEFT, padx=(0, 5))
        mode_combo = tb.Combobox(top, textvariable=self.spike_view_mode_var, values=["Overlay", "Stacked"], state="readonly", width=10)
        mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_spike_view_mode_changed())
        tb.Button(top, text="Stack spectra", bootstyle="warning", command=self._enable_spike_stacked_view).pack(side=tk.LEFT, padx=(0, 8))
        tb.Button(top, text="Overlay", bootstyle="secondary", command=self._enable_spike_overlay_view).pack(side=tk.LEFT, padx=(0, 12))

        tb.Label(top, textvariable=self.spike_count_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        controls = tb.Frame(win)
        controls.pack(fill=tk.X, padx=10, pady=(0, 6))

        tb.Checkbutton(controls, text="All files + Y columns", variable=self.spike_show_all_var, command=self._refresh_spike_plot).pack(side=tk.LEFT, padx=(0, 12))
        tb.Checkbutton(controls, text="Same column only", variable=self.spike_same_y_only_var, command=self._on_spike_same_y_only_changed).pack(side=tk.LEFT, padx=(0, 12))
        tb.Label(controls, text="Context").pack(side=tk.LEFT, padx=(0, 4))
        tb.Entry(controls, textvariable=self.max_spike_context_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        tb.Label(controls, text="Pts").pack(side=tk.LEFT, padx=(0, 4))
        tb.Entry(controls, textvariable=self.max_spike_display_points_var, width=6).pack(side=tk.LEFT, padx=(0, 14))
        tb.Label(controls, text="Visible").pack(side=tk.LEFT, padx=(0, 4))
        tb.Entry(controls, textvariable=self.spike_stack_visible_var, width=5).pack(side=tk.LEFT, padx=(0, 10))
        tb.Label(controls, text="Stack offset").pack(side=tk.LEFT, padx=(0, 4))
        tb.Entry(controls, textvariable=self.spike_stack_offset_var, width=7).pack(side=tk.LEFT, padx=(0, 12))
        tb.Label(controls, text="Stack scroll").pack(side=tk.LEFT, padx=(0, 4))
        self.spike_stack_scale = tk.Scale(
            controls,
            variable=self.spike_stack_position_var,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            resolution=1,
            command=self._on_spike_stack_scroll,
        )
        self.spike_stack_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tb.Button(controls, text="Refresh", bootstyle="secondary", command=self._refresh_spike_plot_from_controls).pack(side=tk.LEFT, padx=(8, 0))

        # Always-visible action bar. On some PCs with high display scaling or a
        # smaller monitor, the right-side button panel can be pushed off-screen.
        # These duplicate the critical actions at the top so saving corrected TXT
        # is never hidden. The Matplotlib floppy-disk icon only saves the figure.
        action_bar = tb.LabelFrame(win, text="Spike actions — corrected TXT saving is here")
        action_bar.pack(fill=tk.X, padx=10, pady=(0, 6))
        action_inner = tb.Frame(action_bar)
        action_inner.pack(fill=tk.X, padx=8, pady=6)
        action_buttons = [
            ("Apply spikes", "success", self.apply_selected_spikes),
            ("Apply + next", "success-outline", self.apply_spikes_and_next_spectrum),
            ("Undo last", "secondary", self.undo_last_spike_action),
            ("Clear selected", "secondary", self.clear_selected_spikes),
            ("Save corrected ACTIVE TXT", "primary", self.save_corrected_active_txt),
            ("Save corrected ALL TXT", "primary", self.save_corrected_all_txt),
            ("Close", "secondary", self._close_spike_editor),
        ]
        for col, (label, style, command) in enumerate(action_buttons):
            action_inner.columnconfigure(col, weight=1)
            tb.Button(action_inner, text=label, bootstyle=style, command=command).grid(
                row=0,
                column=col,
                sticky="ew",
                padx=3,
                pady=2,
            )
        tb.Label(
            action_inner,
            text="Tip: the dark Save corrected TXT buttons write corrected data. The Matplotlib toolbar saves only the image.",
            anchor=tk.W,
        ).grid(row=1, column=0, columnspan=len(action_buttons), sticky="ew", padx=3, pady=(2, 0))

        # Always-visible Matplotlib toolbar. On some PCs / high-DPI displays, the
        # toolbar packed below the canvas can be pushed outside the visible area.
        # Therefore the real NavigationToolbar2Tk is mounted here, above the plot.
        matplotlib_toolbar_box = tb.LabelFrame(win, text="Matplotlib navigation toolbar — home / back / pan / zoom / save figure")
        matplotlib_toolbar_box.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.spike_toolbar_host = tb.Frame(matplotlib_toolbar_box)
        self.spike_toolbar_host.pack(fill=tk.X, padx=8, pady=4)
        tb.Label(
            self.spike_toolbar_host,
            text="Toolbar loading after plot canvas is created...",
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        tb.Label(win, textvariable=self.spike_visible_count_var, anchor=tk.W, font=("Segoe UI", 9, "bold")).pack(fill=tk.X, padx=10, pady=(0, 2))

        body = tb.Frame(win)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        plot_frame = tb.Frame(body)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = tb.Frame(body, width=270)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        self.spike_figure = Figure(figsize=(12.6, 7.6), dpi=100)
        self.spike_ax = self.spike_figure.add_subplot(111)
        plot_body = tb.Frame(plot_frame)
        plot_body.pack(fill=tk.BOTH, expand=True)

        stack_scroll_col = tb.Frame(plot_body, width=58)
        stack_scroll_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        stack_scroll_col.pack_propagate(False)
        tb.Label(stack_scroll_col, text="Stack\nscroll", justify=tk.CENTER).pack(fill=tk.X, pady=(0, 4))
        tb.Button(stack_scroll_col, text="Up", bootstyle="secondary", command=lambda: self._scroll_spike_stack_by(-1)).pack(fill=tk.X, pady=(0, 4))
        self.spike_stack_vscale = tk.Scale(
            stack_scroll_col,
            variable=self.spike_stack_position_var,
            from_=0,
            to=0,
            orient=tk.VERTICAL,
            showvalue=True,
            resolution=1,
            command=self._on_spike_stack_scroll,
        )
        self.spike_stack_vscale.pack(fill=tk.Y, expand=True)
        tb.Button(stack_scroll_col, text="Down", bootstyle="secondary", command=lambda: self._scroll_spike_stack_by(1)).pack(fill=tk.X, pady=(4, 0))

        self.spike_canvas = FigureCanvasTkAgg(self.spike_figure, master=plot_body)
        canvas_widget = self.spike_canvas.get_tk_widget()
        canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas_widget.bind("<MouseWheel>", self._on_spike_mousewheel)
        canvas_widget.bind("<Button-4>", self._on_spike_mousewheel)
        canvas_widget.bind("<Button-5>", self._on_spike_mousewheel)
        self.spike_canvas.mpl_connect("button_press_event", self._on_spike_canvas_click)
        self._make_spike_rectangle_selector()

        # Mount the Matplotlib toolbar in the always-visible top toolbar host.
        # This prevents the Home/Back/Pan/Zoom/Subplots/Save icons from being
        # hidden below the canvas on different screen sizes or Windows scaling.
        toolbar_parent = self.spike_toolbar_host
        if toolbar_parent is None:
            toolbar_parent = tb.Frame(plot_frame)
            toolbar_parent.pack(fill=tk.X)
        try:
            for child in toolbar_parent.winfo_children():
                child.destroy()
        except Exception:
            pass
        self.spike_toolbar = NavigationToolbar2Tk(self.spike_canvas, toolbar_parent, pack_toolbar=False)
        self.spike_toolbar.update()
        self.spike_toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tb.Label(side, text="Selected spike indices", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.spike_listbox = tk.Listbox(side, height=12, activestyle="dotbox")
        self.spike_listbox.pack(fill=tk.BOTH, expand=False)

        tb.Label(
            side,
            text="Drag a box to select every visible point inside it. Click a point for single-point selection. Right-click a red X to unmark.",
            wraplength=250,
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(10, 10))

        tb.Button(side, text="Apply spikes", bootstyle="success", command=self.apply_selected_spikes).pack(fill=tk.X, pady=3)
        tb.Button(side, text="Apply + next spectrum", bootstyle="success-outline", command=self.apply_spikes_and_next_spectrum).pack(fill=tk.X, pady=3)
        nav_row = tb.Frame(side)
        nav_row.pack(fill=tk.X, pady=3)
        tb.Button(nav_row, text="Prev", bootstyle="secondary", command=lambda: self._move_spike_spectrum(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tb.Button(nav_row, text="Next", bootstyle="secondary", command=lambda: self._move_spike_spectrum(1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        tb.Button(side, text="Undo last", bootstyle="secondary", command=self.undo_last_spike_action).pack(fill=tk.X, pady=3)
        tb.Button(side, text="Clear selected spikes", bootstyle="secondary", command=self.clear_selected_spikes).pack(fill=tk.X, pady=3)
        tb.Button(side, text="Save corrected active TXT", bootstyle="primary", command=self.save_corrected_active_txt).pack(fill=tk.X, pady=(12, 3))
        tb.Button(side, text="Save corrected ALL TXT", bootstyle="primary", command=self.save_corrected_all_txt).pack(fill=tk.X, pady=3)
        tb.Button(side, text="Close", bootstyle="secondary", command=self._close_spike_editor).pack(fill=tk.X, pady=(12, 3))

        status = tb.Label(win, textvariable=self.spike_status_var, anchor=tk.W)
        status.pack(fill=tk.X, padx=10, pady=(0, 8))

        # Keyboard shortcuts for the spike editor. These are helpful on PCs where
        # scaling hides part of the side panel.
        win.bind("<Control-s>", lambda _e: self.save_corrected_active_txt())
        win.bind("<Control-S>", lambda _e: self.save_corrected_all_txt())
        win.bind("<Return>", lambda _e: self.apply_selected_spikes())
        win.bind("<Control-Return>", lambda _e: self.apply_spikes_and_next_spectrum())
        win.bind("<Delete>", lambda _e: self.clear_selected_spikes())
        win.bind("<Escape>", lambda _e: self._close_spike_editor())

        self._refresh_spike_file_choices()
        self._refresh_spike_plot()

    def _make_spike_rectangle_selector(self) -> None:
        if self.spike_ax is None:
            return
        selector_kwargs = dict(
            useblit=True,
            button=[1],
            minspanx=0.0,
            minspany=0.0,
            spancoords="data",
            interactive=False,
        )
        props = dict(facecolor="#d7191c", edgecolor="#d7191c", alpha=0.12, fill=True)
        try:
            self.spike_rect_selector = RectangleSelector(
                self.spike_ax,
                self._on_spike_box_select,
                props=props,
                **selector_kwargs,
            )
        except TypeError:
            self.spike_rect_selector = RectangleSelector(
                self.spike_ax,
                self._on_spike_box_select,
                rectprops=props,
                **selector_kwargs,
            )
        try:
            self.spike_rect_selector.set_active(True)
        except Exception:
            pass

    def _on_spike_view_mode_changed(self) -> None:
        if self._spike_is_stacked():
            self.spike_same_y_only_var.set(False)
        self.spike_stack_position_var.set(0.0)
        self._refresh_spike_plot()

    def _enable_spike_stacked_view(self) -> None:
        self.spike_view_mode_var.set("Stacked")
        self.spike_same_y_only_var.set(False)
        self._refresh_spike_plot()

    def _enable_spike_overlay_view(self) -> None:
        self.spike_view_mode_var.set("Overlay")
        self._refresh_spike_plot()

    def _on_spike_same_y_only_changed(self) -> None:
        if self.spike_same_y_only_var.get():
            self.spike_view_mode_var.set("Overlay")
            self.spike_stack_position_var.set(0.0)
            self.spike_status_var.set("Same-Y overlay enabled.")
        self._refresh_spike_plot()

    def _refresh_spike_plot_from_controls(self) -> None:
        offset_text = self.spike_stack_offset_var.get().strip()
        if (offset_text and offset_text.lower() != "auto") or float(self.spike_stack_position_var.get()) > 0:
            self.spike_view_mode_var.set("Stacked")
            self.spike_same_y_only_var.set(False)
        self._refresh_spike_plot()

    def _on_spike_stack_scroll(self, _value: object = None) -> None:
        if not self._spike_is_stacked():
            self.spike_view_mode_var.set("Stacked")
            self.spike_status_var.set("Stacked view enabled.")
        self.spike_same_y_only_var.set(False)
        self._refresh_spike_plot()

    def _stack_scroll_max(self) -> int:
        for scale in (self.spike_stack_vscale, self.spike_stack_scale):
            if scale is None:
                continue
            try:
                return int(round(float(scale.cget("to"))))
            except Exception:
                pass
        return 0

    def _scroll_spike_stack_by(self, delta: int) -> None:
        if not self._spike_is_stacked():
            self.spike_view_mode_var.set("Stacked")
        max_scroll = self._stack_scroll_max()
        current = int(round(float(self.spike_stack_position_var.get())))
        next_value = min(max(current + int(delta), 0), max_scroll)
        self.spike_stack_position_var.set(float(next_value))
        self._refresh_spike_plot()

    def _on_spike_mousewheel(self, event: object) -> None:
        if self.spike_toolbar is not None and getattr(self.spike_toolbar, "mode", ""):
            return
        if not self._spike_is_stacked():
            return
        visible = self._spike_stack_visible_count()
        step = max(1, visible // 3)
        num = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if num == 4 or delta > 0:
            self._scroll_spike_stack_by(-step)
        elif num == 5 or delta < 0:
            self._scroll_spike_stack_by(step)

    def _spike_stack_offset_step(self, curves: Sequence[Dict[str, object]]) -> float:
        text = self.spike_stack_offset_var.get().strip()
        if text and text.lower() != "auto":
            try:
                value = float(text)
                if np.isfinite(value) and value > 0:
                    return value
            except Exception:
                pass

        values = []
        for curve_info in curves[: min(len(curves), 80)]:
            arr = np.asarray(curve_info.get("y", []), dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                values.append(finite)
        if not values:
            return 1.0
        joined = np.concatenate(values)
        if joined.size == 0:
            return 1.0
        p5, p95 = np.nanpercentile(joined, [5, 95])
        robust_range = float(p95 - p5)
        if not np.isfinite(robust_range) or robust_range <= 0:
            robust_range = float(np.nanmax(joined) - np.nanmin(joined))
        if not np.isfinite(robust_range) or robust_range <= 0:
            robust_range = 1.0
        return robust_range * 1.18

    def _apply_spike_display_offsets(self, curves: Sequence[Dict[str, object]]) -> float:
        if not curves:
            return 0.0
        offset_step = self._spike_stack_offset_step(curves) if self._spike_is_stacked() else 0.0
        for stack_index, curve_info in enumerate(curves):
            offset = stack_index * offset_step
            y = np.asarray(curve_info["y"], dtype=float)
            draw_y = np.asarray(curve_info["draw_y"], dtype=float)
            curve_info["stack_index"] = stack_index
            curve_info["stack_offset"] = offset
            curve_info["display_y"] = y + offset
            curve_info["draw_display_y"] = draw_y + offset
        return offset_step

    def _update_spike_stack_scale(self, total_curves: int) -> int:
        visible = self._spike_stack_visible_count()
        max_scroll = max(0, total_curves - visible)
        current = int(round(float(self.spike_stack_position_var.get())))
        if current > max_scroll:
            self.spike_stack_position_var.set(float(max_scroll))
            current = max_scroll
        if current < 0:
            self.spike_stack_position_var.set(0.0)
            current = 0
        if self.spike_stack_scale is not None:
            try:
                self.spike_stack_scale.configure(to=max_scroll, state=tk.NORMAL if max_scroll > 0 else tk.DISABLED)
            except Exception:
                pass
        if self.spike_stack_vscale is not None:
            try:
                self.spike_stack_vscale.configure(to=max_scroll, state=tk.NORMAL if max_scroll > 0 else tk.DISABLED)
            except Exception:
                pass
        return current

    def _set_stacked_spike_ylim(self, curves: Sequence[Dict[str, object]]) -> None:
        if not curves or self.spike_ax is None:
            return
        start = self._update_spike_stack_scale(len(curves))
        visible = self._spike_stack_visible_count()
        stop = min(len(curves), start + visible)
        visible_curves = [c for c in curves if start <= int(c.get("stack_index", 0)) < stop]
        if not visible_curves:
            visible_curves = list(curves)
        vals = []
        for curve_info in visible_curves:
            y = np.asarray(curve_info.get("display_y", []), dtype=float)
            finite = y[np.isfinite(y)]
            if finite.size:
                vals.append(finite)
        if not vals:
            return
        joined = np.concatenate(vals)
        ymin = float(np.nanmin(joined))
        ymax = float(np.nanmax(joined))
        pad = (ymax - ymin) * 0.06
        if not np.isfinite(pad) or pad <= 0:
            pad = 1.0
        self.spike_ax.set_ylim(ymin - pad, ymax + pad)

    def _close_spike_editor(self) -> None:
        if self.spike_window is not None:
            try:
                self.spike_window.destroy()
            except Exception:
                pass
        self.spike_window = None
        self.spike_canvas = None
        self.spike_toolbar = None
        self.spike_toolbar_host = None
        self.spike_figure = None
        self.spike_ax = None
        self.spike_listbox = None
        self.spike_rect_selector = None
        self.spike_stack_scale = None
        self.spike_stack_vscale = None
        self.spike_visible_curves = []
        self.spike_selected_positions.clear()
        self.spike_selection_order.clear()

    def _refresh_spike_file_choices(self) -> None:
        if self.spike_file_combo is None:
            return
        self.spike_file_label_to_index.clear()
        labels = []
        for idx, lf in enumerate(self.files):
            temp = lf.temperature_label or "UNKNOWN"
            label = f"{idx + 1}. {lf.name} [{temp}]"
            labels.append(label)
            self.spike_file_label_to_index[label] = idx
        self.spike_file_combo.configure(values=labels)
        if not labels:
            self.spike_file_var.set("")
            return
        current = self.spike_file_var.get()
        if current not in self.spike_file_label_to_index:
            active = self.active_index if self.active_index is not None else 0
            active = min(max(active, 0), len(labels) - 1)
            self.spike_file_var.set(labels[active])
        self._refresh_spike_y_choices()

    def _refresh_spike_y_choices(self) -> None:
        if self.spike_y_combo is None:
            return
        lf = self._spike_active_file()
        if lf is None:
            self.spike_y_combo.configure(values=[])
            self.spike_y_var.set("")
            return
        values = self._all_spike_y_columns_for_file(lf, include_numeric_extras=True)
        self.spike_y_combo.configure(values=values)
        if values and self.spike_y_var.get() not in values:
            self.spike_y_var.set(values[0])
        elif not values:
            self.spike_y_var.set("")

    def _spike_active_file_index(self) -> Optional[int]:
        label = self.spike_file_var.get()
        idx = self.spike_file_label_to_index.get(label)
        if idx is None or not (0 <= idx < len(self.files)):
            return None
        return idx

    def _spike_active_file(self) -> Optional[LoadedDataFile]:
        idx = self._spike_active_file_index()
        if idx is None:
            return None
        return self.files[idx]

    def _on_spike_file_changed(self) -> None:
        idx = self._spike_active_file_index()
        if idx is not None:
            self.active_index = idx
            self._refresh_all_views()
            self._update_title_and_labels()
        self._refresh_spike_y_choices()
        self._refresh_spike_plot()

    def _on_spike_y_changed(self) -> None:
        self._refresh_spike_plot()

    def _set_spike_active_selection(self, file_index: int, y_col: str, refresh_plot: bool = False) -> None:
        if not (0 <= file_index < len(self.files)):
            return
        target_label = None
        for label, idx in self.spike_file_label_to_index.items():
            if idx == file_index:
                target_label = label
                break
        if target_label is not None and self.spike_file_var.get() != target_label:
            self.spike_file_var.set(target_label)
        self.active_index = file_index
        self._refresh_spike_y_choices()
        values = list(self.spike_y_combo.cget("values")) if self.spike_y_combo is not None else []
        if y_col in values:
            self.spike_y_var.set(y_col)
        self._refresh_all_views()
        self._update_title_and_labels()
        if refresh_plot:
            self._refresh_spike_plot()

    def _spike_curve_for(self, lf: LoadedDataFile, y_col: str) -> Optional[pd.DataFrame]:
        if lf is None or not lf.x_col or lf.x_col not in lf.df.columns or y_col not in lf.df.columns:
            return None
        data = pd.DataFrame(
            {
                "position": np.arange(len(lf.df), dtype=int),
                "x": pd.to_numeric(lf.df[lf.x_col], errors="coerce"),
                "y": pd.to_numeric(lf.df[y_col], errors="coerce"),
            }
        ).dropna(subset=["x", "y"])
        if data.shape[0] < 2:
            return None
        if self.sort_x_var.get():
            data = data.sort_values("x")
        return data

    def _spike_curve_for_current_selection(self) -> Optional[pd.DataFrame]:
        lf = self._spike_active_file()
        y_col = self.spike_y_var.get()
        if lf is None:
            return None
        return self._spike_curve_for(lf, y_col)

    def _all_numeric_y_columns_for_file(self, lf: LoadedDataFile) -> List[str]:
        return [c for c in lf.numeric_cols if c in lf.df.columns and c != lf.x_col]

    def _all_spike_y_columns_for_file(self, lf: LoadedDataFile, include_numeric_extras: bool = False) -> List[str]:
        numeric_y = self._all_numeric_y_columns_for_file(lf)
        values = [c for c in lf.y_cols if c in numeric_y]
        if include_numeric_extras or not values:
            values.extend(c for c in numeric_y if c not in values)
        return values

    def _spike_y_columns_for_file(
        self,
        lf: LoadedDataFile,
        active_y_col: str = "",
        active_lf: Optional[LoadedDataFile] = None,
    ) -> List[str]:
        if self.spike_show_all_var.get() and self.spike_same_y_only_var.get() and active_y_col:
            if active_y_col in lf.df.columns and active_y_col != lf.x_col:
                return [active_y_col]
            if active_lf is not None and active_y_col in active_lf.df.columns:
                active_cols = [str(c) for c in active_lf.df.columns]
                target_cols = [str(c) for c in lf.df.columns]
                try:
                    active_col_idx = active_cols.index(active_y_col)
                except ValueError:
                    active_col_idx = -1
                if 0 <= active_col_idx < len(target_cols):
                    candidate = target_cols[active_col_idx]
                    if candidate in lf.df.columns and candidate != lf.x_col and pd.api.types.is_numeric_dtype(lf.df[candidate]):
                        return [candidate]

        values = self._all_spike_y_columns_for_file(lf, include_numeric_extras=True)
        if active_y_col and active_y_col in lf.df.columns and active_y_col != lf.x_col and active_y_col not in values:
            values.insert(0, active_y_col)
        return values

    def _display_indices_for_spike_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        view_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> np.ndarray:
        max_points = self._max_spike_display_points()
        if view_limits is not None:
            xlim = view_limits[0]
            xmin, xmax = sorted((float(xlim[0]), float(xlim[1])))
            mask = (x >= xmin) & (x <= xmax)
            candidate_idx = np.flatnonzero(mask)
            if candidate_idx.size >= 2:
                local = self._display_indices_for_curve(x[candidate_idx], y[candidate_idx], max_points)
                return candidate_idx[local]
        return self._display_indices_for_curve(x, y, max_points)

    def _build_spike_visible_curves(
        self,
        view_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> List[Dict[str, object]]:
        curves: List[Dict[str, object]] = []
        active_idx = self._spike_active_file_index()
        active_y = self.spike_y_var.get()
        active_lf = self.files[active_idx] if active_idx is not None and 0 <= active_idx < len(self.files) else None
        max_context = self._max_spike_context_curves()
        context_added = 0
        self.spike_context_hidden_count = 0
        if self.spike_show_all_var.get():
            ordered_files = self._files_for_plot()
            file_indices = []
            for ordered_lf in ordered_files:
                for idx, lf in enumerate(self.files):
                    if lf is ordered_lf:
                        file_indices.append(idx)
                        break
        else:
            file_indices = [active_idx] if active_idx is not None else []

        for file_idx in file_indices:
            if file_idx is None or not (0 <= file_idx < len(self.files)):
                continue
            lf = self.files[file_idx]
            if not lf.x_col or lf.x_col not in lf.df.columns:
                continue
            y_cols = self._spike_y_columns_for_file(lf, active_y if file_idx == active_idx or self.spike_same_y_only_var.get() else "", active_lf)
            if not self.spike_show_all_var.get():
                y_cols = [active_y] if active_y in y_cols else y_cols[:1]
            for y_col in y_cols:
                curve = self._spike_curve_for(lf, y_col)
                if curve is None:
                    continue
                is_active = file_idx == active_idx and y_col == active_y
                if self.spike_show_all_var.get() and not is_active and context_added >= max_context:
                    self.spike_context_hidden_count += 1
                    continue
                x = curve["x"].to_numpy(dtype=float)
                y = curve["y"].to_numpy(dtype=float)
                positions = curve["position"].to_numpy(dtype=int)
                display_idx = self._display_indices_for_spike_curve(x, y, view_limits=view_limits)
                if not is_active:
                    context_added += 1
                curves.append(
                    {
                        "file_index": file_idx,
                        "file": lf,
                        "y_col": y_col,
                        "positions": positions,
                        "x": x,
                        "y": y,
                        "draw_positions": positions[display_idx],
                        "draw_x": x[display_idx],
                        "draw_y": y[display_idx],
                        "is_active": is_active,
                    }
                )
        return curves

    def _refresh_spike_plot(self, preserve_view: bool = False) -> None:
        if self.spike_ax is None or self.spike_canvas is None or self.spike_figure is None:
            return
        view_limits = None
        if preserve_view:
            try:
                xlim = tuple(float(v) for v in self.spike_ax.get_xlim())
                ylim = tuple(float(v) for v in self.spike_ax.get_ylim())
                if all(np.isfinite(v) for v in (*xlim, *ylim)) and xlim[0] != xlim[1] and ylim[0] != ylim[1]:
                    view_limits = (xlim, ylim)
            except Exception:
                view_limits = None
        self.spike_ax.clear()
        lf = self._spike_active_file()
        y_col = self.spike_y_var.get()
        self._prune_spike_selection()
        self.spike_visible_curves = self._build_spike_visible_curves(view_limits=view_limits)
        stack_offset_step = self._apply_spike_display_offsets(self.spike_visible_curves)
        if not self._spike_is_stacked():
            self._update_spike_stack_scale(len(self.spike_visible_curves))
        active_curve = next((curve for curve in self.spike_visible_curves if curve.get("is_active")), None)
        total_found = len(self.spike_visible_curves) + self.spike_context_hidden_count
        active_file_y_count = len(self._all_spike_y_columns_for_file(lf, include_numeric_extras=True)) if lf is not None else 0
        if self.spike_show_all_var.get():
            scope_text = "same column only" if self.spike_same_y_only_var.get() else "all selected Y columns"
        else:
            scope_text = "active spectrum only"
        hidden_text = f"; {self.spike_context_hidden_count:,} hidden by Context cap" if self.spike_context_hidden_count else ""
        self.spike_visible_count_var.set(
            f"Visible spectra: {len(self.spike_visible_curves):,}/{total_found:,} ({scope_text}; active file Y columns: {active_file_y_count:,}{hidden_text})"
        )

        if lf is None or active_curve is None or not y_col:
            self.spike_curve_positions = np.array([], dtype=int)
            self.spike_curve_x = np.array([], dtype=float)
            self.spike_curve_y = np.array([], dtype=float)
            self.spike_ax.text(0.5, 0.5, "Choose a file and numeric Y spectrum.", transform=self.spike_ax.transAxes, ha="center", va="center")
            self.spike_ax.grid(True, alpha=0.3)
            if view_limits is not None:
                self.spike_ax.set_xlim(view_limits[0])
                self.spike_ax.set_ylim(view_limits[1])
            self.spike_canvas.draw_idle()
            self._update_spike_selection_list()
            return

        self.spike_curve_positions = np.asarray(active_curve["positions"], dtype=int)
        self.spike_curve_x = np.asarray(active_curve["x"], dtype=float)
        self.spike_curve_y = np.asarray(active_curve["y"], dtype=float)

        cmap, norm = self._temperature_color_context(self.files, force=True)
        context_count = 0
        context_alpha = 0.70 if self._spike_is_stacked() else 0.58
        context_linewidth = 1.05 if self._spike_is_stacked() else 0.95
        for curve_info in self.spike_visible_curves:
            file_idx = int(curve_info["file_index"])
            curve_lf = curve_info["file"]
            x = np.asarray(curve_info["draw_x"], dtype=float)
            y = np.asarray(curve_info["draw_display_y"], dtype=float)
            is_active = bool(curve_info["is_active"])
            if is_active:
                continue
            color = self._color_for_file(curve_lf, cmap, norm) or "#8c8c8c"
            self.spike_ax.plot(
                x,
                y,
                color=color,
                alpha=context_alpha,
                linewidth=context_linewidth,
                zorder=1,
            )
            context_count += 1

        active_x_plot = np.asarray(active_curve["draw_x"], dtype=float)
        active_y_plot = np.asarray(active_curve["draw_display_y"], dtype=float)
        self.spike_ax.plot(active_x_plot, active_y_plot, color="#050505", linewidth=2.0, zorder=4)

        for curve_info in self.spike_visible_curves:
            file_idx = int(curve_info["file_index"])
            y_name = str(curve_info["y_col"])
            positions = np.asarray(curve_info["positions"], dtype=int)
            selected_mask = np.array([(file_idx, y_name, int(pos)) in self.spike_selected_positions for pos in positions], dtype=bool)
            if not np.any(selected_mask):
                continue
            x = np.asarray(curve_info["x"], dtype=float)
            y = np.asarray(curve_info["display_y"], dtype=float)
            self.spike_ax.scatter(
                x[selected_mask],
                y[selected_mask],
                marker="x",
                s=110,
                color="red",
                linewidths=2.3,
                zorder=6,
            )

        temp = lf.temperature_label or "UNKNOWN"
        title = f"{lf.name} | {y_col} | {temp}"
        if self.spike_show_all_var.get():
            title += f" | context: {context_count} other spectra"
            if self.spike_context_hidden_count:
                title += f" ({self.spike_context_hidden_count} hidden)"
            title += " | same column only" if self.spike_same_y_only_var.get() else " | all selected Y columns"
        if self._spike_is_stacked():
            title += f" | stacked offset {stack_offset_step:.4g}"
        self.spike_ax.set_title(title)
        self.spike_ax.set_xlabel(lf.x_col or "X")
        self.spike_ax.set_ylabel(f"{y_col} + stack offset" if self._spike_is_stacked() else y_col)
        self.spike_ax.grid(True, alpha=0.3)
        if view_limits is not None:
            self.spike_ax.set_xlim(view_limits[0])
            self.spike_ax.set_ylim(view_limits[1])
        elif self._spike_is_stacked():
            self._set_stacked_spike_ylim(self.spike_visible_curves)
        self.spike_figure.tight_layout()
        self.spike_canvas.draw_idle()
        self._update_spike_selection_list()

    def _prune_spike_selection(self) -> None:
        valid = set()
        for file_idx, y_col, pos in self.spike_selected_positions:
            if 0 <= file_idx < len(self.files):
                lf = self.files[file_idx]
                if y_col in lf.df.columns and 0 <= pos < len(lf.df):
                    valid.add((file_idx, y_col, pos))
        if len(valid) != len(self.spike_selected_positions):
            self.spike_selected_positions = valid
            self.spike_selection_order = [key for key in self.spike_selection_order if key in valid]

    def _nearest_spike_key(self, event: object, selected_only: bool = False) -> Optional[Tuple[int, str, int]]:
        if event.inaxes != self.spike_ax or not self.spike_visible_curves:
            return None
        best_key = None
        best_distance = float("inf")
        for curve_info in self.spike_visible_curves:
            file_idx = int(curve_info["file_index"])
            y_col = str(curve_info["y_col"])
            x = np.asarray(curve_info["x"], dtype=float)
            y = np.asarray(curve_info.get("display_y", curve_info["y"]), dtype=float)
            positions = np.asarray(curve_info["positions"], dtype=int)
            if selected_only:
                mask = np.array([(file_idx, y_col, int(pos)) in self.spike_selected_positions for pos in positions], dtype=bool)
                if not np.any(mask):
                    continue
                x = x[mask]
                y = y[mask]
                positions = positions[mask]
            points = self.spike_ax.transData.transform(np.column_stack([x, y]))
            dist = np.hypot(points[:, 0] - event.x, points[:, 1] - event.y)
            nearest = int(np.argmin(dist))
            if float(dist[nearest]) < best_distance:
                best_distance = float(dist[nearest])
                best_key = (file_idx, y_col, int(positions[nearest]))
        if best_distance > 13.0:
            return None
        return best_key

    def _on_spike_canvas_click(self, event: object) -> None:
        if self.spike_toolbar is not None and getattr(self.spike_toolbar, "mode", ""):
            return
        button = getattr(event, "button", None)
        if button == 1:
            key = self._nearest_spike_key(event, selected_only=False)
            if key is None:
                return
            file_idx, y_col, pos = key
            self._set_spike_active_selection(file_idx, y_col, refresh_plot=False)
            if key not in self.spike_selected_positions:
                self.spike_selected_positions.add(key)
                self.spike_selection_order.append(key)
                self.spike_status_var.set(f"Marked spike: {self._spike_key_label(key)}.")
            self._refresh_spike_plot(preserve_view=True)
        elif button == 3:
            key = self._nearest_spike_key(event, selected_only=True)
            if key is None:
                return
            self._unmark_spike_position(key)
            self.spike_status_var.set(f"Unmarked spike: {self._spike_key_label(key)}.")
            self._refresh_spike_plot(preserve_view=True)

    def _on_spike_box_select(self, eclick: object, erelease: object) -> None:
        if self.spike_toolbar is not None and getattr(self.spike_toolbar, "mode", ""):
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0, x1 = sorted((float(eclick.xdata), float(erelease.xdata)))
        y0, y1 = sorted((float(eclick.ydata), float(erelease.ydata)))
        if abs(x1 - x0) == 0 and abs(y1 - y0) == 0:
            return

        added = 0
        first_key = None
        for curve_info in self.spike_visible_curves:
            file_idx = int(curve_info["file_index"])
            y_col = str(curve_info["y_col"])
            x = np.asarray(curve_info["x"], dtype=float)
            y = np.asarray(curve_info.get("display_y", curve_info["y"]), dtype=float)
            positions = np.asarray(curve_info["positions"], dtype=int)
            mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            for pos in positions[mask]:
                key = (file_idx, y_col, int(pos))
                if key not in self.spike_selected_positions:
                    self.spike_selected_positions.add(key)
                    self.spike_selection_order.append(key)
                    added += 1
                    if first_key is None:
                        first_key = key

        if first_key is not None:
            self._set_spike_active_selection(first_key[0], first_key[1], refresh_plot=False)
        self.spike_status_var.set(f"Box selected {added} new spike point(s)." if added else "Box found no unselected visible data points.")
        self._refresh_spike_plot(preserve_view=True)

    def _unmark_spike_position(self, key: Tuple[int, str, int]) -> None:
        self.spike_selected_positions.discard(key)
        self.spike_selection_order = [item for item in self.spike_selection_order if item != key]

    def _spike_key_label(self, key: Tuple[int, str, int]) -> str:
        file_idx, y_col, pos = key
        if 0 <= file_idx < len(self.files):
            file_text = self._short(self.files[file_idx].stem, 18)
        else:
            file_text = f"file {file_idx + 1}"
        return f"{file_text} | {self._short(y_col, 18)} | row {pos}"

    def _update_spike_selection_list(self) -> None:
        self._prune_spike_selection()
        if self.spike_listbox is not None:
            self.spike_listbox.delete(0, tk.END)
            for key in sorted(self.spike_selected_positions, key=lambda item: (item[0], item[1], item[2])):
                self.spike_listbox.insert(tk.END, self._spike_key_label(key))
        self.spike_count_var.set(f"Selected spikes: {len(self.spike_selected_positions)}")

    def clear_selected_spikes(self) -> None:
        self.spike_selected_positions.clear()
        self.spike_selection_order.clear()
        self._refresh_spike_plot(preserve_view=True)
        self.spike_status_var.set("Cleared selected spike markers.")

    def undo_last_spike_action(self) -> None:
        if self.spike_selection_order:
            key = self.spike_selection_order.pop()
            self.spike_selected_positions.discard(key)
            self._refresh_spike_plot(preserve_view=True)
            self.spike_status_var.set(f"Removed last pending spike marker: {self._spike_key_label(key)}.")
            return

        if not self.spike_apply_history:
            self.spike_status_var.set("Nothing to undo.")
            return

        action = self.spike_apply_history.pop()
        groups = action.get("groups")
        if not groups:
            groups = [(int(action["file_index"]), str(action["y_col"]), action["old_values"])]

        restored = 0
        restored_by_file: Dict[int, int] = {}
        last_idx = None
        last_y_col = None
        for idx, y_col, old_values in groups:
            idx = int(idx)
            y_col = str(y_col)
            if not (0 <= idx < len(self.files)) or y_col not in self.files[idx].df.columns:
                continue
            col_idx = self.files[idx].df.columns.get_loc(y_col)
            for pos, value in old_values.items():
                if 0 <= int(pos) < len(self.files[idx].df):
                    self.files[idx].df.iat[int(pos), col_idx] = value
                    restored += 1
                    restored_by_file[idx] = restored_by_file.get(idx, 0) + 1
            last_idx = idx
            last_y_col = y_col

        for idx, count in restored_by_file.items():
            self.files[idx].spike_corrections_applied = max(0, self.files[idx].spike_corrections_applied - count)

        if last_idx is not None and last_y_col is not None:
            self._set_spike_active_selection(last_idx, last_y_col, refresh_plot=False)
        self._refresh_all_views()
        self.plot_selected(show_popup_on_error=False)
        self._refresh_spike_file_choices()
        self._refresh_spike_plot(preserve_view=True)
        self.spike_status_var.set(f"Undid last applied correction ({restored} point(s) restored).")

    def apply_selected_spikes(self) -> bool:
        self._prune_spike_selection()
        if not self.spike_selected_positions:
            self.spike_status_var.set("No spike points selected.")
            return False

        grouped: Dict[Tuple[int, str], List[int]] = {}
        for file_idx, y_col, pos in self.spike_selected_positions:
            grouped.setdefault((file_idx, y_col), []).append(pos)

        history_groups: List[Tuple[int, str, Dict[int, object]]] = []
        corrected_total = 0
        corrected_by_file: Dict[int, int] = {}
        last_group: Optional[Tuple[int, str]] = None
        for (idx, y_col), positions_raw in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            if not (0 <= idx < len(self.files)):
                continue
            lf = self.files[idx]
            if y_col not in lf.df.columns:
                continue
            positions = sorted({int(pos) for pos in positions_raw if 0 <= int(pos) < len(lf.df)})
            if not positions:
                continue

            col_idx = lf.df.columns.get_loc(y_col)
            old_values = {pos: lf.df.iat[pos, col_idx] for pos in positions}
            lf.df[y_col] = pd.to_numeric(lf.df[y_col], errors="coerce").astype(float)
            col_idx = lf.df.columns.get_loc(y_col)
            y = lf.df[y_col].to_numpy(dtype=float)
            corrected = self._impute_spike_positions(y, positions)
            for pos in positions:
                lf.df.iat[pos, col_idx] = corrected[pos]
            history_groups.append((idx, y_col, old_values))
            corrected_total += len(positions)
            corrected_by_file[idx] = corrected_by_file.get(idx, 0) + len(positions)
            last_group = (idx, y_col)

        if not history_groups:
            self.spike_status_var.set("No valid selected spike points could be applied.")
            return False

        for idx, count in corrected_by_file.items():
            self.files[idx].spike_corrections_applied += count

        self.spike_apply_history.append({"groups": history_groups})
        self.spike_selected_positions.clear()
        self.spike_selection_order.clear()
        if last_group is not None:
            self._set_spike_active_selection(last_group[0], last_group[1], refresh_plot=False)
        self._refresh_all_views()
        self.plot_selected(show_popup_on_error=False)
        self._refresh_spike_plot(preserve_view=True)
        self.spike_status_var.set(f"Applied {corrected_total} spike correction(s) across {len(history_groups)} spectrum/spectra.")
        return True

    def apply_spikes_and_next_spectrum(self) -> None:
        applied = self.apply_selected_spikes()
        if applied:
            self._select_next_spike_spectrum()

    def _move_spike_spectrum(self, step: int) -> None:
        if step == 0 or self.spike_y_combo is None or self.spike_file_combo is None:
            return
        values = list(self.spike_y_combo.cget("values"))
        current_y = self.spike_y_var.get()
        y_idx = values.index(current_y) if current_y in values else (0 if step > 0 else len(values) - 1)
        next_y_idx = y_idx + step
        if 0 <= next_y_idx < len(values):
            self.spike_y_var.set(values[next_y_idx])
            self._refresh_spike_plot()
            return

        labels = list(self.spike_file_combo.cget("values"))
        current_label = self.spike_file_var.get()
        file_label_idx = labels.index(current_label) if current_label in labels else 0
        next_file_label_idx = file_label_idx + (1 if step > 0 else -1)
        if not (0 <= next_file_label_idx < len(labels)):
            self.spike_status_var.set("No more spectra in that direction.")
            return

        self.spike_file_var.set(labels[next_file_label_idx])
        idx = self._spike_active_file_index()
        if idx is not None:
            self.active_index = idx
            self._refresh_all_views()
            self._update_title_and_labels()
        self._refresh_spike_y_choices()
        values = list(self.spike_y_combo.cget("values")) if self.spike_y_combo is not None else []
        if values:
            self.spike_y_var.set(values[0] if step > 0 else values[-1])
        self._refresh_spike_plot()

    def _select_next_spike_spectrum(self) -> None:
        self._move_spike_spectrum(1)

    @staticmethod
    def _impute_spike_positions(y: np.ndarray, positions: Sequence[int]) -> np.ndarray:
        corrected = np.asarray(y, dtype=float).copy()
        selected = sorted({int(p) for p in positions if 0 <= int(p) < corrected.size})
        selected_set = set(selected)
        i = 0
        while i < len(selected):
            start = selected[i]
            end = start
            while i + 1 < len(selected) and selected[i + 1] == end + 1:
                i += 1
                end = selected[i]

            left = start - 1
            while left >= 0 and (left in selected_set or not np.isfinite(corrected[left])):
                left -= 1
            right = end + 1
            while right < corrected.size and (right in selected_set or not np.isfinite(corrected[right])):
                right += 1

            left_ok = left >= 0 and np.isfinite(corrected[left])
            right_ok = right < corrected.size and np.isfinite(corrected[right])
            if left_ok and right_ok:
                fill = (corrected[left] + corrected[right]) / 2.0
            elif left_ok:
                fill = corrected[left]
            elif right_ok:
                fill = corrected[right]
            else:
                fill = np.nan

            if np.isfinite(fill):
                corrected[start : end + 1] = fill
            i += 1
        return corrected

    def save_corrected_active_txt(self) -> None:
        lf = self._spike_active_file() or self._active_file()
        if lf is None:
            messagebox.showinfo("No active file", "Choose an active file first.")
            return
        if not self._apply_pending_spikes_before_save():
            return
        try:
            out = self._save_corrected_file(lf)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        if out is None:
            self.spike_status_var.set("Save cancelled.")
            return
        messagebox.showinfo(
            "Saved corrected TXT",
            f"Saved the current corrected table here:\n{out}\n\n"
            f"Applied spike replacements in this file: {lf.spike_corrections_applied}",
        )
        self.spike_status_var.set(f"Saved corrected TXT: {out}")
        self.status_var.set(f"Saved corrected TXT: {out}")

    def save_corrected_all_txt(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "Open one or more files first.")
            return
        if not self._apply_pending_spikes_before_save():
            return

        base_paths = [self._corrected_base_path(lf) for lf in self.files]
        existing = [path for path in base_paths if path.exists()]
        overwrite_existing: Optional[bool] = None
        if existing:
            answer = messagebox.askyesnocancel(
                "Corrected files already exist",
                f"{len(existing)} corrected TXT filename(s) already exist.\n\n"
                "Yes = replace those files with the corrections currently in memory.\n"
                "No = keep them and save numbered copies (_2, _3, ...).\n"
                "Cancel = stop saving.\n\n"
                f"Example:\n{existing[0]}",
            )
            if answer is None:
                self.spike_status_var.set("Save cancelled.")
                return
            overwrite_existing = bool(answer)

        saved: List[Path] = []
        reserved_paths: set[Path] = set()
        try:
            for lf in self.files:
                out = self._save_corrected_file(lf, overwrite_existing=overwrite_existing, reserved_paths=reserved_paths)
                if out is not None:
                    saved.append(out)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        if not saved:
            self.spike_status_var.set("Save cancelled.")
            return

        total_replacements = sum(lf.spike_corrections_applied for lf in self.files)
        folders = sorted({str(path.parent) for path in saved})
        folder_text = "\n".join(folders[:6])
        if len(folders) > 6:
            folder_text += f"\n... and {len(folders) - 6} more"
        if existing:
            overwrite_text = (
                "Existing corrected files were replaced."
                if overwrite_existing
                else "Existing corrected files were left alone; numbered copies were used where needed."
            )
        else:
            overwrite_text = "No existing corrected files had to be replaced."
        messagebox.showinfo(
            "Saved corrected TXT files",
            f"Saved {len(saved)} corrected TXT file(s).\n"
            f"Applied spike replacements currently in memory: {total_replacements}\n\n"
            f"{overwrite_text}\n\nOutput folder(s):\n{folder_text}",
        )
        self.spike_status_var.set(f"Saved {len(saved)} corrected TXT file(s). Applied replacements in memory: {total_replacements}.")
        self.status_var.set(f"Saved {len(saved)} corrected TXT file(s) into Spikes_Removed folders.")

    def _apply_pending_spikes_before_save(self) -> bool:
        self._prune_spike_selection()
        if not self.spike_selected_positions:
            return True

        answer = messagebox.askyesnocancel(
            "Apply selected spikes before saving?",
            f"You still have {len(self.spike_selected_positions)} selected red X spike marker(s).\n\n"
            "Those markers are not written to TXT until Apply spikes runs.\n\n"
            "Yes = apply them now and save.\n"
            "No = save only corrections that were already applied.\n"
            "Cancel = stop saving.",
        )
        if answer is None:
            self.spike_status_var.set("Save cancelled.")
            return False
        if answer is False:
            return True
        return self.apply_selected_spikes()

    def _corrected_base_path(self, lf: LoadedDataFile) -> Path:
        out_dir = lf.path.parent / "Spikes_Removed"
        temp_label = lf.temperature_label or "UNKNOWN"
        parent_name = self._safe_filename_part(lf.path.parent.name or "Data")
        temp_part = self._safe_filename_part(temp_label)
        base = f"{parent_name}_TEMP_{temp_part}_Spikes_Removed.txt"
        return out_dir / base

    def _save_corrected_file(
        self,
        lf: LoadedDataFile,
        overwrite_existing: Optional[bool] = None,
        reserved_paths: Optional[set[Path]] = None,
    ) -> Optional[Path]:
        out_path = self._corrected_base_path(lf)
        out_path.parent.mkdir(exist_ok=True)
        out_path = self._resolve_corrected_output_path(out_path, overwrite_existing, reserved_paths=reserved_paths)
        if out_path is None:
            return None
        lf.df.to_csv(out_path, sep="\t", index=False)
        lf.last_corrected_save_path = out_path
        if reserved_paths is not None:
            reserved_paths.add(out_path)
        return out_path

    def _resolve_corrected_output_path(
        self,
        path: Path,
        overwrite_existing: Optional[bool],
        reserved_paths: Optional[set[Path]] = None,
    ) -> Optional[Path]:
        if reserved_paths is not None and path in reserved_paths:
            return self._unique_output_path(path, reserved_paths=reserved_paths)
        if not path.exists():
            return path
        if overwrite_existing is True:
            return path
        if overwrite_existing is False:
            return self._unique_output_path(path, reserved_paths=reserved_paths)

        answer = messagebox.askyesnocancel(
            "Corrected file already exists",
            f"This corrected TXT already exists:\n{path}\n\n"
            "Yes = replace it with the corrections currently in memory.\n"
            "No = keep it and save a numbered copy (_2, _3, ...).\n"
            "Cancel = stop saving.",
        )
        if answer is None:
            return None
        if answer:
            return path
        return self._unique_output_path(path, reserved_paths=reserved_paths)

    @staticmethod
    def _safe_filename_part(text: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(text).strip())
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned.strip("._ ") or "UNKNOWN"

    @staticmethod
    def _unique_output_path(path: Path, reserved_paths: Optional[set[Path]] = None) -> Path:
        reserved = reserved_paths or set()
        if not path.exists() and path not in reserved:
            return path
        stem = path.stem
        suffix = path.suffix
        for idx in range(2, 10000):
            candidate = path.with_name(f"{stem}_{idx}{suffix}")
            if not candidate.exists() and candidate not in reserved:
                return candidate
        raise FileExistsError(f"Could not create a unique output name for {path}")

    # ------------------------------------------------------------------
    # Export/save
    # ------------------------------------------------------------------
    def save_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            self.status_var.set(f"Saved plot: {path}")
        except Exception as exc:
            messagebox.showerror("Could not save plot", str(exc))

    def export_plotted_data(self) -> None:
        if not self.last_plot_payload or not self.last_plot_payload.get("curves"):
            # Create current mode data first if possible.
            self.plot_selected(show_popup_on_error=False)
        if not self.last_plot_payload or not self.last_plot_payload.get("curves"):
            messagebox.showinfo("No plotted data", "Plot something first, then export.")
            return

        path = filedialog.asksaveasfilename(
            title="Export plotted data",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            curves = self.last_plot_payload.get("curves", [])
            if str(path).lower().endswith(".csv"):
                out = self._curves_to_wide_dataframe(curves)
                out.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path) as writer:
                    summary = pd.DataFrame({
                        "Property": ["Plot mode", "Number of curves"],
                        "Value": [self.last_plot_payload.get("mode", ""), len(curves)],
                    })
                    summary.to_excel(writer, index=False, sheet_name="summary")
                    for i, curve in enumerate(curves, start=1):
                        name = self._safe_sheet_name(str(curve.get("name", f"curve_{i}")))
                        df = pd.DataFrame({"x": curve["x"], "y": curve["y"]})
                        if "std" in curve and curve["std"] is not None:
                            std = curve["std"]
                            if len(std) == len(df):
                                df["std"] = std
                        df.to_excel(writer, index=False, sheet_name=self._unique_sheet_name(writer, name))
            self.status_var.set(f"Exported plotted data: {path}")
        except Exception as exc:
            messagebox.showerror("Could not export", str(exc))

    def _curves_to_wide_dataframe(self, curves: Sequence[Dict[str, object]]) -> pd.DataFrame:
        columns: Dict[str, pd.Series] = {}
        for i, curve in enumerate(curves, start=1):
            raw_name = str(curve.get("name", f"curve_{i}"))
            name = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_")[:45] or f"curve_{i}"
            x = np.asarray(curve["x"])
            y = np.asarray(curve["y"])
            columns[f"x_{i}_{name}"] = pd.Series(x)
            columns[f"y_{i}_{name}"] = pd.Series(y)
            if "std" in curve and curve["std"] is not None:
                std = np.asarray(curve["std"])
                if len(std) == len(y):
                    columns[f"std_{i}_{name}"] = pd.Series(std)
        return pd.DataFrame(columns)

    @staticmethod
    def _safe_sheet_name(name: str) -> str:
        name = re.sub(r"[\\/*?:\[\]]", "_", name).strip()
        return (name[:31] or "sheet")

    def _unique_sheet_name(self, writer: pd.ExcelWriter, base: str) -> str:
        existing = set(writer.sheets.keys())
        name = base[:31]
        if name not in existing:
            return name
        for idx in range(2, 999):
            suffix = f"_{idx}"
            candidate = (base[: 31 - len(suffix)] + suffix)[:31]
            if candidate not in existing:
                return candidate
        return base[:28] + "_x"

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _maximize_main_window(self) -> None:
        try:
            self.state("zoomed")
            return
        except Exception:
            pass
        try:
            self.attributes("-zoomed", True)
        except Exception:
            pass

    def _change_theme(self) -> None:
        try:
            self.style.theme_use(self.theme_var.get())
        except Exception as exc:
            messagebox.showerror("Could not change theme", str(exc))

    @staticmethod
    def _safe_int(text: str, default: int, minimum: Optional[int] = None) -> int:
        try:
            value = int(float(str(text).strip()))
        except Exception:
            value = default
        if minimum is not None:
            value = max(minimum, value)
        return value


if __name__ == "__main__":
    app = MultiTxtPlotter()
    app.mainloop()