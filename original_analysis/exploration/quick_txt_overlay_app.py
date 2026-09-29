from __future__ import annotations

import math
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


APP_TITLE = "Quick TXT Spectra Overlay"


@dataclass
class SpectrumTrace:
    file_path: Path
    y_column: int
    x: np.ndarray
    y: np.ndarray
    multiplier: float = 1.0

    @property
    def label(self) -> str:
        suffix = "" if math.isclose(self.multiplier, 1.0) else f" x{self.multiplier:.6g}"
        if self.y_column == 1:
            return f"{self.file_path.stem}{suffix}"
        return f"{self.file_path.stem} | Y{self.y_column}{suffix}"


def _float_or_none(token: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    token = token.replace("\ufeff", "")
    try:
        return float(token)
    except ValueError:
        return None


def read_numeric_table(path: Path) -> np.ndarray:
    """Read a text table where column 1 is X and columns 2..N are spectra.

    The parser accepts whitespace, tab, comma, or semicolon separated files and
    quietly skips header/comment lines. Rows with missing values are trimmed to
    the widest common numeric column count found in the file.
    """
    rows: list[list[float]] = []
    splitter = re.compile(r"[\s,;]+")

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//")):
                continue
            values = [_float_or_none(tok) for tok in splitter.split(stripped)]
            numeric = [v for v in values if v is not None and math.isfinite(v)]
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
        raise ValueError("The file must have one X column and at least one Y column.")
    return data


class QuickTxtOverlayApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1350x850")
        self.minsize(900, 600)

        self.traces: list[SpectrumTrace] = []
        self.last_dir = tk.StringVar(value=str(Path.cwd()))

        self.normalize_var = tk.BooleanVar(value=False)
        self.legend_var = tk.BooleanVar(value=False)
        self.grid_var = tk.BooleanVar(value=False)
        self.x_start_var = tk.StringVar(value="")
        self.x_end_var = tk.StringVar(value="")
        self.offset_var = tk.StringVar(value="0")
        self.line_width_var = tk.StringVar(value="1.2")
        self.alpha_var = tk.StringVar(value="0.90")
        self.title_var = tk.StringVar(value="Overlayed TXT spectra")
        self.multiplier_var = tk.StringVar(value="1")

        self._build_ui()
        self._draw_plot()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=8)
        side.grid(row=0, column=0, sticky="ns")

        plot_area = ttk.Frame(self)
        plot_area.grid(row=0, column=1, sticky="nsew")
        plot_area.columnconfigure(0, weight=1)
        plot_area.rowconfigure(0, weight=1)

        ttk.Button(side, text="Open TXT files", command=self.open_files).grid(row=0, column=0, sticky="ew")
        ttk.Button(side, text="Clear", command=self.clear_files).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(side, text="Save figure", command=self.save_figure).grid(row=2, column=0, sticky="ew", pady=(4, 10))

        options = ttk.LabelFrame(side, text="Plot options", padding=8)
        options.grid(row=3, column=0, sticky="ew")
        options.columnconfigure(1, weight=1)

        ttk.Label(options, text="Title").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.title_var, width=24).grid(row=0, column=1, sticky="ew")

        ttk.Label(options, text="X start").grid(row=1, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.x_start_var, width=12).grid(row=1, column=1, sticky="ew")
        ttk.Label(options, text="X end").grid(row=2, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.x_end_var, width=12).grid(row=2, column=1, sticky="ew")

        ttk.Label(options, text="Offset").grid(row=3, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.offset_var, width=12).grid(row=3, column=1, sticky="ew")
        ttk.Label(options, text="Line width").grid(row=4, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.line_width_var, width=12).grid(row=4, column=1, sticky="ew")
        ttk.Label(options, text="Alpha").grid(row=5, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.alpha_var, width=12).grid(row=5, column=1, sticky="ew")

        ttk.Checkbutton(options, text="Normalize each trace", variable=self.normalize_var, command=self._draw_plot).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Checkbutton(options, text="Show legend", variable=self.legend_var, command=self._draw_plot).grid(
            row=7, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(options, text="Grid", variable=self.grid_var, command=self._draw_plot).grid(
            row=8, column=0, columnspan=2, sticky="w"
        )

        ttk.Button(options, text="Replot", command=self._draw_plot).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        multiplier_box = ttk.LabelFrame(side, text="Selected trace multiplier", padding=8)
        multiplier_box.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        multiplier_box.columnconfigure(1, weight=1)
        ttk.Label(multiplier_box, text="Multiply Y by").grid(row=0, column=0, sticky="w")
        multiplier_entry = ttk.Entry(multiplier_box, textvariable=self.multiplier_var, width=12)
        multiplier_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        multiplier_entry.bind("<Return>", lambda _event: self.apply_multiplier_to_selected())
        ttk.Button(multiplier_box, text="Apply selected", command=self.apply_multiplier_to_selected).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(multiplier_box, text="Apply same TXT", command=self.apply_multiplier_to_same_file).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Button(multiplier_box, text="Reset selected to 1", command=self.reset_selected_multipliers).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Button(multiplier_box, text="Reset all to 1", command=self.reset_all_multipliers).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )

        loaded = ttk.LabelFrame(side, text="Loaded spectra", padding=8)
        loaded.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        side.rowconfigure(5, weight=1)

        self.trace_list = tk.Listbox(loaded, width=52, height=24, selectmode=tk.EXTENDED)
        self.trace_list.grid(row=0, column=0, sticky="nsew")
        self.trace_list.bind("<<ListboxSelect>>", self._on_trace_select)
        loaded.columnconfigure(0, weight=1)
        loaded.rowconfigure(0, weight=1)
        scrollbar = ttk.Scrollbar(loaded, orient="vertical", command=self.trace_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.trace_list.configure(yscrollcommand=scrollbar.set)

        self.status_var = tk.StringVar(value="Open TXT files to overlay spectra.")
        ttk.Label(side, textvariable=self.status_var, wraplength=360, foreground="#444").grid(row=6, column=0, sticky="ew", pady=(8, 0))

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky="ew")

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select TXT spectra files",
            initialdir=self.last_dir.get() if Path(self.last_dir.get()).exists() else str(Path.cwd()),
            filetypes=[
                ("Text spectra", "*.txt *.TXT *.dat *.DAT *.csv *.CSV"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        loaded_count = 0
        failed: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            self.last_dir.set(str(path.parent))
            try:
                table = read_numeric_table(path)
                x = table[:, 0]
                for col_idx in range(1, table.shape[1]):
                    y = table[:, col_idx]
                    finite = np.isfinite(x) & np.isfinite(y)
                    if finite.sum() < 2:
                        continue
                    self.traces.append(
                        SpectrumTrace(
                            file_path=path,
                            y_column=col_idx,
                            x=x[finite].copy(),
                            y=y[finite].copy(),
                        )
                    )
                    loaded_count += 1
            except Exception as exc:  # noqa: BLE001 - shown to user with file name
                failed.append(f"{path.name}: {exc}")

        self._refresh_trace_list()
        self._draw_plot()

        msg = f"Loaded {loaded_count} spectra from {len(paths)} file(s)."
        if failed:
            msg += f" {len(failed)} file(s) skipped."
            messagebox.showwarning("Some files were skipped", "\n".join(failed[:12]))
        self.status_var.set(msg)

    def clear_files(self) -> None:
        self.traces.clear()
        self._refresh_trace_list()
        self._draw_plot()
        self.status_var.set("Cleared loaded spectra.")

    def apply_multiplier_to_selected(self) -> None:
        indices = self._selected_trace_indices()
        if not indices:
            messagebox.showinfo(APP_TITLE, "Select one or more spectra first.")
            return
        multiplier = self._parse_multiplier()
        if multiplier is None:
            return
        for idx in indices:
            self.traces[idx].multiplier = multiplier
        self._refresh_trace_list(select_indices=indices)
        self._draw_plot()
        self.status_var.set(f"Applied multiplier x{multiplier:.6g} to {len(indices)} selected spectrum/spectra.")

    def apply_multiplier_to_same_file(self) -> None:
        indices = self._selected_trace_indices()
        if not indices:
            messagebox.showinfo(APP_TITLE, "Select one spectrum from the TXT file you want to scale.")
            return
        multiplier = self._parse_multiplier()
        if multiplier is None:
            return
        file_path = self.traces[indices[0]].file_path
        affected = [idx for idx, trace in enumerate(self.traces) if trace.file_path == file_path]
        for idx in affected:
            self.traces[idx].multiplier = multiplier
        self._refresh_trace_list(select_indices=affected)
        self._draw_plot()
        self.status_var.set(f"Applied multiplier x{multiplier:.6g} to all {len(affected)} spectrum/spectra from {file_path.name}.")

    def reset_selected_multipliers(self) -> None:
        indices = self._selected_trace_indices()
        if not indices:
            messagebox.showinfo(APP_TITLE, "Select one or more spectra first.")
            return
        for idx in indices:
            self.traces[idx].multiplier = 1.0
        self.multiplier_var.set("1")
        self._refresh_trace_list(select_indices=indices)
        self._draw_plot()
        self.status_var.set(f"Reset {len(indices)} selected multiplier/s to x1.")

    def reset_all_multipliers(self) -> None:
        for trace in self.traces:
            trace.multiplier = 1.0
        self.multiplier_var.set("1")
        self._refresh_trace_list()
        self._draw_plot()
        self.status_var.set("Reset all multipliers to x1.")

    def save_figure(self) -> None:
        if not self.traces:
            messagebox.showinfo(APP_TITLE, "Open at least one TXT file first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save overlay figure",
            initialdir=self.last_dir.get() if Path(self.last_dir.get()).exists() else str(Path.cwd()),
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("TIFF", "*.tif"),
            ],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self.status_var.set(f"Saved figure: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", f"{exc}\n\n{traceback.format_exc()}")

    def _refresh_trace_list(self, select_indices: list[int] | tuple[int, ...] | None = None) -> None:
        self.trace_list.delete(0, tk.END)
        for trace in self.traces:
            self.trace_list.insert(
                tk.END,
                f"x{trace.multiplier:<9.6g} | {trace.file_path.name}   |   Y column {trace.y_column + 1}",
            )
        if select_indices:
            for idx in select_indices:
                if 0 <= idx < len(self.traces):
                    self.trace_list.selection_set(idx)
            self.trace_list.see(select_indices[0])

    def _selected_trace_indices(self) -> list[int]:
        return [int(idx) for idx in self.trace_list.curselection()]

    def _parse_multiplier(self) -> float | None:
        value = self.multiplier_var.get().strip()
        try:
            multiplier = float(value)
        except ValueError:
            messagebox.showerror(APP_TITLE, "Multiplier must be a number, for example 0.5, 2, or 10.")
            return None
        if not math.isfinite(multiplier):
            messagebox.showerror(APP_TITLE, "Multiplier must be a finite number.")
            return None
        return multiplier

    def _on_trace_select(self, _event: tk.Event | None = None) -> None:
        indices = self._selected_trace_indices()
        if not indices:
            return
        values = {self.traces[idx].multiplier for idx in indices}
        if len(values) == 1:
            self.multiplier_var.set(f"{next(iter(values)):.6g}")
        else:
            self.multiplier_var.set("")

    def _float_setting(self, value: str, default: float) -> float:
        try:
            parsed = float(value)
        except ValueError:
            return default
        if not math.isfinite(parsed):
            return default
        return parsed

    def _optional_float(self, value: str) -> float | None:
        value = value.strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    def _draw_plot(self) -> None:
        self.ax.clear()

        if not self.traces:
            self.ax.text(
                0.5,
                0.5,
                "Open TXT files to overlay spectra",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="#555555",
            )
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        self.ax.set_axis_on()
        x_start = self._optional_float(self.x_start_var.get())
        x_end = self._optional_float(self.x_end_var.get())
        offset = self._float_setting(self.offset_var.get(), 0.0)
        line_width = self._float_setting(self.line_width_var.get(), 1.2)
        alpha = min(max(self._float_setting(self.alpha_var.get(), 0.9), 0.05), 1.0)

        plotted = 0
        for idx, trace in enumerate(self.traces):
            x = trace.x
            y = trace.y.astype(float, copy=True)

            mask = np.ones_like(x, dtype=bool)
            if x_start is not None:
                mask &= x >= x_start
            if x_end is not None:
                mask &= x <= x_end
            if mask.sum() < 2:
                continue

            x_plot = x[mask]
            y_plot = y[mask]
            if self.normalize_var.get():
                span = np.nanmax(y_plot) - np.nanmin(y_plot)
                if math.isfinite(span) and span > 0:
                    y_plot = (y_plot - np.nanmin(y_plot)) / span

            y_plot = y_plot * trace.multiplier
            y_plot = y_plot + plotted * offset
            self.ax.plot(x_plot, y_plot, linewidth=line_width, alpha=alpha, label=trace.label)
            plotted += 1

        self.ax.set_title(self.title_var.get().strip() or "Overlayed TXT spectra", fontsize=15)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.grid(self.grid_var.get(), alpha=0.25)

        if self.legend_var.get() and plotted:
            self.ax.legend(loc="best", fontsize=8, frameon=False)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    app = QuickTxtOverlayApp()
    app.mainloop()


if __name__ == "__main__":
    main()
