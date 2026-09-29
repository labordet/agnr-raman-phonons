#!/usr/bin/env python3
"""
Stacked line plots at ~100 K **plus Excel export (variable‑length tolerant)**
=========================================================================

This script replicates the original stacked‑spectra plotting code and adds an
export step that writes **exactly** the X/Y arrays used in the plot to an Excel
workbook called **SPECTRA_MIRA.xlsx** *even when the spectra have different
lengths*.

Key points
----------
* Each spectrum contributes **two** columns to the workbook:
  ``X_<legend_label>`` and ``Y_<legend_label>`` (spaces replaced by underscores
  for Excel‑friendly headers). Five spectra → *10 columns total*.
* Unequal array lengths are handled by storing every array as a
  *pandas Series*; when pandas builds the DataFrame it automatically aligns the
  series on a shared integer index, padding shorter columns with ``NaN``.
* All other behaviour (line styling, offsets, legend labels, etc.) matches the
  previous version.

Run the script in the folder that contains the five ``NORM_SPECTRA_…`` files.

Requirements
------------
- pandas (with openpyxl as Excel writer)
- numpy
- matplotlib
- seaborn (optional, for styling)
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ────────── USER PARAMETERS ──────────────────────────────────────────────
FIG_WIDTH         = 14     # figure width (inches)
FIG_HEIGHT        = 6      # figure height (inches)
LINE_WIDTH        = 3.5    # thickness of each line
OFFSET_STEP       = 1.2    # vertical offset between stacked spectra
MARGIN_LEFT_RIGHT = 0.0    # 0 = no space at all on left/right
X_MIN             = 200    # Raman‑shift lower limit (cm⁻¹) or None
X_MAX             = 1800   # Raman‑shift upper limit (cm⁻¹) or None
NUM_X_TICKS       = 8      # number of major ticks on the x‑axis
OUTPUT_EXCEL      = "SPECTRA_MIRA.xlsx"  # name of the exported workbook
# ─────────────────────────────────────────────────────────────────────────

# ────────── GLOBAL MATPLOTLIB STYLING ───────────────────────────────────
sns.set_style("whitegrid")
mpl.rcParams.update({
    "figure.figsize":    (FIG_WIDTH, FIG_HEIGHT),
    "font.family":       "sans-serif",
    "font.size":         14,
    "axes.labelsize":    16,
    "axes.titlesize":    16,
    "axes.linewidth":    1.5,
    "legend.fontsize":   13,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
})

# ────────── LIST OF INPUT FILES ─────────────────────────────────────────
FILES: List[str] = [
    "NORM_SPECTRA_Aligned_Au_3A.xlsx",
    "NORM_SPECTRA_Aligned_Au_8A.xlsx",
    "NORM_SPECTRA_Unaligned_Au_8A.xlsx",
    "NORM_SPECTRA_Aligned_RO_8A.xlsx",
    "NORM_SPECTRA_Unaligned_RO_8A.xlsx",
]

# ────────── PARSING & STYLING HELPERS ───────────────────────────────────
def parse_sample(fname: str):
    name = Path(fname).stem
    aligned   = "Aligned" in name
    coverage  = "3A" if "3A" in name else ("8A" if "8A" in name else "UnknownCov")
    substrate = "Au" if "_Au_" in name else ("RO" if "_RO_" in name else "UnknownSub")
    return {"aligned": aligned, "coverage": coverage, "substrate": substrate}

BASE_COLORS = {"Au": "goldenrod", "RO": "teal"}

def get_line_style(meta):
    color = BASE_COLORS.get(meta["substrate"], "gray")
    alpha = 1.0 if meta["coverage"] == "8A" else 0.6
    linestyle = "-" if meta["aligned"] else "--"
    return {"color": color, "alpha": alpha, "linestyle": linestyle, "linewidth": LINE_WIDTH}

# ────────── GRAB ~100 K DATA ────────────────────────────────────────────
_COL_PAT = re.compile(r"^(X|Y)_(\d+(?:\.\d+)?)K$")

def find_100K_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    temps = {}
    for col in df.columns:
        m = _COL_PAT.match(col)
        if not m:
            continue
        prefix, temp_str = m.group(1), m.group(2)
        t = float(temp_str)
        temps.setdefault(t, {"X": None, "Y": None})[prefix] = col

    if not temps:
        raise ValueError("No X_??K / Y_??K columns found in DataFrame.")

    all_temps = np.array(list(temps.keys()))
    min_diff = np.abs(all_temps - 100.0).min()
    close_temps = all_temps[np.isclose(np.abs(all_temps - 100.0), min_diff)]

    x_arrs, y_arrs = [], []
    for t in close_temps:
        pair = temps[t]
        if pair["X"] is None or pair["Y"] is None:
            continue
        x_arrs.append(df[pair["X"]].to_numpy())
        y_arrs.append(df[pair["Y"]].to_numpy())

    if not x_arrs:
        raise ValueError("No complete (X, Y) column pairs found for ~100 K in the file.")

    x_final = np.nanmedian(np.vstack(x_arrs), axis=0)
    y_final = np.nanmedian(np.vstack(y_arrs), axis=0)
    return x_final, y_final, close_temps.mean()

# ────────── MAIN PLOTTING & EXPORT LOGIC ───────────────────────────────

def main():
    fig, ax = plt.subplots()

    excel_cols = {}

    for i, fname in enumerate(FILES):
        df = pd.read_excel(fname, engine="openpyxl")
        meta = parse_sample(fname)
        style = get_line_style(meta)

        xvals, yvals, used_temp = find_100K_columns(df)

        offset = i * OFFSET_STEP
        yvals_offset = yvals + offset

        cov, sub = meta["coverage"], meta["substrate"]
        align_str = "aligned" if meta["aligned"] else "unaligned"
        temp_str = "" if np.isclose(used_temp, 100.0) else f" (~{used_temp:.1f}K)"
        legend_label = f"{cov}-{sub} ({align_str}){temp_str}"

        ax.plot(xvals, yvals_offset, label=legend_label, **style)

        header = legend_label.replace(" ", "_")
        # Store as Series so variable lengths are automatically aligned/padded
        excel_cols[f"X_{header}"] = pd.Series(xvals)
        excel_cols[f"Y_{header}"] = pd.Series(yvals)

    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)")
    ax.set_title("100 K stacked spectra")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, 0.98), frameon=False, ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(NUM_X_TICKS))

    if X_MIN is not None and X_MAX is not None:
        ax.set_xlim(X_MIN, X_MAX)

    ax.margins(x=0)
    fig.subplots_adjust(left=MARGIN_LEFT_RIGHT,
                        right=1.0 - MARGIN_LEFT_RIGHT,
                        top=0.95,
                        bottom=0.15)
    fig.tight_layout()
    plt.show()

    # ─── Export to Excel ────────────────────────────────────────────────
    print(f"\nExporting exact plot data to '{OUTPUT_EXCEL}' …")
    df_out = pd.DataFrame(excel_cols)
    # Keep only first 10 columns (5 spectra)
    if df_out.shape[1] > 10:
        df_out = df_out.iloc[:, :10]
    df_out.to_excel(OUTPUT_EXCEL, index=False)
    print("Done.\n")


if __name__ == "__main__":
    main()