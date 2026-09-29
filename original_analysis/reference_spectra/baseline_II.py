"""
ALS baseline-removal + figure export + inline preview
----------------------------------------------------
* Processes every .txt in FOLDER (any # of Y columns).
* Saves all corrected / baseline spectra to two Excel files.
* Saves **every** spectrum plot to PLOTS_ALS/<folder>/.
* Shows **one** representative plot per temperature inline,
  and every plot now includes the original file-name as a subtitle.
"""

# ───────────────────────── imports ─────────────────────────
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Jupyter-only import: harmless in plain Python
try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs): pass

# ─────────────────────── parameters ───────────────────────
FOLDER          = r'Aligned_Au_8A'           # <── change if needed
ROI_RANGE       = (52, 2460)                 # cm-1
MASKED_REGIONS  = [
    # (52, 61), (210, 217), (926, 932),
    # (1094, 1108), (2200, 2600)
    #(52, 61), (2450, 2460)
]

ALS_LAMBDA      = 1e4
ALS_P           = 5e-4
ALS_NITER       = 20

PLOT_ROOT       = 'PLOTS_ALS'                # root for PNGs
# ───────────────────────────────────────────────────────────


# ---------- ALS helper ----------
def als_baseline(y, lam=ALS_LAMBDA, p=ALS_P, niter=ALS_NITER, mask=None):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        if mask is not None:
            w[mask] = 0
        z = spsolve(diags(w, 0) + lam * D @ D.T, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    if mask is not None:
        z[mask] = y[mask]
    return z


# ---------- utilities ----------
TEMP_RE = re.compile(r'_(\d+)K', re.IGNORECASE)
def extract_temp(fname: str) -> str:
    m = TEMP_RE.search(fname)
    if not m:
        raise ValueError(f"No *_NUMK* token in '{fname}'")
    return f"{m.group(1)}K"


# ---------- containers ----------
processed_parts = []          # to concat later
baseline_parts  = []
temp_counter    = defaultdict(int)
shown_once      = set()       # temps already displayed inline

plot_dir = os.path.join(PLOT_ROOT, os.path.basename(FOLDER))
os.makedirs(plot_dir, exist_ok=True)


# ────────────────── main loop ──────────────────
for fname in sorted(os.listdir(FOLDER)):
    if not fname.lower().endswith('.txt'):
        continue

    data  = np.loadtxt(os.path.join(FOLDER, fname))
    x_all = data[:, 0]
    ys    = data[:, 1:] if data.ndim > 1 else data[:, [1]]

    roi_mask = (x_all >= ROI_RANGE[0]) & (x_all <= ROI_RANGE[1])
    x_roi    = x_all[roi_mask]

    hard_mask = np.zeros_like(x_roi, dtype=bool)
    for lo, hi in MASKED_REGIONS:
        hard_mask |= (x_roi >= lo) & (x_roi <= hi)

    temperature = extract_temp(fname)

    for col_idx in range(ys.shape[1]):
        y_roi = ys[:, col_idx][roi_mask]
        z            = als_baseline(y_roi, mask=hard_mask)
        y_corrected  = y_roi - z

        temp_counter[temperature] += 1
        suffix = f"_{temp_counter[temperature]}" if temp_counter[temperature] > 1 else ""

        # collect for Excel
        processed_parts.append(
            pd.DataFrame({f"X_{temperature}{suffix}": x_roi,
                          f"Y_{temperature}{suffix}": y_corrected})
        )
        baseline_parts.append(
            pd.DataFrame({f"X_{temperature}{suffix}": x_roi,
                          f"BL_{temperature}{suffix}": z})
        )

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_roi, y_roi,        label='Original')
        ax.plot(x_roi, z,            label='Baseline')
        ax.plot(x_roi, y_corrected,  label='Corrected')
        ax.set_xlabel('Raman Shift (cm$^{-1}$)')
        ax.set_ylabel('Intensity')

        # main title + subtitle
        ax.set_title(f"{temperature}{suffix} – ALS baseline\n{fname}",
                     fontweight='bold', fontsize=11, pad=14)

        ax.legend()
        fig.tight_layout()

        # save PNG
        fig.savefig(os.path.join(plot_dir, f"{temperature}{suffix}.png"), dpi=300)

        # show inline only once per temperature
        if temperature not in shown_once:
            display(fig)
            shown_once.add(temperature)

        plt.close(fig)


# ────────────────── final save ──────────────────
processed_df = pd.concat(processed_parts, axis=1)
baseline_df  = pd.concat(baseline_parts,  axis=1)

processed_out = f"PROCESSED_SPECTRA_{os.path.basename(FOLDER)}.xlsx"
baseline_out  = f"BASELINE_SPECTRA_{os.path.basename(FOLDER)}.xlsx"

processed_df.to_excel(processed_out, index=False)
baseline_df.to_excel(baseline_out,  index=False)

print(
    "✅ Finished!\n"
    f"  • {processed_out}\n"
    f"  • {baseline_out}\n"
    f"  • plots → {plot_dir}\n"
    "Displayed one plot per temperature inline ✔️"
)