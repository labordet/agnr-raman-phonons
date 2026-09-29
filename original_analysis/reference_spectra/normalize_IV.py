# ── Normalise baseline-corrected spectra to their 1590–1620 cm-¹ peak ──────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

try:                              # pretty inline display if running in Jupyter
    from IPython.display import display
except ImportError:
    def display(*_): pass

FOLDER_TAG  = "Unaligned_RO_8A"           # => PROCESSED_SPECTRA_Aligned_Au_8A.xlsx
PEAK_RANGE  = (1595, 1615)              # cm-¹ window used for scaling

in_file  = Path(f"PROCESSED_SPECTRA_{FOLDER_TAG}.xlsx")
out_file = Path(f"NORM_SPECTRA_{FOLDER_TAG}.xlsx")

if not in_file.exists():
    raise FileNotFoundError(f"\nCannot find {in_file.resolve()}\n"
                            "Run (or move) the ALS preprocessing first!")

df = pd.read_excel(in_file)

shown_once = set()                       # temperatures already plotted
TEMP_RE = re.compile(r'_(\d+)K', re.IGNORECASE)

for y_col in [c for c in df.columns if c.startswith("Y_")]:
    x_col = "X_" + y_col[2:]             # matching X column
    if x_col not in df.columns:
        print(f"⚠️  Skipped {y_col}: no {x_col} column found.")
        continue

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    mask = (x >= PEAK_RANGE[0]) & (x <= PEAK_RANGE[1])
    if not mask.any():
        print(f"⚠️  {y_col}: no data inside {PEAK_RANGE}; left unchanged.")
        continue

    peak_max = y[mask].max()
    if peak_max == 0 or np.isnan(peak_max):
        print(f"⚠️  {y_col}: peak max is zero/NaN; left unchanged.")
        continue

    y_norm = y / peak_max
    df[y_col] = y_norm                   # overwrite with normalised values

    # ­­­­­­­­­­­­­­­­­­­­­ inline preview (one per temperature) ­­­­­­­­­­­­­­­
    m = TEMP_RE.search(y_col)
    if m:
        temp = f"{m.group(1)}K"
        if temp not in shown_once:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(x, y_norm, lw=1)
            ax.set_xlabel('Raman Shift (cm$^{-1}$)')
            ax.set_ylabel('Normalised Intensity')
            ax.set_title(f"{temp} – normalised (band {PEAK_RANGE[0]}–{PEAK_RANGE[1]} cm⁻¹)")
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            shown_once.add(temp)

df.to_excel(out_file, index=False)
print(f"✅ Normalised spectra saved as {out_file.resolve()}")