# Code map

| Location | Role | What to use it for |
|---|---|---|
| `paper_reproduction/reproduce_all.py` | Paper reproduction | One command for the recorded calculation chain and paper outputs |
| `paper_reproduction/stages/preprocessing/` | Paper reproduction | Replay spike changes, selection, ALS and means |
| `paper_reproduction/plot_fig1c_fitting_means.py` | Paper reproduction | Figure 1c numerical curves from fitting means |
| `paper_reproduction/stages/analysis/` | Paper reproduction | Compare accepted peak positions, widths and uncertainties with component records |
| `paper_reproduction/stages/thermal/` | Paper reproduction | Thermal-expansion inputs, model calculations and Tables S1–S4 |
| `paper_reproduction/stages/figures_main/` | Paper reproduction | Main figure numerical components and Figure S8 |
| `paper_reproduction/stages/figures_si/` | Paper reproduction | Supplementary spectral and linewidth figures |
| `paper_reproduction/stages/fitting/` | Optional diagnostic | Plan or explicitly launch a new global fitting campaign |
| `raman_tools/` | Reusable utility | ALS, Lorentzian methods, recorded selection access and fit quality |
| `thermal_models/` | Reusable utility | Three-phonon and TEC functions |
| `original_analysis/fitting/` | Original analysis | Global optimizer, local RBLM refit, bounds editor and fit viewers |
| `original_analysis/preprocessing/` | Original analysis | Graphical spike/selection and baseline applications |
| `original_analysis/figures_main/`, `figures_si/`, `thermal/` | Original analysis | Paper figure and physical-model programs |
| `original_analysis/reference_spectra/` | Historical analysis | Earlier seven-input reference-spectrum processing for Figure 1c |
| `original_analysis/exploration/` | Exploratory analysis | Additional comparisons and inspection tools, not final parameter generators |
| `examples/` | Reusable example | Small demonstrations independent of the full paper run |

The original programs are retained as research source. The paper runner uses portable stage programs that read relative paths within the unpacked data archive and write to its separate output area. Only the launcher under `paper_reproduction/stages/fitting/` starts new global peak fitting, and only with `--run`.
