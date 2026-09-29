# From spectra to paper outputs

Run `python paper_reproduction/reproduce_all.py --data /path/to/ZENODO_RELEASE`. Its output directory contains stage logs and numerical checks. Paths below are relative to the unpacked Zenodo archive unless they begin with a repository directory.

| Step | Input and calculation | Program | Output |
|---|---|---|---|
| 1. Spectrum treatment | `data/raw/`, `metadata/spike_correction_deltas.csv`, `metadata/spectrum_selection.csv`, `metadata/preprocessing_provenance.json`; replay changes, selection and ALS | `paper_reproduction/stages/preprocessing/reproduce_preprocessing.py` | `preprocessing/baseline_corrected/`, `preprocessing/temperature_mean/`, verification |
| 2. Fitting means | Recorded ALS-corrected, selected columns grouped by temperature | Same preprocessing stage | 155 means checked against `data/processed/temperature_mean/` |
| 3. Spectral parameters | Accepted Lorentzian component tables in `data/derived/fits/` | `paper_reproduction/stages/analysis/verify_reported_results.py` | Comparison with `data/derived/peak_parameters.csv` (positions, FWHM and uncertainty columns) |
| 4. Temperature and thermal models | Accepted peaks, Au/CNT/sapphire thermal-expansion inputs | `paper_reproduction/stages/thermal/reproduce_thermal.py` | Figure S6 and Tables S1–S4, with numerical comparisons |
| 5. Main figure data | Fitting means, selected spectra, accepted peaks and model coefficients | `paper_reproduction/stages/figures_main/reproduce_main.py` | Figures 2–4 and S8 components and numerical arrays |
| 6. Supplementary figures | Processed spectra, accepted components and linewidth coefficients | `paper_reproduction/stages/figures_si/reproduce_si.py` | Figures S1–S5, S7, S9 and S10 numerical/plot outputs |

The global differential-evolution/bootstrap fit and the later 15 local configuration-III RBLM refits are retained in `original_analysis/fitting/`, along with the final accepted parameters in the data archive. They are computationally expensive and partly involved interactive review. The standard command evaluates their retained results. The optional fresh-fitting launcher is described in [Fitting and selection](FITTING_AND_SELECTION.md).

## Figure and table guide

| Paper item | Released input | Program and generated result | Presentation boundary |
|---|---|---|---|
| Fig. 1 | Five fitting temperature means: I/II/V 100 K, III 95 K, IV 105 K | `paper_reproduction/plot_fig1c_fitting_means.py`: normalized five-curve CSV and plot | STM display images and final arrangement were assembled in PowerPoint; archived artwork is in `figures/main/` |
| Fig. 2 | 109 selected temperature spectra; `metadata/main_spectral_selection.csv` | Main-figure stage: selected waterfall arrays and plot | Fourteen V inputs use interpolation on a common display grid; the measured files are unchanged. Insets and headings use the retained slide |
| Fig. 3 | Accepted RBLM/D/G positions, model coefficients, TEC curves | Main-figure stage: data, thermomechanical and Klemens curves | Displacement illustrations and layout use the retained slide |
| Fig. 4 | Accepted peak positions | Main-figure stage: zero-temperature reference frequencies | Numerical figure generated directly |
| Fig. S1 | Raw/spike-cleaned example pairs | SI stage: spike-change display arrays | Direct plot |
| Fig. S2 | Spike-cleaned and ALS-corrected examples | SI stage: baseline comparison | Direct plot |
| Fig. S3 | Final center-position bounds and representative spectra | SI stage: fitting-window plot | Direct plot |
| Fig. S4 | Five representative means and accepted components | SI stage: full-window Lorentzian decomposition | Direct plot |
| Fig. S5 | I 100 K G component | SI stage: Lorentzian parameter illustration | Direct plot |
| Fig. S6 | Au/CNT/sapphire TEC inputs | Thermal stage: recalculated curves | Source range and low-temperature continuations are documented in the data archive |
| Fig. S7 | Aligned-Au G-region means and accepted fits | SI stage: local residual arrays | Direct plot |
| Fig. S8 | Three configuration-II thermal paths | Main-figure stage: independent model curves | Direct plot |
| Fig. S9 | Accepted main-path FWHM values and linewidth coefficients | SI stage: Klemens linewidth curves | Direct plot |
| Fig. S10 | Configuration-II path FWHM values and coefficients | SI stage: path linewidth curves | Direct plot |
| Table S1 | Accepted peak positions and TEC inputs | Thermal stage: thermomechanical coefficients | Compare with `tables/supplementary/Table_S1_model_parameters.csv` |
| Table S2 | Measured and model peak positions | Thermal stage: temperature slopes | Compare with `tables/supplementary/Table_S2_temperature_slopes.csv` |
| Table S3 | Accepted model coefficients | Thermal stage: 80–290 K shift decomposition | Compare with `tables/supplementary/Table_S3_shift_decomposition.csv` |
| Table S4 | Configuration-II Heating 1, Cooling 1 and Heating 2 peaks | Thermal stage: independent path fits and slopes | Compare with `tables/supplementary/Table_S4_thermal_paths.csv` |

For exact source filenames, acquisition paths, settings and figure assembly files, consult the data archive's `metadata/figure_provenance.csv` and `metadata/table_provenance.csv`. The generated plots are numerical components; final figure artwork in the archive remains the publication reference.
