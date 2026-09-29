# Fitting and selection

The recommended workflow reads decisions and accepted parameters from the Zenodo archive. It does not infer new selection rules from the final data.

## Spectrum treatment

The recorded raw-to-mean chain is raw export → stored spike changes → selected acquisition columns → wavelength interval → ALS baseline subtraction → temperature mean. `paper_reproduction/stages/preprocessing/reproduce_preprocessing.py` replays 158 processing chains and 155 means against the archived arrays. `raman_tools.baseline` provides the same ALS equation for independent use. Different configurations have recorded ALS settings; for the five replacement Figure 1c fitting means, configurations I–III use λ = 10⁶ and IV–V use λ = 6000, with p = 0.0055 and 10 iterations. Read the per-spectrum values from `metadata/preprocessing_provenance.json`; do not apply a single setting to every spectrum.

Configuration III has two different selection stages:

1. **Before averaging:** 28 candidate acquisitions were reviewed; 20 were retained. The omitted acquisitions generally had weaker Raman signal. There was no recorded fixed SNR cutoff. `metadata/spectrum_selection.csv` and the original before-averaging records supply the decisions.
2. **After fitting:** temperature means at 125, 145, 185 and 195 K were omitted from the accepted set for RBLM, D and G together. The original rejection record does not state one common fit-failure reason for all four. The final accepted parameters are in `data/derived/peak_parameters.csv`.

## Full-window fits

`original_analysis/fitting/cluster_fitting_current_euler.py` implements the primary 200–2000 cm⁻¹ sum-of-Lorentzians fit, with independent parameters for each spectrum. It uses differential evolution (`popsize=200`, `tol=1e-7`, `maxiter=100000`) followed by bounded nonlinear least squares. The configuration-V run used `cluster_fitting_aligned_ro_chmid_locked.py`. Bounds, adaptive-bound settings, and the regularisation in the overlapping 1211–1271 cm⁻¹ region are retained in the original code and archived configuration.

Outside the peak-profile windows, objective weight is 1. Inside, its base value is 6, multiplied by a locally normalized intensity factor between 1 and 2. `make_weight_vector` returns the **square root** of this objective weight because it multiplies residuals before squaring. Unweighted residuals over the full fitting window produce the reported R² and RMSE.

Full-window uncertainty is the standard deviation from 100 successful residual-bootstrap refits. Each synthetic spectrum is the accepted best fit plus residuals sampled with replacement; the refit starts from parameters perturbed by 3% relative to the optimum. The retained fitting programs and `metadata/fitting_protocol.json` specify the run. Optimization package versions may produce small differences in newly fitted parameters.

## Configuration-III RBLM refinement

Fifteen of the 16 retained configuration-III RBLM points have a local replacement. `original_analysis/fitting/Gamma_T_Comparison.py:trust_region_refit_single_peak` fixes the other fitted components and refines RBLM alone. Its 100 residual-bootstrap refits start from the accepted local optimum. D and G remain from the full-window fit. The archive's `metadata/original_records/unaligned_au_local_refit_replacements.csv` records those replacements, and the accepted parameter table incorporates the retained results.

The optional fitting launcher runs a **new** global campaign against released means; it does not create these manual local replacements automatically. The standard paper command evaluates the retained accepted parameters without presenting a fresh fit as the historical one.
