# Original fitting and inspection programs

`cluster_fitting_current_euler.py` implements the primary full-window differential-evolution, bounded least-squares and residual-bootstrap fit. `cluster_fitting_aligned_ro_chmid_locked.py` is the configuration-V branch. `Gamma_T_Comparison.py` contains the later local configuration-III RBLM refit and interactive selection tools. `cluster_fitting.py` and `cluster_fitting_aligned_ro_middle_passive.py` preserve distinct earlier fitting variants for comparison and adaptation.

`raman_peak_bounds_tuner_app.py` is a bounds editor. `lorentzian_individual_plotter.py` inspects fits to individual spectra; `lorentzian_mean_plotter.py` inspects fitted temperature means. The two viewers serve different stages of the analysis.

For the accepted paper results, use `paper_reproduction/`. To plan a new global campaign, use `paper_reproduction/stages/fitting/plan_lorentzian_fit.py` with the Zenodo archive. These original programs retain research-era defaults and graphical application behavior.
