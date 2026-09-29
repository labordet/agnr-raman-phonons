# Earlier Figure 1c reference-spectrum treatment

The `baseline_*.py`, `normalize_*.py` and `select_reference_spectra.py` scripts were extracted from project notebooks. They preserve the earlier seven-input Figure 1c processing workflow, which combined separately normalized 95 and 105 K spectra for III and IV by pointwise median.

The final Figure 1c uses the actual temperature means used in fitting. Generate its five curves with `paper_reproduction/plot_fig1c_fitting_means.py`; the exact measured temperatures are I/II/V 100 K, III 95 K and IV 105 K. The Zenodo data archive records the source spectra and both processing histories.
