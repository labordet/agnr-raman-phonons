# Raman and phonon analysis of 9-armchair graphene nanoribbon arrays

Scientific Python code associated with **“Surface-Dependent Phonon Dynamics in 9-Armchair Graphene Nanoribbon Arrays”** by Ángel Labordet Álvarez, Gabriela Borin Barin, Michel Calame and Mirjana Dimitrievska.

The repository contains the programs developed for Raman spectrum treatment, Lorentzian fitting, temperature-dependent peak analysis, thermal-expansion calculations, linewidth models and figure data. `paper_reproduction/` gives one ordered route through the calculations used for the paper. `raman_tools/` and `thermal_models/` expose useful functions for other measurements. The research spectra and accepted numerical results are distributed separately as a Zenodo data archive.

## Start here

Use Python 3.11. Download and unpack the research data from **<ZENODO_DATA_DOI>**. Until the record DOI is assigned, obtain the final `ZENODO_RESEARCH_DATA.zip` from the authors. Keep its internal directory structure.

```sh
python -m venv .venv
# Activate the environment using your shell's normal command.
python -m pip install -r requirements.txt
python -m pip install -e .
python paper_reproduction/reproduce_all.py --data /path/to/ZENODO_RELEASE
```

The command writes to `outputs/github_reproduction/` inside the unpacked data directory and reports each stage in `run_summary.json`. It replays recorded spike changes, selection, ALS subtraction and temperature means, checks the accepted spectral parameters, recalculates thermal and temperature models, and writes numerical data and plots for the main and supplementary figures. Its comparisons use the accepted tables and figure data shipped with the archive. The command does **not** launch the long stochastic spectral-fitting campaign or replace accepted fit parameters. See [Paper workflow](docs/PAPER_WORKFLOW.md) for the exact sequence and outputs.

## Scientific methods and measurements

All temperature-dependent Raman measurements used 785 nm excitation, a 300 g mm⁻¹ grating and the same 50× long-working-distance objective. The five configurations are:

| Paper label | Sample | Measurement |
|---|---|---|
| I | Aligned Au, low coverage | 2 µm line scan, 12 points, 40 s per point |
| II | Aligned Au, high coverage | Single point, 14 accumulations × 60 s |
| III | Unaligned Au, high coverage | Single point, 20 accumulations × 60 s |
| IV | Unaligned RO, high coverage | 69.73 µm line scan, 20 points, 1 s per point |
| V | Aligned RO, high coverage | 10 × 10 µm² map, 5 × 5 grid, 10 s per point |

Configuration II contains three thermal paths. The historical `DOWN_1` identifier for configuration V is the paper's Heating 1 path. A `488nm_R` token in some configuration-IV raw filenames is a naming error confirmed by the data owner; those measurements used 785 nm excitation. The archive retains the original filenames and explains the mapping in its metadata.

The full-window spectral fit spans 200–2000 cm⁻¹. It uses simultaneous Lorentzians, differential evolution, bounded least-squares refinement, and 100 successful residual-bootstrap refits. The objective's peak-profile windows receive base weight 6 and a local intensity multiplier from 1 to 2; reported R² and RMSE use unweighted full-window residuals. Fifteen accepted configuration-III RBLM points were subsequently refined locally while other components stayed fixed. [Fitting and selection](docs/FITTING_AND_SELECTION.md) gives the retained settings and the two distinct configuration-III selection stages.

## Repository layout

| Directory | Purpose |
|---|---|
| `paper_reproduction/` | Data-archive runner, figure 1c generator, and independently callable calculation stages |
| `raman_tools/` | ALS, Lorentzian fitting access, selection and fit-quality functions |
| `thermal_models/` | Retained three-phonon and thermal-expansion functions |
| `original_analysis/` | Scientifically useful original research scripts, including fitting and graphical applications |
| `examples/` | Small standalone demonstrations |
| `tests/` | Numerical unit tests and optional data-archive agreement tests |
| `docs/` | Paper-to-code mapping and program guide |

The original scripts preserve their historical numerical methods and may refer to old local directory names in defaults or comments. Use `paper_reproduction/` for a portable analysis of the Zenodo archive. [Code map](docs/CODE_MAP.md) distinguishes paper programs, reusable tools, optional diagnostics and exploratory work.

## Run parts independently

Each stage under `paper_reproduction/stages/` can be run separately. Set `RAMAN_DATA_ROOT` to the unpacked archive directory and supply an output path inside its `outputs/` directory. For example, in a POSIX shell:

```sh
RAMAN_DATA_ROOT=/path/to/ZENODO_RELEASE python paper_reproduction/stages/preprocessing/reproduce_preprocessing.py --output /path/to/ZENODO_RELEASE/outputs/preprocessing
```

In PowerShell, set `$env:RAMAN_DATA_ROOT = 'C:\path\to\ZENODO_RELEASE'` before the same Python command.

The optional full-window fitting launcher is `paper_reproduction/stages/fitting/plan_lorentzian_fit.py`. It prints and saves the command first; `--run` starts a new differential-evolution/bootstrap campaign. This is computationally expensive and does not replay the historical interactive configuration-III local-refit decisions. Install `.[fitting]` for the optional workbook dependency. Accepted parameters remain in the data archive.

Run the tests with `python -m unittest discover -s tests -v`. Set `ZENODO_DATA_ROOT` to include scientific agreement tests against the archive. The tested Windows environment was Python 3.11.14 with the package versions in `requirements.txt`; the recorded original cluster run used Python 3.11.6, and its package versions were not retained. Fonts and optimizers can differ across systems; the numerical checks specify tolerances at meaningful reported precision.

## Outputs, citation and license

The data archive supplies raw and processed spectra, accepted parameters, reference thermal-expansion inputs, published artwork and final tables. The repository supplies code only. Figure 1c uses the actual fitting means at 100, 100, 95, 105 and 100 K in I–V order. Figures 1–3 include final manual slide assembly beyond the numerical components. The released results and provenance records document these boundaries.

Use [CITATION.cff](CITATION.cff) to cite this software and cite the associated paper and Zenodo data record when identifiers are available. The authors' original code and documentation are licensed under [CC BY 4.0](LICENSE). External reference data and software dependencies retain their own terms.
