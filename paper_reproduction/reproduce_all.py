"""Run the recorded paper calculations against an unpacked Zenodo data archive.

The standard workflow replays preprocessing and evaluates the retained accepted
fit parameters. The historical stochastic fitting campaign is available
separately and is not silently substituted for the accepted results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_reproduction.output_paths import prepare_output_directory

STAGES = (
    ("preprocessing", "preprocessing/reproduce_preprocessing.py", "--output"),
    ("fig1c_fitting_means", "../plot_fig1c_fitting_means.py", "--output"),
    ("reported_results", "analysis/verify_reported_results.py", "--output"),
    ("thermal", "thermal/reproduce_thermal.py", "--output-dir"),
    ("main_figures", "figures_main/reproduce_main.py", "--output-dir"),
    ("si_figures", "figures_si/reproduce_si.py", "--output"),
)


def data_directory(path: Path) -> Path:
    """Require the expected archive layout before running any stage."""
    root = path.expanduser().resolve(strict=True)
    required = (
        "metadata/preprocessing_provenance.json",
        "metadata/fitting_protocol.json",
        "data/derived/peak_parameters.csv",
        "data/processed/temperature_mean",
        "checksums_sha256.txt",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise ValueError(f"Not an unpacked Zenodo data archive: missing {', '.join(missing)}")
    return root


def output_directory(root: Path, path: Path) -> Path:
    """Create a result folder away from the input archive."""
    return prepare_output_directory(root, path)


def run(data: Path, output: Path) -> dict:
    root = data_directory(data)
    destination = output_directory(root, output)
    here = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(RAMAN_DATA_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(destination / ".matplotlib"),
               PYTHONPATH=str(here.parent) + os.pathsep + env.get("PYTHONPATH", ""))
    summary = {
        "data_root": str(root),
        "method": "Replay recorded treatment; evaluate retained accepted spectral fits",
        "stages": [],
    }
    for label, relative, option in STAGES:
        script = (here / "stages" / relative).resolve(strict=True)
        stage_output = destination / label
        command = [sys.executable, "-B", str(script), option, str(stage_output)]
        print(f"[{label}] {script.name}", flush=True)
        result = subprocess.run(command, cwd=here.parent, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        log = destination / f"{label}.log"
        log.write_text(result.stdout, encoding="utf-8")
        summary["stages"].append({"name": label, "exit_code": result.returncode,
                                  "log": log.name})
        (destination / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {log}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="Directory containing the unpacked Zenodo data archive")
    parser.add_argument("--output", type=Path,
                        help="Writable result directory outside the data archive (default: ./outputs/paper_reproduction)")
    args = parser.parse_args()
    root = data_directory(args.data)
    summary = run(root, args.output or Path("outputs/paper_reproduction"))
    print(f"Completed {len(summary['stages'])} stages; see the output run_summary.json")


if __name__ == "__main__":
    main()
