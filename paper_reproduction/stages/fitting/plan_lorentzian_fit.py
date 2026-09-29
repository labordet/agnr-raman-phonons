# SPDX-License-Identifier: CC-BY-4.0
"""Plan an optional Lorentzian fit of released means; --run starts the calculation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paper_reproduction.output_paths import prepare_output_directory as _output_directory

ROOT = Path(os.environ["RAMAN_DATA_ROOT"]).resolve()


def prepare_output_directory(path):
    return _output_directory(ROOT, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["primary", "configuration_V"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Run the full differential-evolution/bootstrap fit after writing its plan.")
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    output = prepare_output_directory(args.output or Path('outputs/paper_reproduction') / ('fitting_' + args.branch))
    protocol = json.loads((ROOT / "metadata/fitting_protocol.json").read_text(encoding="utf-8"))[args.branch]["config"]
    name = "cluster_fitting_current_euler.py" if args.branch == "primary" else "cluster_fitting_aligned_ro_chmid_locked.py"
    script = Path(__file__).resolve().parents[3] / "original_analysis/fitting" / name
    settings = ROOT / "metadata/original_records" / ("fitting_bounds_primary.json" if args.branch == "primary" else "fitting_bounds_configuration_V.json")
    settings_hash = hashlib.sha256(settings.read_bytes()).hexdigest()
    if settings_hash != protocol["settings_sha256"]:
        raise ValueError("Bounds do not match the recorded run manifest.")
    inputs = ROOT / "data/processed/temperature_mean"
    command = [sys.executable, "-B", str(script), "--root", str(inputs), "--settings", str(settings), "--output-root", str(output / "results"), "--n-jobs", str(args.jobs), "--families", *protocol["families"]]
    numeric_options = [
        "bootstrap_runs", "de_maxiter", "de_popsize", "de_tol", "ls_max_nfev",
        "fit_padding", "fit_x_min", "fit_x_max", "adaptive_bound_max_iterations",
        "adaptive_bound_tol", "adaptive_height_expand_factor", "adaptive_height_upper_rel_tol",
        "adaptive_position_expand_cm", "adaptive_width_expand_factor", "anti_burial_penalty",
        "required_min_area_share", "required_min_visibility", "ch_left_master",
        "ch_left_center_dominance_penalty", "ch_left_identity_penalty", "ch_left_min_area_share",
        "ch_left_min_visibility", "ch_left_own_center_share", "ch_left_slave_max_area_ratio",
        "ch_left_slave_min_area_ratio", "ch_left_width_ratio_max", "ch_left_width_ratio_min",
        "ch_left_width_ratio_penalty",
    ]
    for key in numeric_options:
        command.extend(["--" + key.replace("_", "-"), str(protocol[key])])
    command.append("--adaptive-bounds" if protocol["adaptive_bounds"] else "--no-adaptive-bounds")
    if protocol["anti_burial_optional"]:
        command.append("--anti-burial-optional")
    if not protocol["resume_enabled"]:
        command.append("--force-rerun")
    # These values are explicit defaults of the retained source, absent from the run manifest.
    source_defaults = {"bootstrap_jitter": 0.03, "seed": 20260622, "peak_window_weight": 6.0,
                       "peak_window_margin": 0.0, "peak_window_height_weight": 2.0,
                       "peak_window_height_floor_percentile": 10.0}
    command.extend(["--peak-window-weighting", "--peak-window-height-weighting"])
    for key, value in source_defaults.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    counts = {family: len(list((inputs / family).rglob("*.txt"))) for family in protocol["families"]}
    plan = {
        "branch": args.branch, "command": command, "recorded_bounds_sha256": settings_hash,
        "released_mean_counts": counts, "historical_campaign_spectra": protocol["discovered_txt_files"],
        "retained_source_defaults_not_recorded_in_manifest": source_defaults,
        "scope": "Fits the available released temperature means. Does not replace accepted results or replay the interactive local-refit/selection history.",
        "verification": "Command construction and input/bounds validation checked; the stochastic fitting campaign has not been rerun or verified bitwise.",
    }
    (output / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if not args.run:
        return
    environment = {k: v for k, v in os.environ.items() if not k.startswith("RAMAN_")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(output / ".matplotlib")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
