#!/usr/bin/env python3
"""
Vergleichsskript: Standard-Deviation vs. Time-Weighted-Deviation (twdev)
im Rolling-Horizon-Framework mit stochastischer Varianz (Sigma)

Dieses Skript führt beide Varianten parallel aus und vergleicht die Ergebnisse.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from datetime import datetime

try:
    import tomllib as pytoml
except ModuleNotFoundError:
    import tomli as pytoml

from config.project_config import get_config_path
from src.Logger import Logger
from src.domain.Initializer import ExperimentInitializer
from src.CP_Experiment_Runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vergleich: Standard-Deviation vs. Time-Weighted-Deviation (twdev) im Rolling-Horizon-Framework"
    )
    parser.add_argument(
        "--util",
        type=float,
        required=True,
        help="Max bottleneck utilization (z.B. 0.75)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Simulation noise sigma (z.B. 0.1)",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=1800,
        help="Solver time limit in seconds (default: 1800)",
    )
    parser.add_argument(
        "--bound_no_improvement_time",
        type=int,
        default=600,
        help="Time in seconds with no bound improvement (default: 600)",
    )
    parser.add_argument(
        "--bound_warmup_time",
        type=int,
        default=60,
        help="Warmup time in seconds (default: 60)",
    )
    # Time-Weighted Deviation Parameter
    parser.add_argument(
        "--deviation_window_minutes",
        type=int,
        default=8 * 60,
        help="Time window in minutes for twdev scaling (default: 480)",
    )
    parser.add_argument(
        "--deviation_bucket_minutes",
        type=int,
        default=60,
        help="Bucket size in minutes for twdev scaling (default: 60)",
    )
    parser.add_argument(
        "--deviation_max_factor",
        type=int,
        default=None,
        help="Maximum scaling factor for twdev (default: None)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output/twdev_comparison",
        help="Output directory for comparison results (default: data/output/twdev_comparison)",
    )
    # Machine Blockade Parameter
    parser.add_argument(
        "--machine_blockade",
        type=str,
        default=None,
        help="Machine blockade in format 'machine:start:end' (e.g. 'M00:1500:1560'). Can be specified multiple times.",
        action="append",
    )

    args = parser.parse_args()

    # Load config
    config_path = get_config_path("experiments_config.toml", as_string=False)
    with open(config_path, "rb") as f:
        cfg = pytoml.load(f)

    run_cfg = cfg["run"]
    grid = cfg["grid"]

    source_name: str = run_cfg["source_name"]
    shift_length: int = int(run_cfg["shift_length"])
    total_shift_number: int = int(run_cfg["total_shift_number"])

    # Verwende erste Kombination aus Grid (oder Default-Werte)
    absolute_lateness_ratio = grid["absolute_lateness_ratio"][0]
    inner_tardiness_ratio = grid["inner_tardiness_ratio"][0]

    # Parse machine blockades
    machine_blockades = None
    if args.machine_blockade:
        machine_blockades = []
        for blockade_str in args.machine_blockade:
            parts = blockade_str.split(':')
            if len(parts) != 3:
                raise SystemExit(
                    f"Error: --machine_blockade must be in format 'machine:start:end', got: {blockade_str}"
                )
            machine_blockades.append({
                'machine': parts[0],
                'start': int(parts[1]),
                'end': int(parts[2])
            })

    # Output-Verzeichnis erstellen
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_results = {
        "timestamp": timestamp,
        "parameters": {
            "util": args.util,
            "sigma": args.sigma,
            "time_limit": args.time_limit,
            "bound_no_improvement_time": args.bound_no_improvement_time,
            "bound_warmup_time": args.bound_warmup_time,
            "deviation_window_minutes": args.deviation_window_minutes,
            "deviation_bucket_minutes": args.deviation_bucket_minutes,
            "deviation_max_factor": args.deviation_max_factor,
            "machine_blockades": machine_blockades,
            "shift_length": shift_length,
            "total_shift_number": total_shift_number,
        },
        "experiments": {}
    }

    print("\n" + "="*80)
    print("VERGLEICH: Standard-Deviation vs. Time-Weighted-Deviation (twdev)")
    print("="*80)
    print(f"Utilization: {args.util}")
    print(f"Sigma: {args.sigma}")
    print(f"Time Limit: {args.time_limit}s")
    print(f"Shifts: {total_shift_number} × {shift_length} min")
    if machine_blockades:
        print(f"Machine Blockades: {len(machine_blockades)}")
        for blockade in machine_blockades:
            print(f"  - {blockade['machine']}: {blockade['start']} to {blockade['end']}")
    else:
        print("Machine Blockades: None")
    print("="*80 + "\n")

    # ========================================================================
    # 1. Standard-Deviation Experiment
    # ========================================================================
    print("\n[1/2] Running STANDARD DEVIATION experiment...")
    print("-" * 80)

    exp_id_std = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{args.util:.2f}"),
        sim_sigma=args.sigma,
        experiment_type="CP",
    )

    logger_std = Logger(
        name=f"comparison_std_{timestamp}",
        log_file=f"comparison_std_{timestamp}.log"
    )

    result_std = run_experiment(
        experiment_id=exp_id_std,
        shift_length=shift_length,
        total_shift_number=total_shift_number,
        logger=logger_std,
        time_limit=args.time_limit,
        bound_no_improvement_time=args.bound_no_improvement_time,
        bound_warmup_time=args.bound_warmup_time,
        use_time_weighted_deviation=False,
        machine_blockades=machine_blockades,
    )

    comparison_results["experiments"]["standard_deviation"] = {
        "experiment_id": exp_id_std,
        "result": result_std,
    }

    print(f"✓ Standard Deviation experiment completed (ID: {exp_id_std})")

    # ========================================================================
    # 2. Time-Weighted-Deviation Experiment
    # ========================================================================
    print("\n[2/2] Running TIME-WEIGHTED DEVIATION (twdev) experiment...")
    print("-" * 80)

    exp_id_twdev = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{args.util:.2f}"),
        sim_sigma=args.sigma,
        experiment_type="CP",
    )

    logger_twdev = Logger(
        name=f"comparison_twdev_{timestamp}",
        log_file=f"comparison_twdev_{timestamp}.log"
    )

    result_twdev = run_experiment(
        experiment_id=exp_id_twdev,
        shift_length=shift_length,
        total_shift_number=total_shift_number,
        logger=logger_twdev,
        time_limit=args.time_limit,
        bound_no_improvement_time=args.bound_no_improvement_time,
        bound_warmup_time=args.bound_warmup_time,
        use_time_weighted_deviation=True,
        deviation_window_minutes=args.deviation_window_minutes,
        deviation_bucket_minutes=args.deviation_bucket_minutes,
        deviation_max_factor=args.deviation_max_factor,
        machine_blockades=machine_blockades,
    )

    comparison_results["experiments"]["time_weighted_deviation"] = {
        "experiment_id": exp_id_twdev,
        "result": result_twdev,
    }

    print(f"✓ Time-Weighted Deviation experiment completed (ID: {exp_id_twdev})")

    # ========================================================================
    # 3. Ergebnisse speichern
    # ========================================================================
    output_file = output_dir / f"comparison_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n" + "="*80)
    print("VERGLEICH ABGESCHLOSSEN")
    print("="*80)
    print(f"Standard Deviation Experiment ID: {exp_id_std}")
    print(f"Time-Weighted Deviation Experiment ID: {exp_id_twdev}")
    print(f"\nErgebnisse gespeichert in: {output_file}")
    print("="*80 + "\n")

    # ========================================================================
    # 4. Zusammenfassung
    # ========================================================================
    print("\nZUSAMMENFASSUNG:")
    print("-" * 80)
    
    # Shift-Summaries vergleichen
    std_shifts = result_std.get("shift_summaries", [])
    twdev_shifts = result_twdev.get("shift_summaries", [])
    
    if std_shifts and twdev_shifts:
        print(f"\n{'Shift':<8} {'Std-Dev Status':<20} {'twdev Status':<20}")
        print("-" * 80)
        for std_shift, twdev_shift in zip(std_shifts, twdev_shifts):
            shift_num = std_shift.get("shift", "?")
            std_status = std_shift.get("status", "?")
            twdev_status = twdev_shift.get("status", "?")
            print(f"{shift_num:<8} {std_status:<20} {twdev_status:<20}")
    
    print("\n" + "="*80)
    print("Weitere Analyse kann in der Datenbank durchgeführt werden:")
    print(f"  - Experiment {exp_id_std}: Standard Deviation")
    print(f"  - Experiment {exp_id_twdev}: Time-Weighted Deviation")
    print("="*80 + "\n")


if __name__ == "__main__":
    """
    Example usage:
    
    # Schneller Test (niedrige Utilization, kurze Shifts):
    python run_cp_twdev_comparison.py --util 0.65 --sigma 0.05 --time_limit 600 --bound_no_improvement_time 180 --bound_warmup_time 30
    
    # Standard-Test:
    python run_cp_twdev_comparison.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60
    
    # Mit custom twdev-Parametern:
    python run_cp_twdev_comparison.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 --deviation_window_minutes 480 --deviation_bucket_minutes 60 --deviation_max_factor 8
    
    # Mit Maschinenblockade:
    python run_cp_twdev_comparison.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 --machine_blockade M00:1500:1560
    
    # Mit mehreren Maschinenblockaden:
    python run_cp_twdev_comparison.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 --machine_blockade M00:1500:1560 --machine_blockade M03:2000:2100
    
    # Hohe Auslastung:
    python run_cp_twdev_comparison.py --util 0.85 --sigma 0.15 --time_limit 3600 --bound_no_improvement_time 900 --bound_warmup_time 120
    """
    main()

