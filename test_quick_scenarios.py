#!/usr/bin/env python3
"""
Schnelltests fuer fehlende Szenarien:
1. Deterministisches Baseline (sigma=0) - Standard-Dev + twdev
2. twdev + Maschinenblockade kombiniert

Jeder Test laeuft nur 2 Shifts fuer schnelle Ausfuehrung.
"""
from __future__ import annotations

from decimal import Decimal

try:
    import tomllib as pytoml
except ModuleNotFoundError:
    import tomli as pytoml

from config.project_config import get_config_path
from src.Logger import Logger
from src.domain.Initializer import ExperimentInitializer
from src.CP_Experiment_Runner import run_experiment


SHIFT_COUNT = 2
TIME_LIMIT = 300
BOUND_NO_IMPROVEMENT = 60
BOUND_WARMUP = 30


def _load_config():
    config_path = get_config_path("experiments_config.toml", as_string=False)
    with open(config_path, "rb") as f:
        cfg = pytoml.load(f)
    return cfg


def _create_experiment(cfg, sigma: float) -> int:
    run_cfg = cfg["run"]
    grid = cfg["grid"]
    return ExperimentInitializer.insert_experiment(
        source_name=run_cfg["source_name"],
        absolute_lateness_ratio=grid["absolute_lateness_ratio"][0],
        inner_tardiness_ratio=grid["inner_tardiness_ratio"][0],
        max_bottleneck_utilization=Decimal(f"{grid['max_bottleneck_utilization'][0]:.2f}"),
        sim_sigma=sigma,
        experiment_type="CP",
    )


def _safe_run(func, name: str) -> bool:
    try:
        func()
        print(f"  BESTANDEN: {name}")
        return True
    except Exception as e:
        if "Username and Password not accepted" in str(e) or "BadCredentials" in str(e):
            print(f"  BESTANDEN: {name} (E-Mail uebersprungen)")
            return True
        print(f"  FEHLGESCHLAGEN: {name}: {e}")
        return False


def test_deterministic_baseline_std():
    """sigma=0 -> keine stochastische Varianz, Standard-Deviation."""
    cfg = _load_config()
    exp_id = _create_experiment(cfg, sigma=0.0)
    logger = Logger(name="test_det_std", log_file="test_det_std.log")
    run_experiment(
        experiment_id=exp_id,
        shift_length=int(cfg["run"]["shift_length"]),
        total_shift_number=SHIFT_COUNT,
        logger=logger,
        time_limit=TIME_LIMIT,
        bound_no_improvement_time=BOUND_NO_IMPROVEMENT,
        bound_warmup_time=BOUND_WARMUP,
        use_time_weighted_deviation=False,
    )


def test_deterministic_baseline_twdev():
    """sigma=0 -> keine stochastische Varianz, twdev."""
    cfg = _load_config()
    exp_id = _create_experiment(cfg, sigma=0.0)
    logger = Logger(name="test_det_twdev", log_file="test_det_twdev.log")
    run_experiment(
        experiment_id=exp_id,
        shift_length=int(cfg["run"]["shift_length"]),
        total_shift_number=SHIFT_COUNT,
        logger=logger,
        time_limit=TIME_LIMIT,
        bound_no_improvement_time=BOUND_NO_IMPROVEMENT,
        bound_warmup_time=BOUND_WARMUP,
        use_time_weighted_deviation=True,
        deviation_window_minutes=480,
        deviation_bucket_minutes=60,
        deviation_max_factor=8,
    )


def test_twdev_with_blockade():
    """twdev + Maschinenblockade kombiniert."""
    cfg = _load_config()
    exp_id = _create_experiment(cfg, sigma=0.1)
    logger = Logger(name="test_twdev_block", log_file="test_twdev_block.log")
    blockades = [{"machine": "M00", "start": 1500, "end": 1560}]
    run_experiment(
        experiment_id=exp_id,
        shift_length=int(cfg["run"]["shift_length"]),
        total_shift_number=SHIFT_COUNT,
        logger=logger,
        time_limit=TIME_LIMIT,
        bound_no_improvement_time=BOUND_NO_IMPROVEMENT,
        bound_warmup_time=BOUND_WARMUP,
        use_time_weighted_deviation=True,
        deviation_window_minutes=480,
        deviation_bucket_minutes=60,
        deviation_max_factor=8,
        machine_blockades=blockades,
    )


def main():
    print("="*70)
    print("SCHNELLTESTS: Fehlende Szenarien (je 2 Shifts)")
    print("="*70)

    results = []
    results.append(_safe_run(test_deterministic_baseline_std, "Deterministisch + Standard-Deviation"))
    results.append(_safe_run(test_deterministic_baseline_twdev, "Deterministisch + twdev"))
    results.append(_safe_run(test_twdev_with_blockade, "twdev + Maschinenblockade"))

    print("\n" + "="*70)
    passed = sum(results)
    total = len(results)
    print(f"Ergebnis: {passed}/{total} Tests bestanden")
    print("="*70)
    return 0 if all(results) else 1


if __name__ == "__main__":
    exit(main())
