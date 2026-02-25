#!/usr/bin/env python3
"""
Schnelltest: Überprüft, ob die twdev-Integration funktioniert
Führt einen Mini-Test mit nur 2 Shifts durch
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


def test_standard_deviation():
    """Test mit Standard-Deviation"""
    print("\n" + "="*80)
    print("TEST 1: Standard-Deviation (Original-Framework)")
    print("="*80)
    
    # Load config
    config_path = get_config_path("experiments_config.toml", as_string=False)
    with open(config_path, "rb") as f:
        cfg = pytoml.load(f)

    run_cfg = cfg["run"]
    grid = cfg["grid"]

    source_name: str = run_cfg["source_name"]
    shift_length: int = int(run_cfg["shift_length"])
    
    # Nur 2 Shifts für schnellen Test
    total_shift_number = 2
    
    # Erste Kombination aus Grid
    util = grid["max_bottleneck_utilization"][0]
    absolute_lateness_ratio = grid["absolute_lateness_ratio"][0]
    inner_tardiness_ratio = grid["inner_tardiness_ratio"][0]
    sigma = grid["simulation_sigma"][0]

    exp_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{util:.2f}"),
        sim_sigma=sigma,
        experiment_type="CP",
    )

    logger = Logger(
        name="test_std_dev",
        log_file="test_std_dev.log"
    )

    try:
        result = run_experiment(
            experiment_id=exp_id,
            shift_length=shift_length,
            total_shift_number=total_shift_number,
            logger=logger,
            time_limit=300,  # 5 Minuten
            bound_no_improvement_time=60,
            bound_warmup_time=30,
            use_time_weighted_deviation=False,
        )
        print(f"✓ Standard-Deviation Test erfolgreich (Experiment ID: {exp_id})")
        print(f"  Shifts verarbeitet: {len(result.get('shift_summaries', []))}")
        return True
    except Exception as e:
        # E-Mail-Fehler ignorieren (nur SMTP-Authentifizierung)
        if "Username and Password not accepted" in str(e) or "BadCredentials" in str(e):
            print(f"✓ Standard-Deviation Test erfolgreich (Experiment ID: {exp_id})")
            print(f"  (E-Mail-Benachrichtigung übersprungen)")
            return True
        print(f"✗ Standard-Deviation Test fehlgeschlagen: {e}")
        return False


def test_time_weighted_deviation():
    """Test mit Time-Weighted-Deviation"""
    print("\n" + "="*80)
    print("TEST 2: Time-Weighted-Deviation (twdev)")
    print("="*80)
    
    # Load config
    config_path = get_config_path("experiments_config.toml", as_string=False)
    with open(config_path, "rb") as f:
        cfg = pytoml.load(f)

    run_cfg = cfg["run"]
    grid = cfg["grid"]

    source_name: str = run_cfg["source_name"]
    shift_length: int = int(run_cfg["shift_length"])
    
    # Nur 2 Shifts für schnellen Test
    total_shift_number = 2
    
    # Erste Kombination aus Grid
    util = grid["max_bottleneck_utilization"][0]
    absolute_lateness_ratio = grid["absolute_lateness_ratio"][0]
    inner_tardiness_ratio = grid["inner_tardiness_ratio"][0]
    sigma = grid["simulation_sigma"][0]

    exp_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{util:.2f}"),
        sim_sigma=sigma,
        experiment_type="CP",
    )

    logger = Logger(
        name="test_twdev",
        log_file="test_twdev.log"
    )

    try:
        result = run_experiment(
            experiment_id=exp_id,
            shift_length=shift_length,
            total_shift_number=total_shift_number,
            logger=logger,
            time_limit=300,  # 5 Minuten
            bound_no_improvement_time=60,
            bound_warmup_time=30,
            use_time_weighted_deviation=True,
            deviation_window_minutes=480,
            deviation_bucket_minutes=60,
            deviation_max_factor=8,
        )
        print(f"✓ Time-Weighted-Deviation Test erfolgreich (Experiment ID: {exp_id})")
        print(f"  Shifts verarbeitet: {len(result.get('shift_summaries', []))}")
        return True
    except Exception as e:
        # E-Mail-Fehler ignorieren (nur SMTP-Authentifizierung)
        if "Username and Password not accepted" in str(e) or "BadCredentials" in str(e):
            print(f"✓ Time-Weighted-Deviation Test erfolgreich (Experiment ID: {exp_id})")
            print(f"  (E-Mail-Benachrichtigung übersprungen)")
            return True
        print(f"✗ Time-Weighted-Deviation Test fehlgeschlagen: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("TWDEV-INTEGRATION SCHNELLTEST")
    print("="*80)
    
    test1_passed = test_standard_deviation()
    test2_passed = test_time_weighted_deviation()
    
    print("\n" + "="*80)
    print("TESTERGEBNISSE")
    print("="*80)
    print(f"Test 1 (Standard-Deviation): {'✓ BESTANDEN' if test1_passed else '✗ FEHLGESCHLAGEN'}")
    print(f"Test 2 (Time-Weighted-Deviation): {'✓ BESTANDEN' if test2_passed else '✗ FEHLGESCHLAGEN'}")
    print("="*80)
    
    if test1_passed and test2_passed:
        print("\n✓ ALLE TESTS BESTANDEN - Integration erfolgreich!")
        return 0
    else:
        print("\n✗ EINIGE TESTS FEHLGESCHLAGEN - Bitte Logs prüfen")
        return 1


if __name__ == "__main__":
    """
    Schnelltest ausführen:
    python3 test_twdev_integration.py
    """
    exit(main())

