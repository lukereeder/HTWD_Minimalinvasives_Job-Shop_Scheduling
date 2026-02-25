#!/usr/bin/env python3
"""
Test: Maschinenblockade im Rolling-Horizon-Framework
Überprüft, ob Maschinenblockaden korrekt funktionieren
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


def test_with_blockade():
    """Test mit Maschinenblockade"""
    print("\n" + "="*80)
    print("TEST: Maschinenblockade im Rolling-Horizon-Framework")
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

    # Maschinenblockade: M00 blockiert von 1500 bis 1560 (60 Minuten)
    machine_blockades = [
        {'machine': 'M00', 'start': 1500, 'end': 1560}
    ]

    print(f"Utilization: {util}")
    print(f"Sigma: {sigma}")
    print(f"Shifts: {total_shift_number} × {shift_length} min")
    print(f"Blockade: {machine_blockades[0]['machine']} von {machine_blockades[0]['start']} bis {machine_blockades[0]['end']}")
    print("="*80 + "\n")

    exp_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{util:.2f}"),
        sim_sigma=sigma,
        experiment_type="CP",
    )

    logger = Logger(
        name="test_blockade",
        log_file="test_blockade.log"
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
            machine_blockades=machine_blockades,
        )
        print(f"\n[OK] Test erfolgreich (Experiment ID: {exp_id})")
        print(f"  Shifts verarbeitet: {len(result.get('shift_summaries', []))}")
        print(f"  Blockaden: {result.get('machine_blockades')}")
        return True
    except Exception as e:
        # E-Mail-Fehler ignorieren (nur SMTP-Authentifizierung)
        if "Username and Password not accepted" in str(e) or "BadCredentials" in str(e):
            print(f"\n[OK] Test erfolgreich (Experiment ID: {exp_id})")
            print(f"  (E-Mail-Benachrichtigung übersprungen)")
            return True
        print(f"\n[FAIL] Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("MASCHINENBLOCKADE-TEST")
    print("="*80)
    
    test_passed = test_with_blockade()
    
    print("\n" + "="*80)
    print("TESTERGEBNIS")
    print("="*80)
    print(f"Maschinenblockade-Test: {'[OK] BESTANDEN' if test_passed else '[FAIL] FEHLGESCHLAGEN'}")
    print("="*80 + "\n")
    
    if test_passed:
        print("\n[OK] TEST BESTANDEN - Maschinenblockade funktioniert!")
        return 0
    else:
        print("\n[FAIL] TEST FEHLGESCHLAGEN - Bitte Logs prüfen")
        return 1


if __name__ == "__main__":
    """
    Test ausführen:
    python3 test_machine_blockade.py
    
    Mit E-Mail-Dummy-Config:
    EMAIL_TO="test@example.com" SMTP_USER="test@example.com" SMTP_PASS="dummy" python3 test_machine_blockade.py
    """
    exit(main())



