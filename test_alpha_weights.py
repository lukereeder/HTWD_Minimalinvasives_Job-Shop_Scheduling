#!/usr/bin/env python3
"""
Unit-Tests fuer die Alpha-Gewichtungsfunktion (_near_future_weight)
des CP-Solvers. Prueft 8-Bucket-System mit Faktoren 8..1.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from solvers.CP_Solver import Solver as CP_Solver


def test_alpha_eight_buckets():
    """window=480, bucket=60 -> 8 Buckets, Faktoren 8 down to 1."""
    w = CP_Solver._near_future_weight
    cases = [
        (0,   8), (30,  8), (59,  8),   # Bucket 0
        (60,  7), (90,  7), (119, 7),    # Bucket 1
        (120, 6),                         # Bucket 2
        (180, 5),                         # Bucket 3
        (240, 4),                         # Bucket 4
        (300, 3),                         # Bucket 5
        (360, 2),                         # Bucket 6
        (420, 1), (479, 1),               # Bucket 7
        (480, 1), (600, 1), (1000, 1),    # Beyond window
    ]
    for offset, expected in cases:
        actual = w(
            original_start=100 + offset,
            schedule_start=100,
            window_minutes=480,
            bucket_minutes=60,
        )
        assert actual == expected, (
            f"offset={offset}: expected factor {expected}, got {actual}"
        )
    print("  PASS test_alpha_eight_buckets")


def test_alpha_past_operations():
    """Operationen vor schedule_start -> dt=0 -> hoechster Faktor."""
    w = CP_Solver._near_future_weight
    for past_start in [0, 50, 99]:
        actual = w(
            original_start=past_start,
            schedule_start=100,
            window_minutes=480,
            bucket_minutes=60,
        )
        assert actual == 8, f"past start={past_start}: expected 8, got {actual}"
    print("  PASS test_alpha_past_operations")


def test_alpha_window_zero():
    """window=0 -> immer Faktor 1."""
    actual = CP_Solver._near_future_weight(
        original_start=100,
        schedule_start=100,
        window_minutes=0,
        bucket_minutes=60,
    )
    assert actual == 1, f"window=0: expected 1, got {actual}"
    print("  PASS test_alpha_window_zero")


def test_alpha_max_factor():
    """max_factor begrenzt den Rueckgabewert."""
    actual = CP_Solver._near_future_weight(
        original_start=100,
        schedule_start=100,
        window_minutes=480,
        bucket_minutes=60,
        max_factor=5,
    )
    assert actual == 5, f"max_factor=5: expected 5, got {actual}"

    actual2 = CP_Solver._near_future_weight(
        original_start=400,
        schedule_start=100,
        window_minutes=480,
        bucket_minutes=60,
        max_factor=5,
    )
    assert actual2 == 3, f"offset=300, max_factor=5: expected 3, got {actual2}"
    print("  PASS test_alpha_max_factor")


def test_alpha_formula():
    """Pruefe die Formel: factor = 1 + (steps - 1 - bucket_idx)."""
    w = CP_Solver._near_future_weight
    steps = 8  # 480/60
    for bucket_idx in range(steps):
        offset = bucket_idx * 60
        expected = 1 + (steps - 1 - bucket_idx)
        actual = w(
            original_start=offset,
            schedule_start=0,
            window_minutes=480,
            bucket_minutes=60,
        )
        assert actual == expected, (
            f"bucket_idx={bucket_idx}: expected {expected}, got {actual}"
        )
    print("  PASS test_alpha_formula")


if __name__ == "__main__":
    print("Alpha-Gewichtung Unit-Tests:")
    test_alpha_eight_buckets()
    test_alpha_past_operations()
    test_alpha_window_zero()
    test_alpha_max_factor()
    test_alpha_formula()
    print("\nAlle Alpha-Tests BESTANDEN")
