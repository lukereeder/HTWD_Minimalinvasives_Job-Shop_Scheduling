#!/bin/bash

# ============================================================================
# SCHNELLE EXPERIMENTE: Nur die wichtigsten 3 Szenarien
# Laufzeit: ~2-3 Stunden statt 24 Stunden
# ============================================================================

set -e

echo "Aktiviere Virtual Environment..."
source .venv/bin/activate

export EMAIL_TO="test@example.com"
export SMTP_USER="test@example.com"
export SMTP_PASS="dummy"

OUTPUT_BASE="data/output/quick_experiments"
mkdir -p "$OUTPUT_BASE"

LOG_FILE="$OUTPUT_BASE/quick_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================================"
echo "SCHNELLE EXPERIMENTE (3 Szenarien, kürzere Zeitlimits)"
echo "============================================================================"
echo "Start: $(date)"
echo "Geschätzte Laufzeit: 2-3 Stunden"
echo "============================================================================"
echo ""

# Kürzere Zeitlimits für schnellere Durchläufe
TIME_LIMIT=600        # 10 Minuten statt 30
NO_IMPROVEMENT=180    # 3 Minuten statt 10
WARMUP=30            # 30 Sekunden

# ============================================================================
# Test 1: Mittlere Auslastung, mittlere Varianz, keine Blockade
# ============================================================================
echo "[1/3] Test 1: util=0.75, sigma=0.1, keine Blockade"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit $TIME_LIMIT \
  --bound_no_improvement_time $NO_IMPROVEMENT \
  --bound_warmup_time $WARMUP \
  --output_dir "$OUTPUT_BASE/test_01_baseline"

# ============================================================================
# Test 2: Mittlere Auslastung, mittlere Varianz, mit Blockade
# ============================================================================
echo "[2/3] Test 2: util=0.75, sigma=0.1, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit $TIME_LIMIT \
  --bound_no_improvement_time $NO_IMPROVEMENT \
  --bound_warmup_time $WARMUP \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_02_blockade"

# ============================================================================
# Test 3: Hohe Auslastung, hohe Varianz, mit Blockade
# ============================================================================
echo "[3/3] Test 3: util=0.85, sigma=0.15, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.85 --sigma 0.15 \
  --time_limit $TIME_LIMIT \
  --bound_no_improvement_time $NO_IMPROVEMENT \
  --bound_warmup_time $WARMUP \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_03_high_load"

echo ""
echo "============================================================================"
echo "SCHNELLE EXPERIMENTE ABGESCHLOSSEN"
echo "============================================================================"
echo "Ende: $(date)"
echo "Ergebnisse in: $OUTPUT_BASE"
echo "Log-Datei: $LOG_FILE"
echo "============================================================================"
echo ""
echo "Nächste Schritte:"
echo "python3 analyze_final_results.py  # Ergebnisse ansehen"
echo "============================================================================"



