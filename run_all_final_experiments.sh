#!/bin/bash

# ============================================================================
# FINALE EXPERIMENTE: Standard-Deviation vs. Time-Weighted-Deviation
# mit verschiedenen Störszenarien (Varianz + Maschinenblockaden)
# ============================================================================

set -e  # Bei Fehler abbrechen

# Aktiviere Virtual Environment
echo "Aktiviere Virtual Environment..."
source .venv/bin/activate

# E-Mail-Dummy-Config
export EMAIL_TO="test@example.com"
export SMTP_USER="test@example.com"
export SMTP_PASS="dummy"

# Output-Verzeichnis
OUTPUT_BASE="data/output/final_experiments"
mkdir -p "$OUTPUT_BASE"

# Logging
LOG_FILE="$OUTPUT_BASE/batch_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================================"
echo "FINALE EXPERIMENTE STARTEN"
echo "============================================================================"
echo "Start: $(date)"
echo "Log-Datei: $LOG_FILE"
echo "============================================================================"
echo ""

# ============================================================================
# SZENARIO 1: Nur stochastische Varianz (keine Blockade)
# ============================================================================

echo ""
echo "========================================================================"
echo "SZENARIO 1: Nur stochastische Varianz"
echo "========================================================================"
echo ""

# Test 1.1: Niedrige Auslastung, niedrige Varianz
echo "[1/12] Test 1.1: util=0.65, sigma=0.05, keine Blockade"
python3 run_cp_twdev_comparison.py \
  --util 0.65 --sigma 0.05 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --output_dir "$OUTPUT_BASE/test_01_util065_sig005_noblock"

# Test 1.2: Mittlere Auslastung, mittlere Varianz
echo "[2/12] Test 1.2: util=0.75, sigma=0.1, keine Blockade"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --output_dir "$OUTPUT_BASE/test_02_util075_sig010_noblock"

# Test 1.3: Hohe Auslastung, hohe Varianz
echo "[3/12] Test 1.3: util=0.85, sigma=0.15, keine Blockade"
python3 run_cp_twdev_comparison.py \
  --util 0.85 --sigma 0.15 \
  --time_limit 3600 --bound_no_improvement_time 900 --bound_warmup_time 120 \
  --output_dir "$OUTPUT_BASE/test_03_util085_sig015_noblock"

# ============================================================================
# SZENARIO 2: Nur Maschinenblockade (keine Varianz)
# ============================================================================

echo ""
echo "========================================================================"
echo "SZENARIO 2: Nur Maschinenblockade"
echo "========================================================================"
echo ""

# Test 2.1: Niedrige Auslastung, M00 blockiert
echo "[4/12] Test 2.1: util=0.65, sigma=0.05, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.65 --sigma 0.05 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_04_util065_sig005_M00block"

# Test 2.2: Mittlere Auslastung, M00 blockiert
echo "[5/12] Test 2.2: util=0.75, sigma=0.1, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_05_util075_sig010_M00block"

# Test 2.3: Mittlere Auslastung, M03 blockiert
echo "[6/12] Test 2.3: util=0.75, sigma=0.1, M03 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --machine_blockade M03:2000:2100 \
  --output_dir "$OUTPUT_BASE/test_06_util075_sig010_M03block"

# ============================================================================
# SZENARIO 3: Kombinierte Störungen (Varianz + Blockade)
# ============================================================================

echo ""
echo "========================================================================"
echo "SZENARIO 3: Kombinierte Störungen (Varianz + Blockade)"
echo "========================================================================"
echo ""

# Test 3.1: Niedrige Auslastung, niedrige Varianz + M00 blockiert
echo "[7/12] Test 3.1: util=0.65, sigma=0.05, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.65 --sigma 0.05 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_07_util065_sig005_M00block_combined"

# Test 3.2: Mittlere Auslastung, mittlere Varianz + M00 blockiert
echo "[8/12] Test 3.2: util=0.75, sigma=0.1, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_08_util075_sig010_M00block_combined"

# Test 3.3: Mittlere Auslastung, mittlere Varianz + mehrere Blockaden
echo "[9/12] Test 3.3: util=0.75, sigma=0.1, M00+M03 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.1 \
  --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 \
  --machine_blockade M00:1500:1560 --machine_blockade M03:2000:2100 \
  --output_dir "$OUTPUT_BASE/test_09_util075_sig010_M00M03block"

# Test 3.4: Hohe Auslastung, hohe Varianz + M00 blockiert
echo "[10/12] Test 3.4: util=0.85, sigma=0.15, M00 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.85 --sigma 0.15 \
  --time_limit 3600 --bound_no_improvement_time 900 --bound_warmup_time 120 \
  --machine_blockade M00:1500:1560 \
  --output_dir "$OUTPUT_BASE/test_10_util085_sig015_M00block"

# ============================================================================
# SZENARIO 4: Extreme Bedingungen
# ============================================================================

echo ""
echo "========================================================================"
echo "SZENARIO 4: Extreme Bedingungen"
echo "========================================================================"
echo ""

# Test 4.1: Hohe Auslastung, hohe Varianz + längere Blockade
echo "[11/12] Test 4.1: util=0.85, sigma=0.15, M00 lange blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.85 --sigma 0.15 \
  --time_limit 3600 --bound_no_improvement_time 900 --bound_warmup_time 120 \
  --machine_blockade M00:1500:1700 \
  --output_dir "$OUTPUT_BASE/test_11_util085_sig015_M00longblock"

# Test 4.2: Mittlere Auslastung, hohe Varianz + mehrere Blockaden
echo "[12/12] Test 4.2: util=0.75, sigma=0.15, M00+M03+M05 blockiert"
python3 run_cp_twdev_comparison.py \
  --util 0.75 --sigma 0.15 \
  --time_limit 3600 --bound_no_improvement_time 900 --bound_warmup_time 120 \
  --machine_blockade M00:1500:1560 --machine_blockade M03:2000:2100 --machine_blockade M05:2500:2600 \
  --output_dir "$OUTPUT_BASE/test_12_util075_sig015_M00M03M05block"

# ============================================================================
# ABSCHLUSS
# ============================================================================

echo ""
echo "============================================================================"
echo "ALLE EXPERIMENTE ABGESCHLOSSEN"
echo "============================================================================"
echo "Ende: $(date)"
echo "Ergebnisse in: $OUTPUT_BASE"
echo "Log-Datei: $LOG_FILE"
echo "============================================================================"
echo ""



