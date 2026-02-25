# Minimalinvasives Job-Shop Scheduling


# Projektsetup

Dieses Projekt nutzt verschiedene Python-Bibliotheken für Datenanalyse, Simulation und Optimierung. Unten findest du die Anweisungen zur Installation der Abhängigkeiten – jeweils für **Windows** und **Unix-basierte Systeme** (Linux, macOS).

---

## Installation

### Voraussetzungen

- **Python 3.10 oder höher**
- **Aktuelle pip-Version**
- Optional: Verwendung einer **virtuellen Umgebung**

---

### Installation unter Windows

```cmd
:: Virtuelle Umgebung erstellen (optional, empfohlen)
python -m venv venv
venv\Scripts\activate
```

```cmd
:: pip aktualisieren
python -m pip install --upgrade pip
```

```cmd
:: Pakete installieren
pip install pandas matplotlib simpy pulp ortools editdistance scipy sqlalchemy colorama yagmail scikit-learn python-dotenv seaborn tomli
```
---

### Installation unter Linux / macOS

```bash
# Virtuelle Umgebung erstellen (optional, empfohlen)
python3 -m venv venv
source venv/bin/activate
```

```bash
# pip aktualisieren
python3 -m pip install --upgrade pip
```

```bash
# Pakete installieren
python3 -m pip install pandas matplotlib simpy pulp ortools editdistance scipy sqlalchemy colorama yagmail scikit-learn python-dotenv seaborn tomli
```

---

## Zeitgewichtetes Constraint-Programming im Rolling-Horizon-Framework

Dieses Repository enthält eine Constraint-Programming-Variante, die die **Start-Abweichung** (Deviation) zeitabhängig gewichtet:
Änderungen an Operationen, die **nah** an der aktuellen Schicht liegen, werden **teurer** als Änderungen weit in der Zukunft.

### Integration: twdev im Rolling-Horizon mit Störszenarien

Die zeitgewichtete Deviation (twdev) wurde in das bestehende Rolling-Horizon-Framework integriert und kann mit verschiedenen Störszenarien getestet werden:
- **Stochastische Varianz (Sigma)** - Lognormal-verteilte Störungen der Operationsdauern
- **Maschinenblockaden** - Deterministische Maschinenausfälle
- **Kombinierte Störungen** - Beide Szenarien gleichzeitig

#### 1) Daten/Datenbank vorbereiten (einmalig)

```bash
python3 00_Problem_Generation/all.py
```

Hinweis: dabei werden Tabellen zurückgesetzt und der Datensatz `Fisher and Thompson 10x10` (Routings, Jobs, Due Dates, Transition Times) in `experiments.db` aufgebaut.

#### 2) Rolling-Horizon-Experimente ausführen

**Standard-Deviation (Original-Framework):**
```bash
python3 run_cp_experiments.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60
```

**Time-Weighted-Deviation (twdev - Neue Methode):**
```bash
python3 run_cp_experiments.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 --use_time_weighted_deviation --deviation_window_minutes 480 --deviation_bucket_minutes 60
```

**Mit Maschinenblockade:**
```bash
python3 run_cp_experiments.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 --machine_blockade M00:1500:1560
```

**Direkter Vergleich (Standard vs. twdev):**
```bash
python3 run_cp_twdev_comparison.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60
```

**Vergleich mit Maschinenblockade:**
```bash
python3 run_cp_twdev_comparison.py --util 0.75 --sigma 0.1 --time_limit 1800 --bound_no_improvement_time 600 --bound_warmup_time 60 --machine_blockade M00:1500:1560
```

Ausgaben:
- **Datenbank**: Schedule/Simulation werden in `experiments.db` gespeichert
- **Logs**: Solver-Logs für jede Schicht in `data/solver_logs/`
- **Vergleichsergebnisse**: JSON-Datei mit Shift-Summaries in `data/output/twdev_comparison/`

#### 3) Finale Experimente (alle Szenarien)

```bash
./run_all_final_experiments.sh
```

Führt 12 systematische Experimente durch:
- Szenario 1: Nur stochastische Varianz (3 Tests)
- Szenario 2: Nur Maschinenblockade (3 Tests)
- Szenario 3: Kombinierte Störungen (4 Tests)
- Szenario 4: Extreme Bedingungen (2 Tests)

#### 4) Ergebnisse analysieren

```bash
python3 analyze_final_results.py
```

Erstellt:
- Zusammenfassungstabelle aller Experimente
- Übersichts-Visualisierungen
- CSV-Export für weitere Analysen

---

## Neue und geänderte Dateien (Projektarbeit)

### Neue Dateien

| Datei | Beschreibung |
|-------|--------------|
| **`run_cp_twdev_comparison.py`** | Vergleichsskript: Standard-Deviation vs. Time-Weighted-Deviation im Rolling-Horizon-Framework |
| **`run_all_final_experiments.sh`** | Batch-Skript: 12 systematische Experimente mit allen Störszenarien |
| **`analyze_final_results.py`** | Analyse-Skript: Zusammenfassung und Visualisierung aller Experimente |
| **`test_twdev_integration.py`** | Schnelltest: Überprüft twdev-Integration (2 Shifts) |
| **`test_machine_blockade.py`** | Test: Überprüft Maschinenblockaden-Funktionalität |
| **`test_alpha_weights.py`** | Unit-Test: Prüft die acht Alpha-Gewichtungsfaktoren |
| **`test_quick_scenarios.py`** | Test: Deterministisches Baseline und twdev+Blockade |

---

### Geänderte Dateien mit Zeilenreferenzen

#### `src/solvers/CP_Solver.py`

| Was wurde hinzugefügt? |
|------------------------|
| Neue Methode `build_model__absolute_lateness__time_weighted_start_deviation__minimization()` – kombiniert Verspätung + Verfrühung + **zeitgewichtete Abweichung** |
| Neue Hilfsmethode `_add_time_weighted_start_deviation_var()` – berechnet Gewicht basierend auf zeitlicher Nähe zur Schicht |
| Parameter `machine_blockades` im `__init__` – ermöglicht deterministische Maschinenausfälle |
| Maschinenblockaden-Constraints in `_add_machine_no_overlap_constraints()` – blockiert Maschinen für definierte Zeiträume |

#### `src/CP_Experiment_Runner.py`

| Was wurde hinzugefügt? |
|------------------------|
| Neue Parameter für `use_time_weighted_deviation`, `deviation_window_minutes`, `deviation_bucket_minutes`, `deviation_max_factor`, `machine_blockades` |
| Bedingte Solver-Auswahl: twdev vs. Standard-Deviation basierend auf `use_time_weighted_deviation` |
| Automatische Erkennung aktiver Blockaden pro Shift |

#### `run_cp_experiments.py`

| Was wurde hinzugefügt? |
|------------------------|
| CLI-Argumente für twdev-Parameter (`--use_time_weighted_deviation`, `--deviation_window_minutes`, etc.) |
| CLI-Argument für Maschinenblockaden (`--machine_blockade`, mehrfach verwendbar) |
| Übergabe der twdev- und Blockaden-Parameter an `run_experiment()` |

#### `src/solvers/CP_Collections.py`

| Was wurde hinzugefügt? |
|------------------------|
| Neue Klasse `WeightedCostVarCollection` – erlaubt individuelle Gewichte pro Variable (für zeitgewichtete Abweichung) |

#### `src/DataFrameAnalyses.py`

| Zeile | Änderung |
|-------|----------|
| **1** | `from __future__ import annotations` hinzugefügt (Python 3.9 Kompatibilität für `list[...]` Type Hints) |

#### `src/analyses/fig_startdeviation.py`

| Zeile | Änderung |
|-------|----------|
| **1** | `from __future__ import annotations` hinzugefügt |

#### `src/analyses/fig_tardiness_earliness.py`

| Zeile | Änderung |
|-------|----------|
| **1** | `from __future__ import annotations` hinzugefügt |

#### `README.md`

| Abschnitt | Änderung |
|-----------|----------|
| Ende | Dokumentation für eigenen Solver + Dateiübersicht hinzugefügt |

---

## Kernkonzepte der Implementierung

### 1. Makespan-Minimierung (Baseline)
```
Zielfunktion: minimiere(makespan)
              wobei makespan = max(Endzeit_i) über alle Operationen
```

### 2. Neuplanung mit Kostentermen
```
Zielfunktion: minimiere(
    Gewicht_Verspätung × Σ Verspätung_j        # Tardiness
  + Gewicht_Verfrühung × Σ Verfrühung_j        # Earliness  
  + Gewicht_Abweichung × Σ Zeitgewicht_i × |Startzeit_i - Startzeit_i_alt|   # Deviation
)
```

### 3. Zeitgewichtete Abweichung (time-weighted deviation)
Operationen, die näher am aktuellen Zeitpunkt liegen, bekommen ein **höheres Gewicht**:
```
Bucket 0 (0–60 Minuten):   Gewicht = Maximalfaktor
Bucket 1 (60–120 Minuten): Gewicht = Maximalfaktor - 1
...
Bucket n (> Zeitfenster):  Gewicht = 1
```

---

## Tests und Verifikation

### Schnelltests

**twdev-Integration testen (2 Shifts):**
```bash
EMAIL_TO="test@example.com" SMTP_USER="test@example.com" SMTP_PASS="dummy" \
  python3 test_twdev_integration.py
```

**Maschinenblockade testen:**
```bash
EMAIL_TO="test@example.com" SMTP_USER="test@example.com" SMTP_PASS="dummy" \
  python3 test_machine_blockade.py
```

---

## Experimentelle Auswertung

### Batch-Experimente durchführen

**Alle finalen Experimente (12 Tests, ~24h Laufzeit):**
```bash
./run_all_final_experiments.sh
```

### Ergebnisse analysieren

**Zusammenfassung erstellen:**
```bash
python3 analyze_final_results.py
```

### Erzeugte Ausgaben

Nach dem Durchlauf werden folgende Dateien erstellt:

- **`experiments_overview.png`** - Übersicht aller durchgeführten Experimente
- **`experiments_summary.csv`** - Tabellarische Zusammenfassung
- Detaillierte Metriken in der Datenbank (`experiments.db`)

---

## Dokumentation

- **`README.md`** - Diese Datei (Projektübersicht, Setup, Nutzung)
