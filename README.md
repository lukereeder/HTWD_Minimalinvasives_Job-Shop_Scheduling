# Minimalinvasives Job-Shop Scheduling


# 🧮 Projektsetup

Dieses Projekt nutzt verschiedene Python-Bibliotheken für Datenanalyse, Simulation und Optimierung. Unten findest du die Anweisungen zur Installation der Abhängigkeiten – jeweils für **Windows** und **Unix-basierte Systeme** (Linux, macOS).

---

## 🛠️ Installation

### 🔹 Voraussetzungen

- **Python 3.10 oder höher**
- **Aktuelle pip-Version**
- Optional: Verwendung einer **virtuellen Umgebung**

---

### 🪟 Installation unter Windows

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

### 🐧 Installation unter Linux / macOS

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

## ▶️ Zeitgewichtetes Constraint-Programming im Rolling-Horizon-Framework

Dieses Repository enthält eine Constraint-Programming-Variante, die die **Start-Abweichung** (Deviation) zeitabhängig gewichtet:
Änderungen an Operationen, die **nah** an der aktuellen Schicht liegen, werden **teurer** als Änderungen weit in der Zukunft.

### 🔄 Integration: twdev im Rolling-Horizon mit Störszenarien

Die zeitgewichtete Deviation (twdev) wurde in das bestehende Rolling-Horizon-Framework integriert und kann mit verschiedenen Störszenarien getestet werden:
- ✅ **Stochastische Varianz (Sigma)** - Lognormal-verteilte Störungen der Operationsdauern
- ✅ **Maschinenblockaden** - Deterministische Maschinenausfälle
- ✅ **Kombinierte Störungen** - Beide Szenarien gleichzeitig

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

## 📂 Neue und geänderte Dateien (Projektarbeit)

### 🆕 Neue Dateien

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

### ✏️ Geänderte Dateien mit Zeilenreferenzen

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

## 🔑 Kernkonzepte der Implementierung

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

## 🧪 Tests und Verifikation

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

## 📊 Experimentelle Auswertung

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

## 📝 Dokumentation

- **`README.md`** - Diese Datei (Projektübersicht, Setup, Nutzung)

---

## 🔑 Kernkonzepte der Implementierung (VERALTET - siehe ABSCHLUSSBERICHT.md)

### Alte Batch-Experiment-Diagramme (nicht mehr relevant)

Die folgenden Abschnitte beschreiben die alten isolierten Batch-Experimente.
Diese wurden durch die Rolling-Horizon-Integration ersetzt.

<details>
<summary>Klicken zum Anzeigen der alten Diagramm-Beschreibungen</summary>

### 📈 Diagramm 1: `batch_scatter_makespan_vs_tardiness.png` (VERALTET)

**Was zeigt dieses Diagramm?**

Ein Punktdiagramm (Scatter-Plot), das für jede der 100 Konfigurationen die **Gesamtdurchlaufzeit (Makespan)** auf der X-Achse gegen die **Verspätungskosten (Tardiness)** auf der Y-Achse aufträgt.

**Wie liest man dieses Diagramm?**

- **Grüne gestrichelte Linie bei 930:** Der optimale Baseline-Makespan ohne Störung. Das ist der bestmögliche Wert, wenn keine Maschine ausfällt.
- **Rote Punkte:** Konfigurationen mit zeitgewichteter Abweichung (`dev_mode=twdev`). Hier werden Änderungen an nahen Operationen stärker bestraft.
- **Türkise Punkte:** Konfigurationen mit ungewichteter Abweichung (`dev_mode=dev`). Hier werden alle Operationen gleich behandelt.
- **Punktgröße:** Je größer der Punkt, desto höher das Verspätungs-Gewicht (w_t). Große Punkte bedeuten, dass Pünktlichkeit in dieser Konfiguration hoch priorisiert wurde.
- **Goldener Stern:** Die beste ausgewogene Konfiguration, die sowohl niedrigen Makespan als auch niedrige Verspätung erreicht.

**Konkrete Werte aus unseren Experimenten:**

| Markierung | Konfiguration | Makespan (Gesamtdurchlaufzeit) | Verspätungskosten | Bedeutung |
|------------|---------------|--------------------------------|-------------------|-----------|
| Minimaler Makespan | #94 | **1136 Zeiteinheiten** | 11555 | Kürzeste Gesamtdurchlaufzeit aller Konfigurationen |
| Minimale Verspätung | #57 | 1362 Zeiteinheiten | **2223** | Pünktlichste Lösung mit niedrigster Verspätung |
| Beste Balance | #70 | 1185 Zeiteinheiten | 2234 | Optimaler Kompromiss zwischen beiden Zielen |

**Was bedeutet das?**

- Alle Punkte liegen **rechts** der grünen Linie → jede Störung (Maschinenblockade) erhöht die Gesamtdurchlaufzeit gegenüber dem Optimum
- Es gibt einen **Trade-off (Zielkonflikt)**: Eine niedrige Gesamtdurchlaufzeit geht oft mit höherer Verspätung einher und umgekehrt
- Die beste Balance liegt bei einer Gesamtdurchlaufzeit von circa 1185 Zeiteinheiten, was **+27% über dem Baseline** von 930 liegt

---

### 📈 Diagramm 2: `batch_scatter_makespan_vs_deviation.png`

**Was zeigt dieses Diagramm?**

Die Gesamtdurchlaufzeit (Makespan) auf der X-Achse gegen die **Abweichungskosten (Deviation)** auf der Y-Achse – also wie stark sich der neue Plan gegenüber dem ursprünglichen Baseline-Plan geändert hat.

**Wie liest man dieses Diagramm?**

- **Punktgröße:** Je größer der Punkt, desto höher das Abweichungs-Gewicht (w_dev). Große Punkte bedeuten, dass Planstabilität in dieser Konfiguration hoch priorisiert wurde.
- Punkte **links unten** sind ideal: Das bedeutet wenig Verlust bei der Gesamtdurchlaufzeit UND wenig Planänderung gegenüber dem Original.

**Konkrete Werte:**

| Konfiguration | Makespan (Gesamtdurchlaufzeit) | Abweichungskosten | Interpretation |
|---------------|--------------------------------|-------------------|----------------|
| #1 | 1271 Zeiteinheiten | 22.087 | Minimale Planänderung gegenüber Baseline |
| #91 | 1269 Zeiteinheiten | 560.070 | Maximale Planänderung gegenüber Baseline |
| #70 | 1185 Zeiteinheiten | 86.528 | Guter Kompromiss |

**Was bedeutet das?**

- Höhere Abweichungs-Gewichte (w_dev) führen zu **höheren gewichteten Kosten**, aber die **absolute Planänderung** (also wie viele Operationen tatsächlich verschoben wurden) bleibt ähnlich
- Die Abweichungskosten variieren von circa 22.000 bis circa 560.000 – das ist ein **Faktor von 25**! Das liegt daran, dass bei höheren Gewichten jede kleine Änderung viel teurer bewertet wird.

---

### 📈 Diagramm 3: `batch_scatter_tardiness_vs_deviation.png`

**Was zeigt dieses Diagramm?**

Den **Zielkonflikt (Trade-off)** zwischen Pünktlichkeit (Verspätungskosten) und Planstabilität (Abweichungskosten).

**Wie liest man dieses Diagramm?**

- **Farbskala (rechts):** 
  - Rote Punkte = Verspätungs-fokussiert (Verhältnis Verspätungs-Gewicht zu Abweichungs-Gewicht größer als 1)
  - Grüne Punkte = Abweichungs-fokussiert (Verhältnis kleiner als 1)
- **Schwarze Rauten:** Die sogenannten **Pareto-optimalen** Konfigurationen. Das sind Konfigurationen, bei denen keine andere Konfiguration in **beiden** Dimensionen (Verspätung UND Abweichung) besser ist.
- **Gestrichelte Linie:** Die approximierte **Pareto-Front** – sie zeigt die Grenze des Machbaren.

**Konkrete Pareto-optimale Konfigurationen:**

| Konfiguration | Verspätungs-Gewicht | Abweichungs-Gewicht | Verspätungskosten | Abweichungskosten | Warum ist diese Konfiguration Pareto-optimal? |
|---------------|---------------------|---------------------|-------------------|-------------------|-----------------------------------------------|
| #1 | 1 | 1 | 2.370 | 22.087 | Hat die niedrigsten Abweichungskosten aller Konfigurationen |
| #70 | 1 | 2 | 2.234 | 86.528 | Beste Balance zwischen beiden Zielen |
| #57 | 1 | 2 | 2.223 | 52.968 | Hat die niedrigsten Verspätungskosten aller Konfigurationen |

**Was bedeutet das?**

- Man kann **nicht** gleichzeitig beide Ziele minimieren → Die Pareto-Front zeigt den bestmöglichen Kompromiss
- Bewegung entlang der Front: Weniger Verspätung führt zu mehr Abweichung (und umgekehrt)
- Konfigurationen **unterhalb** der Pareto-Front sind nicht erreichbar

---

### 📈 Diagramm 4: `batch_histograms_overview.png`

**Was zeigt dieses Diagramm?**

Vier Histogramme, die die **Verteilung** aller 100 Ergebnisse zeigen. So sieht man, wie häufig bestimmte Wertebereiche vorkommen.

**Die vier Teildiagramme im Detail:**

**Oben links: Verteilung der Gesamtdurchlaufzeit (Makespan)**
- **Rote gestrichelte Linie:** Der Baseline-Wert (930 Zeiteinheiten) – das Optimum ohne Störung
- **Orange Linie:** Der Mittelwert aller 100 Experimente (1247 Zeiteinheiten)
- Die meisten Werte liegen zwischen 1150 und 1350 Zeiteinheiten
- **Konkrete Werte:** 
  - Minimum: 1136 Zeiteinheiten
  - Maximum: 1452 Zeiteinheiten
  - Mittelwert: **1247 Zeiteinheiten (das ist +34% gegenüber dem Baseline von 930)**

**Oben rechts: Verteilung der Verspätungskosten (Tardiness)**
- Die Verteilung ist **zweigipflig**: 
  - Viele niedrige Werte zwischen circa 2.000 und 5.000 (von Konfigurationen mit niedrigem Verspätungs-Gewicht)
  - Einige hohe Werte zwischen circa 20.000 und 25.000 (von Konfigurationen mit hohem Verspätungs-Gewicht)
- **Konkrete Werte:**
  - Minimum: 2.223
  - Maximum: 27.640
  - Mittelwert: **8.711**

**Unten links: Verteilung der Abweichungskosten (Deviation)**
- Die Verteilung ist **stark rechtsschief**: Die meisten Werte liegen unter 100.000, aber es gibt Ausreißer bis 700.000
- **Konkrete Werte:**
  - Minimum: 22.087
  - Maximum: 713.860
  - Mittelwert: **131.730**

**Unten rechts: Verteilung der Gesamtkosten (Objective)**
- Ähnlich wie die Abweichungskosten, da die Abweichung den größten Anteil an den Gesamtkosten ausmacht
- **Konkrete Werte:**
  - Minimum: 24.457
  - Maximum: 720.976
  - Mittelwert: **140.441**

---

### 📈 Diagramm 5: `batch_heatmaps_weights.png`

**Was zeigt dieses Diagramm?**

Drei Heatmaps (Wärmebilder), die den **Einfluss der Gewichte** auf die Ergebnisse zeigen.

**Die Achsen:**
- **Y-Achse:** Verspätungs-Gewicht (w_t) mit Werten 1, 2, 5 und 10
- **X-Achse:** Abweichungs-Gewicht (w_dev) mit Werten 1, 2, 3, 5 und 10

**Linke Heatmap: Durchschnittliche Gesamtdurchlaufzeit (Makespan)**
- **Farbskala:** Gelb = niedrige Werte (gut), Rot = hohe Werte (schlecht)
- **Wichtige Erkenntnis:** Die Gesamtdurchlaufzeit ist **weitgehend unabhängig** von den Gewichten!
- Alle Felder zeigen ähnliche Werte zwischen circa 1.200 und 1.300 Zeiteinheiten
- **Bedeutung:** Die Störung (Maschinenblockade) bestimmt die Gesamtdurchlaufzeit, nicht die Gewichte der Zielfunktion

**Mittlere Heatmap: Durchschnittliche Verspätungskosten (Tardiness)**
- Zeigt einen klaren Trend: **Je höher das Verspätungs-Gewicht (w_t), desto höher die gewichteten Verspätungskosten**
- Bei Verspätungs-Gewicht = 1: circa 2.000 bis 5.000
- Bei Verspätungs-Gewicht = 10: circa 20.000 bis 25.000
- **Bedeutung:** Das ist ein **Skalierungseffekt**, keine echte Verbesserung der Pünktlichkeit! Die absolute Verspätung in Minuten bleibt gleich, nur die gewichteten Kosten steigen.

**Rechte Heatmap: Durchschnittliche Abweichungskosten (Deviation)**
- Zeigt einen klaren Trend: **Je höher das Abweichungs-Gewicht (w_dev), desto höher die gewichteten Abweichungskosten**
- Bei Abweichungs-Gewicht = 1: circa 20.000 bis 50.000
- Bei Abweichungs-Gewicht = 10: circa 200.000 bis 500.000
- **Bedeutung:** Auch hier nur ein Skalierungseffekt, keine echte Reduktion der Planänderung!

---

### 📈 Diagramm 6: `batch_boxplots_parameters.png`

**Was zeigt dieses Diagramm?**

Sechs Box-Plots (Kastendiagramme), die den Einfluss **einzelner Parameter** auf die Ergebnisse isoliert darstellen. So kann man sehen, welcher Parameter welchen Effekt hat.

**Oben links: Gesamtdurchlaufzeit nach Störungsdauer (block_until)**
- **X-Achse:** 30, 60, 90, 120, 150, 180 Minuten Maschinenblockade
- **Erkenntnis:** Längere Blockade führt zu etwas höherer Gesamtdurchlaufzeit, aber der Unterschied ist gering
- Alle Werte liegen deutlich über der Baseline-Linie bei 930 Zeiteinheiten (rot gestrichelt)

**Oben Mitte: Verspätungskosten nach Liefertermin-Enge (due_tighten_min)**
- **X-Achse:** 0, 20, 40, 50, 100 Minuten, um die die Liefertermine enger gemacht wurden
- **Erkenntnis:** Engere Liefertermine führen zu **deutlich höheren** Verspätungskosten!
- Bei Verengung um 0 Minuten: circa 5.000 bis 10.000
- Bei Verengung um 100 Minuten: circa 20.000 bis 25.000
- **Bedeutung:** Unrealistisch enge Liefertermine führen zu hohen Verspätungskosten. Die Liefertermine sollten realistisch gewählt werden.

**Oben rechts: Abweichungskosten nach Modus (ungewichtet vs. zeitgewichtet)**
- **Erkenntnis:** Der zeitgewichtete Modus (`twdev`) führt zu **circa doppelt so hohen** Abweichungskosten wie der ungewichtete Modus (`dev`)
- Das ist beabsichtigt: Der zeitgewichtete Modus bestraft Änderungen an nahen Operationen stärker

**Unten links: Gesamtdurchlaufzeit nach blockierter Maschine**
- **X-Achse:** Maschinen M00, M03, M05
- Alle drei Maschinen zeigen ähnliche Gesamtdurchlaufzeiten zwischen circa 1.200 und 1.300 Zeiteinheiten
- M05 hat etwas mehr Ausreißer nach oben
- **Bedeutung:** Alle getesteten Maschinen sind ähnlich kritisch für den Produktionsablauf

**Unten Mitte: Zeitgewichtete Abweichungskosten nach Zeitfenster (dev_window_min)**
- **X-Achse:** 240, 480, 720 Minuten (entspricht 4, 8 und 12 Stunden)
- **Erkenntnis:** Größeres Zeitfenster führt zu **VIEL höheren** zeitgewichteten Abweichungskosten!
- Bei 240 Minuten Zeitfenster: circa 80.000 bis 100.000
- Bei 720 Minuten Zeitfenster: circa 300.000 bis 600.000
- **Bedeutung:** Ein großes Zeitfenster bedeutet, dass mehr Operationen in den „nahen" Bereich fallen und somit bei Änderung höher bestraft werden

**Unten rechts: Zeitgewichtete Abweichungskosten nach Bucket-Größe (dev_bucket_min)**
- **X-Achse:** 30, 60, 120 Minuten
- **Erkenntnis:** Kleinere Buckets führen zu höheren Kosten
- Bei 30 Minuten Bucket-Größe: circa 300.000 bis 600.000
- Bei 120 Minuten Bucket-Größe: circa 100.000 bis 150.000
- **Bedeutung:** Feinere Zeiteinteilung bedeutet strengere Gewichtung nach zeitlicher Nähe

---

## 🏆 Optimales Gewichtsverhältnis (VERALTET - siehe ABSCHLUSSBERICHT.md)

Basierend auf den 100 Experimenten ergibt sich folgendes optimales Verhältnis:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│   OPTIMALES VERHÄLTNIS:                                                                     │
│   Verspätungs-Gewicht : Verfrühungs-Gewicht : Abweichungs-Gewicht  =  1 : 1 : 2            │
│                                                                                             │
│   Erwartete Ergebnisse bei typischer Störung (Maschine M00 blockiert 60-90 Minuten):       │
│   • Gesamtdurchlaufzeit: 1185 Zeiteinheiten (+27% gegenüber Baseline 930)                  │
│   • Verspätungskosten: circa 2.234 (niedrig)                                               │
│   • Abweichungskosten: circa 43.000 ungewichtet (moderat)                                  │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Warum dieses Verhältnis?

| Aspekt | Bei Verhältnis 1:1:2 | Bei Verhältnis 1:1:5 | Bei Verhältnis 1:1:10 |
|--------|----------------------|----------------------|-----------------------|
| Gesamtdurchlaufzeit | +27% gegenüber Baseline | +29% gegenüber Baseline | +34% gegenüber Baseline |
| Flexibilität bei Störungen | ✅ Hoch | ⚠️ Mittel | ❌ Niedrig |
| Rechenzeit des Solvers | ✅ Schnell | ⚠️ Mittel | ❌ Langsam |
| Absolute Verbesserung der Planstabilität | Referenz | +1% | +3% |

**Fazit:** Ab einem Abweichungs-Gewicht größer als 2 gibt es kaum noch Verbesserung bei der absoluten Planstabilität, aber die Kosten und die Rechenzeit steigen stark an. **Das Verhältnis 1:1:2 ist der optimale Kompromiss (Sweet Spot).**

---

</details>
