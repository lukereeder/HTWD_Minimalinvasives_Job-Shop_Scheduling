#!/usr/bin/env python3
"""
Erstellt die 3 fehlenden Plots für den Abschlussbericht (Abb. 8.1-8.3)
Aus Experiment 34 (Standard-Deviation) und 35 (Time-Weighted-Deviation)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Output-Verzeichnis
OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(exist_ok=True)

# Datenbank
DB_PATH = "experiments.db"


def load_shift_data():
    """Lädt Shift-Daten für Experimente 34 und 35"""
    conn = sqlite3.connect(DB_PATH)
    
    # Hole alle Shifts mit ihren Metriken
    query = """
    WITH shift_stats AS (
        SELECT 
            e.id as experiment_id,
            e.type as experiment_type,
            sj.shift_number,
            COUNT(DISTINCT sj.id) as num_jobs,
            COUNT(DISTINCT so.position_number) as num_operations,
            MIN(so.start) as min_start,
            MAX(so.end) as max_end,
            (MAX(so.end) - MIN(so.start)) as makespan
        FROM experiment e
        JOIN schedule_job sj ON e.id = sj.experiment_id
        JOIN schedule_operation so ON sj.id = so.job_id 
            AND sj.experiment_id = so.experiment_id 
            AND sj.shift_number = so.shift_number
        WHERE e.id IN (34, 35)
        GROUP BY e.id, e.type, sj.shift_number
    )
    SELECT * FROM shift_stats
    ORDER BY experiment_id, shift_number
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def parse_log_files():
    """Parst Log-Dateien für Solver-Status und Kosten"""
    log_data = []
    
    # Parse comparison log files (diese enthalten alle Shift-Daten)
    comparison_logs = [
        ("data/comparison_std_20260126_012203.log", 34, "Standard-Deviation"),
        ("data/comparison_twdev_20260126_012203.log", 35, "Time-Weighted-Deviation"),
    ]
    
    for log_path, exp_id, exp_type in comparison_logs:
        if not Path(log_path).exists():
            print(f"⚠️  Log nicht gefunden: {log_path}")
            continue
        
        data_list = parse_comparison_log(log_path, exp_id, exp_type)
        log_data.extend(data_list)
        print(f"✅ {len(data_list)} Shifts aus {log_path} geladen")
    
    return pd.DataFrame(log_data)


def parse_comparison_log(log_path, exp_id, exp_type):
    """Parst eine Comparison-Log-Datei und extrahiert alle Shift-Daten"""
    data_list = []
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        current_shift = None
        current_data = None
        
        for line in lines:
            # Neue Shift-Nummer
            if 'shift' in line.lower() and ':' in line and 'to' in line:
                # Parse: "Experiment 35 shift 2: 2880 to 4320"
                try:
                    parts = line.split('shift')[1].split(':')
                    shift_num = int(parts[0].strip())
                    
                    # Speichere vorherigen Shift
                    if current_data is not None:
                        data_list.append(current_data)
                    
                    # Starte neuen Shift
                    current_shift = shift_num
                    current_data = {
                        'experiment_id': exp_id,
                        'experiment_type': exp_type,
                        'shift_number': shift_num,
                        'status': None,
                        'wall_time': None,
                        'tardiness_cost': None,
                        'earliness_cost': None,
                        'deviation_cost': None,
                        'twdev_cost': None,
                    }
                except:
                    pass
            
            # Parse Daten für aktuellen Shift
            if current_data is not None:
                if 'Status' in line and ':' in line:
                    if 'OPTIMAL' in line:
                        current_data['status'] = 'OPTIMAL'
                    elif 'FEASIBLE' in line:
                        current_data['status'] = 'FEASIBLE'
                    elif 'INFEASIBLE' in line:
                        current_data['status'] = 'INFEASIBLE'
                
                elif 'Wall time' in line:
                    try:
                        current_data['wall_time'] = float(line.split(':')[1].strip())
                    except:
                        pass
                
                elif 'Tardiness cost' in line:
                    try:
                        current_data['tardiness_cost'] = float(line.split(':')[1].strip())
                    except:
                        pass
                
                elif 'Earliness cost' in line:
                    try:
                        current_data['earliness_cost'] = float(line.split(':')[1].strip())
                    except:
                        pass
                
                elif 'Deviation cost' in line and 'weighted' not in line:
                    try:
                        current_data['deviation_cost'] = float(line.split(':')[1].strip())
                    except:
                        pass
                
                elif 'Time weighted deviation cost' in line:
                    try:
                        current_data['twdev_cost'] = float(line.split(':')[1].strip())
                    except:
                        pass
        
        # Speichere letzten Shift
        if current_data is not None:
            data_list.append(current_data)
        
        return data_list
    
    except Exception as e:
        print(f"❌ Fehler beim Parsen von {log_path}: {e}")
        return []


def parse_single_log(log_path, exp_id, shift_num, exp_type):
    """Parst eine einzelne Log-Datei (Legacy-Fallback)"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        data = {
            'experiment_id': exp_id,
            'experiment_type': exp_type,
            'shift_number': shift_num,
            'status': None,
            'wall_time': None,
            'tardiness_cost': None,
            'earliness_cost': None,
            'deviation_cost': None,
            'twdev_cost': None,
        }
        
        # Parse Status
        if 'Status              : OPTIMAL' in content:
            data['status'] = 'OPTIMAL'
        elif 'Status              : FEASIBLE' in content:
            data['status'] = 'FEASIBLE'
        elif 'Status              : INFEASIBLE' in content:
            data['status'] = 'INFEASIBLE'
        
        # Parse Wall Time
        for line in content.split('\n'):
            if 'Wall time' in line:
                try:
                    data['wall_time'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Tardiness cost' in line:
                try:
                    data['tardiness_cost'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Earliness cost' in line:
                try:
                    data['earliness_cost'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Deviation cost' in line and 'weighted' not in line:
                try:
                    data['deviation_cost'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Time weighted deviation cost' in line:
                try:
                    data['twdev_cost'] = float(line.split(':')[1].strip())
                except:
                    pass
        
        return data
    
    except Exception as e:
        print(f"⚠️  Fehler beim Parsen von {log_path}: {e}")
        return None


def create_solver_status_comparison(df):
    """Abb. 8.1: Balkendiagramm - Solver-Status-Vergleich"""
    print("\n📊 Erstelle Abb. 8.1: Solver-Status-Vergleich...")
    
    # Zähle Status pro Experiment
    status_counts = df.groupby(['experiment_type', 'status']).size().unstack(fill_value=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(status_counts.index))
    width = 0.35
    
    colors = {'OPTIMAL': '#2ecc71', 'FEASIBLE': '#f39c12', 'INFEASIBLE': '#e74c3c'}
    
    bottom = np.zeros(len(status_counts.index))
    for status in ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE']:
        if status in status_counts.columns:
            values = status_counts[status].values
            ax.bar(x, values, width, label=status, bottom=bottom, 
                   color=colors.get(status, '#95a5a6'))
            
            # Beschriftung
            for i, v in enumerate(values):
                if v > 0:
                    ax.text(i, bottom[i] + v/2, str(int(v)), 
                           ha='center', va='center', fontweight='bold', fontsize=12)
            
            bottom += values
    
    ax.set_ylabel('Anzahl Shifts', fontsize=12, fontweight='bold')
    ax.set_title('Abb. 8.1: Solver-Status-Vergleich (22 Shifts)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(status_counts.index, fontsize=11)
    ax.legend(title='Solver-Status', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 24)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'abb_8_1_solver_status.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gespeichert: {output_path}")
    plt.close()
    
    # Statistik ausgeben
    print(f"\n   Standard-Deviation:")
    print(f"     OPTIMAL:  {status_counts.loc['Standard-Deviation', 'OPTIMAL'] if 'OPTIMAL' in status_counts.columns else 0:.0f} Shifts")
    print(f"     FEASIBLE: {status_counts.loc['Standard-Deviation', 'FEASIBLE'] if 'FEASIBLE' in status_counts.columns else 0:.0f} Shifts")
    
    print(f"\n   Time-Weighted-Deviation:")
    print(f"     OPTIMAL:  {status_counts.loc['Time-Weighted-Deviation', 'OPTIMAL'] if 'OPTIMAL' in status_counts.columns else 0:.0f} Shifts")
    print(f"     FEASIBLE: {status_counts.loc['Time-Weighted-Deviation', 'FEASIBLE'] if 'FEASIBLE' in status_counts.columns else 0:.0f} Shifts")


def create_tardiness_deviation_scatter(df):
    """Abb. 8.2: Scatter-Plot - Tardiness vs. Deviation"""
    print("\n📊 Erstelle Abb. 8.2: Tardiness vs. Deviation...")
    
    # Filtere Shifts mit Daten
    plot_df = df[df['tardiness_cost'].notna() & df['deviation_cost'].notna()].copy()
    
    if len(plot_df) == 0:
        print("⚠️  Keine Daten für Tardiness/Deviation gefunden!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Standard-Deviation
    std_df = plot_df[plot_df['experiment_type'] == 'Standard-Deviation']
    ax.scatter(std_df['deviation_cost'], std_df['tardiness_cost'], 
               s=100, alpha=0.6, c='#3498db', marker='o', 
               label='Standard-Deviation', edgecolors='black', linewidth=1)
    
    # Time-Weighted-Deviation
    twdev_df = plot_df[plot_df['experiment_type'] == 'Time-Weighted-Deviation']
    ax.scatter(twdev_df['deviation_cost'], twdev_df['tardiness_cost'], 
               s=100, alpha=0.6, c='#e74c3c', marker='s', 
               label='Time-Weighted-Deviation', edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Deviation Cost (Summe |Δstart|)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tardiness Cost (Summe Verspätungen)', fontsize=12, fontweight='bold')
    ax.set_title('Abb. 8.2: Tardiness vs. Deviation - Vergleich der Ansätze', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Pareto-Front andeuteten (optisch)
    ax.axhline(y=plot_df['tardiness_cost'].median(), color='gray', 
               linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=plot_df['deviation_cost'].median(), color='gray', 
               linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'abb_8_2_tardiness_deviation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gespeichert: {output_path}")
    plt.close()
    
    # Statistik
    print(f"\n   Standard-Deviation:")
    print(f"     Tardiness (Mittel): {std_df['tardiness_cost'].mean():.1f}")
    print(f"     Deviation (Mittel): {std_df['deviation_cost'].mean():.1f}")
    
    print(f"\n   Time-Weighted-Deviation:")
    print(f"     Tardiness (Mittel): {twdev_df['tardiness_cost'].mean():.1f}")
    print(f"     Deviation (Mittel): {twdev_df['deviation_cost'].mean():.1f}")


def create_solving_time_boxplot(df):
    """Abb. 8.3: Box-Plot - Lösungszeiten pro Shift"""
    print("\n📊 Erstelle Abb. 8.3: Lösungszeiten Box-Plot...")
    
    # Filtere Shifts mit Zeitdaten
    plot_df = df[df['wall_time'].notna()].copy()
    
    if len(plot_df) == 0:
        print("⚠️  Keine Zeitdaten gefunden!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Box-Plot
    positions = [1, 2]
    data = [
        plot_df[plot_df['experiment_type'] == 'Standard-Deviation']['wall_time'],
        plot_df[plot_df['experiment_type'] == 'Time-Weighted-Deviation']['wall_time']
    ]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='green', linestyle='--', linewidth=2))
    
    # Farben
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Standard-Deviation', 'Time-Weighted-Deviation'], fontsize=11)
    ax.set_ylabel('Lösungszeit (Sekunden)', fontsize=12, fontweight='bold')
    ax.set_title('Abb. 8.3: Lösungszeiten pro Shift - Vergleich der Ansätze', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Box: Q1-Q3 (50% der Daten)'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Mittelwert')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'abb_8_3_solving_times.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gespeichert: {output_path}")
    plt.close()
    
    # Statistik
    for exp_type in ['Standard-Deviation', 'Time-Weighted-Deviation']:
        times = plot_df[plot_df['experiment_type'] == exp_type]['wall_time']
        print(f"\n   {exp_type}:")
        print(f"     Median:  {times.median():.2f}s")
        print(f"     Mittel:  {times.mean():.2f}s")
        print(f"     Min:     {times.min():.2f}s")
        print(f"     Max:     {times.max():.2f}s")


def main():
    print("\n" + "="*80)
    print("ERSTELLE ABSCHLUSSBERICHT PLOTS (Abb. 8.1-8.3)")
    print("="*80)
    
    # Prüfe Datenbank
    if not Path(DB_PATH).exists():
        print(f"❌ Datenbank nicht gefunden: {DB_PATH}")
        return 1
    
    print(f"✅ Datenbank gefunden: {DB_PATH}")
    
    # Lade Daten
    print("\n📂 Lade Daten aus Log-Dateien...")
    df = parse_log_files()
    
    if len(df) == 0:
        print("❌ Keine Log-Daten gefunden!")
        print("   Bitte prüfe: data/logs/Experiment_034/ und data/logs/Experiment_035/")
        return 1
    
    print(f"✅ {len(df)} Shifts geladen")
    print(f"   Experiment 34 (std-dev): {len(df[df['experiment_id']==34])} Shifts")
    print(f"   Experiment 35 (twdev):   {len(df[df['experiment_id']==35])} Shifts")
    
    # Erstelle Plots
    create_solver_status_comparison(df)
    create_tardiness_deviation_scatter(df)
    create_solving_time_boxplot(df)
    
    print("\n" + "="*80)
    print("✅ ALLE PLOTS ERFOLGREICH ERSTELLT")
    print("="*80)
    print(f"\nPlots gespeichert in: {OUTPUT_DIR}/")
    print("  - abb_8_1_solver_status.png")
    print("  - abb_8_2_tardiness_deviation.png")
    print("  - abb_8_3_solving_times.png")
    print("\nNächster Schritt: Plots in Abschlussbericht einbinden")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
