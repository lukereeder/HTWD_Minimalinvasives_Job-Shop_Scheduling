#!/usr/bin/env python3
"""
Analysiert alle finalen Experimente und erstellt Zusammenfassungen
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_all_results() -> List[Dict]:
    """Lädt alle Experiment-Ergebnisse"""
    results_dir = Path("data/output/final_experiments")
    all_results = []
    
    if not results_dir.exists():
        print(f"❌ Verzeichnis nicht gefunden: {results_dir}")
        return []
    
    for json_file in sorted(results_dir.glob("*/comparison_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data['_source_file'] = str(json_file)
                all_results.append(data)
        except Exception as e:
            print(f"⚠️  Fehler beim Laden von {json_file}: {e}")
    
    return all_results


def extract_metrics(results: List[Dict]) -> pd.DataFrame:
    """Extrahiert Metriken aus allen Experimenten"""
    rows = []
    
    for result in results:
        params = result['parameters']
        
        # Standard-Deviation Experiment
        std_exp = result['experiments'].get('standard_deviation', {})
        std_result = std_exp.get('result', {})
        std_shifts = std_result.get('shift_summaries', [])
        
        # Time-Weighted-Deviation Experiment
        twdev_exp = result['experiments'].get('time_weighted_deviation', {})
        twdev_result = twdev_exp.get('result', {})
        twdev_shifts = twdev_result.get('shift_summaries', [])
        
        # Blockaden
        blockades = params.get('machine_blockades', [])
        blockade_str = "none"
        if blockades:
            blockade_str = "+".join([b['machine'] for b in blockades])
        
        row = {
            'util': params['util'],
            'sigma': params['sigma'],
            'blockade': blockade_str,
            'num_blockades': len(blockades) if blockades else 0,
            'time_limit': params['time_limit'],
            'std_shifts': len(std_shifts),
            'twdev_shifts': len(twdev_shifts),
            'std_exp_id': std_exp.get('experiment_id'),
            'twdev_exp_id': twdev_exp.get('experiment_id'),
            'source_file': result.get('_source_file', ''),
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_summary_table(df: pd.DataFrame) -> None:
    """Erstellt Zusammenfassungstabelle"""
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG ALLER EXPERIMENTE")
    print("="*80)
    print(f"\nAnzahl Experimente: {len(df)}")
    print(f"Utilization-Werte: {sorted(df['util'].unique())}")
    print(f"Sigma-Werte: {sorted(df['sigma'].unique())}")
    print(f"Blockade-Szenarien: {sorted(df['blockade'].unique())}")
    print("\n" + "-"*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


def create_comparison_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """Erstellt Übersichts-Visualisierung"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experimente-Übersicht: Standard vs. Time-Weighted Deviation', fontsize=16, fontweight='bold')
    
    # 1. Experimente nach Utilization
    ax1 = axes[0, 0]
    util_counts = df.groupby(['util', 'blockade']).size().unstack(fill_value=0)
    util_counts.plot(kind='bar', ax=ax1, colormap='viridis')
    ax1.set_title('Experimente nach Utilization und Blockade')
    ax1.set_xlabel('Utilization')
    ax1.set_ylabel('Anzahl Experimente')
    ax1.legend(title='Blockade', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Experimente nach Sigma
    ax2 = axes[0, 1]
    sigma_counts = df.groupby(['sigma', 'blockade']).size().unstack(fill_value=0)
    sigma_counts.plot(kind='bar', ax=ax2, colormap='plasma')
    ax2.set_title('Experimente nach Sigma und Blockade')
    ax2.set_xlabel('Sigma')
    ax2.set_ylabel('Anzahl Experimente')
    ax2.legend(title='Blockade', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Blockade-Typen
    ax3 = axes[1, 0]
    blockade_counts = df['blockade'].value_counts()
    blockade_counts.plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_title('Verteilung der Blockade-Szenarien')
    ax3.set_xlabel('Anzahl Experimente')
    ax3.set_ylabel('Blockade-Typ')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Utilization vs. Sigma Heatmap
    ax4 = axes[1, 1]
    pivot = df.groupby(['util', 'sigma']).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Anzahl Experimente'})
    ax4.set_title('Experimente: Utilization × Sigma')
    ax4.set_xlabel('Sigma')
    ax4.set_ylabel('Utilization')
    
    plt.tight_layout()
    output_file = output_dir / 'experiments_overview.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Übersicht gespeichert: {output_file}")
    plt.close()


def export_to_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """Exportiert Zusammenfassung als CSV"""
    output_file = output_dir / 'experiments_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ CSV exportiert: {output_file}")


def main():
    print("\n" + "="*80)
    print("ANALYSE DER FINALEN EXPERIMENTE")
    print("="*80)
    print("Lade Ergebnisse...")
    
    results = load_all_results()
    
    if not results:
        print("❌ Keine Ergebnisse gefunden!")
        print("   Bitte zuerst Experimente durchführen: ./run_all_final_experiments.sh")
        return 1
    
    print(f"✓ {len(results)} Experimente geladen")
    
    # Extrahiere Metriken
    df = extract_metrics(results)
    
    # Erstelle Zusammenfassung
    create_summary_table(df)
    
    # Output-Verzeichnis
    output_dir = Path("data/output/final_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Erstelle Visualisierungen
    print("Erstelle Visualisierungen...")
    create_comparison_overview(df, output_dir)
    
    # Exportiere CSV
    export_to_csv(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*80)
    print(f"Ergebnisse in: {output_dir}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    """
    Analyse ausführen:
    python3 analyze_final_results.py
    """
    exit(main())



