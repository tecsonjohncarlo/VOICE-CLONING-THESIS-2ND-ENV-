"""
Dynamic Metrics Analyzer with Automatic Hardware Detection

Generates thesis-ready visualizations from monitoring data.
No hardcoding - automatically detects all hardware from CSV files.

Usage:
    python analyze_metrics.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11


class DynamicMetricsAnalyzer:
    """Analyze metrics with dynamic hardware detection from actual test data"""
    
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.data: Dict[str, pd.DataFrame] = {}
        self.hardware_info: Dict[str, dict] = {}
        self._load_all_csv_with_metadata()
    
    def _load_all_csv_with_metadata(self):
        """Load CSV files and extract actual hardware info from data"""
        
        print("\n" + "="*70)
        print("LOADING MONITORING DATA")
        print("="*70)
        
        for csv_file in self.metrics_dir.glob("synthesis_*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                if df.empty:
                    print(f"⚠️  Skipping empty file: {csv_file.name}")
                    continue
                
                # Extract hardware info from ACTUAL data (first row)
                first_row = df.iloc[0]
                
                # Use the actual hardware_tier from data
                hardware_tier = first_row['hardware_tier']
                device_type = first_row['device_type']
                
                # Create unique key from actual hardware data
                hw_key = f"{hardware_tier}_{device_type}"
                
                self.data[hw_key] = df
                
                # Store hardware metadata
                self.hardware_info[hw_key] = {
                    'hardware_tier': hardware_tier,
                    'device_type': device_type,
                    'cpu_model': first_row.get('cpu_model', 'Unknown'),
                    'cpu_cores': first_row.get('cpu_cores', 0),
                    'total_memory_gb': first_row.get('total_memory_gb', 0),
                    'gpu_name': first_row.get('gpu_name', ''),
                    'csv_file': str(csv_file),
                    'num_syntheses': len(df),
                    'date': csv_file.stem.split('_')[-1],
                }
                
                print(f"✅ Loaded {hw_key}: {len(df)} syntheses")
                print(f"   Device: {device_type} | CPU: {first_row.get('cpu_model', 'Unknown')[:40]}")
                
            except Exception as e:
                print(f"❌ Error loading {csv_file.name}: {e}")
        
        print(f"\n📊 Total hardware configurations: {len(self.data)}")
        print("="*70 + "\n")
    
    def _get_dynamic_color(self, hw_key: str) -> str:
        """Assign color dynamically based on device type"""
        device_type = self.hardware_info[hw_key]['device_type']
        
        color_map = {
            'mps': '#A2AAAD',      # Apple Silicon - Gray
            'cpu': '#0071C5',      # CPU-only - Intel Blue
            'cuda': '#76B900',     # NVIDIA GPU - Green
        }
        
        return color_map.get(device_type, '#999999')
    
    def _get_dynamic_label(self, hw_key: str) -> str:
        """Create readable label from actual hardware data"""
        hw = self.hardware_info[hw_key]
        return f"{hw['hardware_tier']}\n({hw['device_type']})"
    
    # ============================================
    # FIGURE 1: Dynamic Memory Comparison
    # ============================================
    def plot_memory_comparison(self):
        """Memory comparison using actual detected hardware"""
        if not self.data:
            print("⚠️  No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        hw_keys = sorted(self.data.keys())
        peak_memory = [self.data[k]['peak_memory_mb'].mean() for k in hw_keys]
        std_memory = [self.data[k]['peak_memory_mb'].std() for k in hw_keys]
        
        # Use actual hardware names and dynamic colors
        labels = [self._get_dynamic_label(k) for k in hw_keys]
        colors = [self._get_dynamic_color(k) for k in hw_keys]
        
        bars = ax.bar(range(len(hw_keys)), peak_memory, yerr=std_memory,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(hw_keys)))
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Peak Memory Usage (MB)', fontsize=13, fontweight='bold')
        ax.set_title('Memory Usage Comparison: Actual Detected Hardware', 
                     fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_memory[i] + 50,
                   f'{height:.0f}MB\n(n={len(self.data[hw_keys[i]])})',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        output_path = self.metrics_dir / 'fig1_memory_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    # ============================================
    # FIGURE 2: Dynamic RTF Comparison
    # ============================================
    def plot_rtf_comparison(self):
        """RTF with actual hardware detection"""
        if not self.data:
            print("⚠️  No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        hw_keys = sorted(self.data.keys())
        rtf = [self.data[k]['rtf'].mean() for k in hw_keys]
        std_rtf = [self.data[k]['rtf'].std() for k in hw_keys]
        
        labels = [self._get_dynamic_label(k) for k in hw_keys]
        colors = [self._get_dynamic_color(k) for k in hw_keys]
        
        bars = ax.bar(range(len(hw_keys)), rtf, yerr=std_rtf,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Real-time reference line
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2.5, 
                  label='Real-time (RTF=1.0)', alpha=0.8)
        
        ax.set_xticks(range(len(hw_keys)))
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Real-Time Factor (RTF)', fontsize=13, fontweight='bold')
        ax.set_title('Synthesis Speed: Real-Time Factor by Detected Device', 
                     fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_rtf[i] + 0.2,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        output_path = self.metrics_dir / 'fig2_rtf_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    # ============================================
    # FIGURE 3: Success Rate Comparison
    # ============================================
    def plot_success_rate(self):
        """Success rate by hardware"""
        if not self.data:
            print("⚠️  No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        hw_keys = sorted(self.data.keys())
        success_rates = [(self.data[k]['success'].sum() / len(self.data[k])) * 100 
                        for k in hw_keys]
        
        labels = [self._get_dynamic_label(k) for k in hw_keys]
        colors = [self._get_dynamic_color(k) for k in hw_keys]
        
        bars = ax.bar(range(len(hw_keys)), success_rates,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(hw_keys)))
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title('Synthesis Success Rate by Hardware', 
                     fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        output_path = self.metrics_dir / 'fig3_success_rate.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    # ============================================
    # DYNAMIC HARDWARE INFO TABLE
    # ============================================
    def create_hardware_summary_table(self):
        """Create summary table with actual hardware specs"""
        if not self.data:
            print("⚠️  No data to create table")
            return
        
        fig, ax = plt.subplots(figsize=(16, len(self.hardware_info) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        headers = ['Hardware Tier', 'Device', 'CPU Model', 'Cores', 'RAM (GB)',
                  'Avg Memory (MB)', 'Peak Memory (MB)', 'Avg RTF', 'Success Rate', 'Tests']
        
        for hw_key in sorted(self.data.keys()):
            data = self.data[hw_key]
            hw_info = self.hardware_info[hw_key]
            
            success_rate = (data['success'].sum() / len(data)) * 100
            
            row = [
                hw_info['hardware_tier'],
                hw_info['device_type'],
                hw_info['cpu_model'][:30] + '...' if len(hw_info['cpu_model']) > 30 else hw_info['cpu_model'],
                str(hw_info['cpu_cores']),
                f"{hw_info['total_memory_gb']:.1f}",
                f"{data['peak_memory_mb'].mean():.0f}",
                f"{data['peak_memory_mb'].max():.0f}",
                f"{data['rtf'].mean():.2f}x",
                f"{success_rate:.1f}%",
                str(hw_info['num_syntheses'])
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.12, 0.08, 0.18, 0.06, 0.08, 0.10, 0.10, 0.08, 0.10, 0.06])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by device type
        for i, hw_key in enumerate(sorted(self.data.keys()), start=1):
            color = self._get_dynamic_color(hw_key)
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_alpha(0.3)
        
        plt.title('Hardware Summary: Actual Detected Devices\n', 
                 fontsize=16, fontweight='bold')
        output_path = self.metrics_dir / 'fig_hardware_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    # ============================================
    # GENERATE METADATA FILE
    # ============================================
    def export_hardware_metadata(self):
        """Export detected hardware info as JSON for reference"""
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_hardware_configs': len(self.hardware_info),
            'hardware_detected': []
        }
        
        for hw_key, hw_info in self.hardware_info.items():
            data = self.data[hw_key]
            
            metadata['hardware_detected'].append({
                'key': hw_key,
                'hardware_tier': hw_info['hardware_tier'],
                'device_type': hw_info['device_type'],
                'cpu_model': hw_info['cpu_model'],
                'cpu_cores': hw_info['cpu_cores'],
                'total_memory_gb': hw_info['total_memory_gb'],
                'gpu_name': hw_info['gpu_name'],
                'num_tests': hw_info['num_syntheses'],
                'date_tested': hw_info['date'],
                'csv_source': hw_info['csv_file'],
                'memory_stats': {
                    'mean_mb': float(data['peak_memory_mb'].mean()),
                    'std_mb': float(data['peak_memory_mb'].std()),
                    'max_mb': float(data['peak_memory_mb'].max()),
                    'min_mb': float(data['peak_memory_mb'].min()),
                },
                'performance_stats': {
                    'avg_rtf': float(data['rtf'].mean()),
                    'avg_tokens_per_sec': float(data['tokens_per_second'].mean()),
                    'avg_duration_s': float(data['total_duration_s'].mean()),
                    'success_rate': float((data['success'].sum() / len(data)) * 100),
                }
            })
        
        metadata_file = self.metrics_dir / 'hardware_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Exported hardware metadata: {metadata_file}")
        return metadata
    
    def generate_all_plots(self):
        """Generate all visualizations with dynamic hardware detection"""
        print("\n" + "="*70)
        print("GENERATING THESIS VISUALIZATIONS - DYNAMIC HARDWARE DETECTION")
        print("="*70 + "\n")
        
        if not self.data:
            print("❌ No data found! Run some syntheses first.")
            return
        
        print("Detected Hardware Configurations:")
        for hw_key, hw_info in sorted(self.hardware_info.items()):
            print(f"  • {hw_key}: {hw_info['num_syntheses']} tests")
            print(f"    CPU: {hw_info['cpu_model'][:50]}")
            print(f"    Device: {hw_info['device_type']}")
        print()
        
        self.plot_memory_comparison()
        self.plot_rtf_comparison()
        self.plot_success_rate()
        self.create_hardware_summary_table()
        self.export_hardware_metadata()
        
        print("\n" + "="*70)
        print(f"✅ All plots saved to: {self.metrics_dir}")
        print("="*70)


# Run with automatic hardware detection
if __name__ == "__main__":
    analyzer = DynamicMetricsAnalyzer()
    analyzer.generate_all_plots()
