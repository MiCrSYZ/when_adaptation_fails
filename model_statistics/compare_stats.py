"""
Compare parameter statistics across multiple trainers.
Run from repo root, e.g.:
python -m model_statistics.compare_stats --output-dir ./model_comparison
"""

import argparse
import os
import sys
import json
import subprocess
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt


def run_statistics(trainer, dataset, backbone, num_classes, output_dir, compute_flops=False):
    """Run unified_stats for one trainer."""
    print(f"\n{'='*80}")
    print(f"Statistics for {trainer}...")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, "-m", "model_statistics.unified_stats",
        "--trainer", trainer,
        "--dataset", dataset,
        "--backbone", backbone,
        "--num-classes", str(num_classes),
        "--output-dir", output_dir
    ]
    
    if compute_flops:
        cmd.extend(["--compute-flops", "--flops-method", "all"])
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Statistics failed for {trainer}: {e}")
        return False


def load_statistics(output_dir, trainers):
    """Load JSON outputs from unified_stats."""
    results = {}
    for trainer in trainers:
        stats_file = os.path.join(output_dir, f"{trainer}_statistics.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                results[trainer] = json.load(f)
        else:
            print(f"Warning: missing stats file for {trainer}")
    
    return results


def create_comparison_table(results):
    """Build a summary DataFrame."""
    data = []
    
    for trainer, stats in results.items():
        row = {
            'Trainer': trainer,
            'Total Params': stats['total_params'],
            'Trainable Params': stats['trainable_params'],
            'Frozen Params': stats['frozen_params'],
            'Trainable Ratio': stats['trainable_ratio'],
            'Memory (MB)': stats['memory_usage_mb']['total_est_mb']
        }
        
        if 'prompt_params' in stats and stats['prompt_params']:
            row['Prompt Params'] = stats['prompt_params'].get('prompt_total', 0)
        
        if 'flops' in stats:
            if 'thop' in stats['flops']:
                row['FLOPs'] = stats['flops']['thop']['flops']
                row['FLOPs Str'] = stats['flops']['thop']['flops_str']
            elif 'fvcore' in stats['flops']:
                row['FLOPs'] = stats['flops']['fvcore']['flops']
                row['FLOPs Str'] = stats['flops']['fvcore']['flops_str']
            elif 'manual' in stats['flops']:
                row['FLOPs'] = stats['flops']['manual']['flops']
                row['FLOPs Str'] = stats['flops']['manual']['flops_str']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def format_number(num):
    """Human-readable large numbers."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_comparison_table(df):
    """Print comparison table."""
    print("\n" + "="*140)
    print("Model parameter comparison")
    print("="*140)
    
    has_flops = 'FLOPs Str' in df.columns
    
    if has_flops:
        print(f"\n{'Trainer':<25} | {'Total Params':>15} | {'Trainable':>15} | "
              f"{'Frozen':>15} | {'Ratio':>10} | {'FLOPs':>12} | {'Memory(MB)':>12}")
        print("-"*140)
        
        for _, row in df.iterrows():
            flops_str = row.get('FLOPs Str', 'N/A')
            print(f"{row['Trainer']:<25} | "
                  f"{format_number(row['Total Params']):>15} | "
                  f"{format_number(row['Trainable Params']):>15} | "
                  f"{format_number(row['Frozen Params']):>15} | "
                  f"{row['Trainable Ratio']:>10} | "
                  f"{flops_str:>12} | "
                  f"{row['Memory (MB)']:>12.2f}")
    else:
        print(f"\n{'Trainer':<25} | {'Total Params':>15} | {'Trainable':>15} | "
              f"{'Frozen':>15} | {'Ratio':>10} | {'Memory(MB)':>12}")
        print("-"*120)
        
        for _, row in df.iterrows():
            print(f"{row['Trainer']:<25} | "
                  f"{format_number(row['Total Params']):>15} | "
                  f"{format_number(row['Trainable Params']):>15} | "
                  f"{format_number(row['Frozen Params']):>15} | "
                  f"{row['Trainable Ratio']:>10} | "
                  f"{row['Memory (MB)']:>12.2f}")
    
    print("="*140 + "\n")


def plot_comparison(df, output_dir):
    """Save comparison plots."""
    has_flops = 'FLOPs' in df.columns

    if has_flops:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    trainers = df['Trainer']

    ax1 = axes[0, 0]
    total_params = df['Total Params'] / 1e6
    trainable_params = df['Trainable Params'] / 1e6

    x = range(len(trainers))
    width = 0.35

    ax1.bar([i - width / 2 for i in x], total_params, width, label='Total', alpha=0.8)
    ax1.bar([i + width / 2 for i in x], trainable_params, width, label='Trainable', alpha=0.8)
    ax1.set_xlabel('Trainer')
    ax1.set_ylabel('Parameters (M)')
    ax1.set_title('Parameter Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(trainers, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ratios = [float(r.strip('%')) for r in df['Trainable Ratio']]
    colors = plt.cm.viridis(range(len(trainers)))

    bars = ax2.bar(trainers, ratios, color=colors, alpha=0.8)
    ax2.set_xlabel('Trainer')
    ax2.set_ylabel('Trainable Ratio (%)')
    ax2.set_title('Trainable Parameter Ratio')
    ax2.set_xticklabels(trainers, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{ratio:.2f}%', ha='center', va='bottom', fontsize=9)

    if has_flops:
        ax3 = axes[1, 0]
    else:
        ax3 = axes[1, 0]

    memory = df['Memory (MB)']
    ax3.bar(trainers, memory, color='coral', alpha=0.8)
    ax3.set_xlabel('Trainer')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.set_title('Estimated Memory Usage')
    ax3.set_xticklabels(trainers, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    if has_flops:
        ax4 = axes[1, 1]
    else:
        ax4 = axes[1, 1]

    if 'Prompt Params' in df.columns:
        prompt_params = df['Prompt Params'] / 1e3
        ax4.bar(trainers, prompt_params, color='lightgreen', alpha=0.8)
        ax4.set_xlabel('Trainer')
        ax4.set_ylabel('Prompt Parameters (K)')
        ax4.set_title('Prompt Parameter Comparison')
        ax4.set_xticklabels(trainers, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No Prompt Data Available',
                 ha='center', va='center', transform=ax4.transAxes)

    if has_flops:
        ax5 = axes[0, 2]
        flops_values = df['FLOPs'] / 1e9
        ax5.bar(trainers, flops_values, color='skyblue', alpha=0.8)
        ax5.set_xlabel('Trainer')
        ax5.set_ylabel('FLOPs (G)')
        ax5.set_title('FLOPs Comparison')
        ax5.set_xticklabels(trainers, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')

        ax6 = axes[1, 2]
        efficiency = df['FLOPs'] / df['Trainable Params']
        ax6.bar(trainers, efficiency, color='mediumpurple', alpha=0.8)
        ax6.set_xlabel('Trainer')
        ax6.set_ylabel('FLOPs per Trainable Param')
        ax6.set_title('Computational Efficiency')
        ax6.set_xticklabels(trainers, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    plt.close()


def create_detailed_comparison(results, output_dir):
    """Per-component prompt parameter table and CSV."""
    prompt_comparison = OrderedDict()
    
    for trainer, stats in results.items():
        if 'prompt_params' in stats and stats['prompt_params']:
            prompt_comparison[trainer] = stats['prompt_params']
    
    if not prompt_comparison:
        print("No prompt parameter breakdown found.")
        return
    
    print("\n" + "="*120)
    print("Prompt parameters (detail)")
    print("="*120)
    
    all_param_types = set()
    for params in prompt_comparison.values():
        all_param_types.update(params.keys())
    
    print(f"\n{'Parameter Type':<30} | " + " | ".join([f"{t:>18}" for t in prompt_comparison.keys()]))
    print("-"*120)
    
    for param_type in sorted(all_param_types):
        if param_type == 'prompt_total':
            continue
        
        values = []
        for trainer in prompt_comparison.keys():
            value = prompt_comparison[trainer].get(param_type, 0)
            values.append(format_number(value))
        
        print(f"{param_type:<30} | " + " | ".join([f"{v:>18}" for v in values]))
    
    print("-"*120)
    values = []
    for trainer in prompt_comparison.keys():
        value = prompt_comparison[trainer].get('prompt_total', 0)
        values.append(format_number(value))
    
    print(f"{'prompt_total':<30} | " + " | ".join([f"{v:>18}" for v in values]))
    print("="*120 + "\n")
    
    csv_file = os.path.join(output_dir, 'prompt_params_comparison.csv')
    
    data = []
    for param_type in sorted(all_param_types):
        row = {'Parameter Type': param_type}
        for trainer in prompt_comparison.keys():
            row[trainer] = prompt_comparison[trainer].get(param_type, 0)
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")


def generate_summary_report(results, output_dir):
    """Write comparison_report.txt."""
    report_file = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Model parameter comparison report\n")
        f.write("="*100 + "\n\n")
        
        f.write("1. Trainers:\n")
        for trainer in results.keys():
            f.write(f"   - {trainer}\n")
        f.write("\n")
        
        f.write("2. Parameter counts:\n\n")
        f.write(f"{'Trainer':<25} | {'Total':>15} | {'Trainable':>15} | {'Ratio':>10}\n")
        f.write("-"*70 + "\n")
        
        for trainer, stats in results.items():
            f.write(f"{trainer:<25} | "
                   f"{format_number(stats['total_params']):>15} | "
                   f"{format_number(stats['trainable_params']):>15} | "
                   f"{stats['trainable_ratio']:>10}\n")
        
        f.write("\n")
        
        f.write("3. Estimated memory (MB):\n\n")
        for trainer, stats in results.items():
            memory = stats['memory_usage_mb']['total_est_mb']
            f.write(f"{trainer:<25}: {memory:>10.2f} MB\n")
        
        f.write("\n")
        
        f.write("4. Summary:\n\n")
        
        max_trainer = max(results.items(), key=lambda x: x[1]['trainable_params'])
        min_trainer = min(results.items(), key=lambda x: x[1]['trainable_params'])
        
        f.write(f"   - Most trainable params: {max_trainer[0]} "
               f"({format_number(max_trainer[1]['trainable_params'])})\n")
        f.write(f"   - Fewest trainable params: {min_trainer[0]} "
               f"({format_number(min_trainer[1]['trainable_params'])})\n")
        
        if len(results) >= 2:
            base_params = min_trainer[1]['trainable_params']
            max_params = max_trainer[1]['trainable_params']
            increase = (max_params - base_params) / base_params * 100
            f.write(f"   - Relative spread (max vs min trainable): {increase:.2f}%\n")
        
        f.write("\n")
        
        f.write("5. Prompt parameters (total):\n\n")
        for trainer, stats in results.items():
            if 'prompt_params' in stats and 'prompt_total' in stats['prompt_params']:
                prompt_total = stats['prompt_params']['prompt_total']
                f.write(f"{trainer:<25}: {format_number(prompt_total)}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"Saved report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare parameter stats across trainers")
    parser.add_argument("--trainers", type=str, nargs='+',
                       default=["MaPLe", "BiDirMaPLe", "AdaptiveMaPLe", "AdaptiveBiDirMaPLe"],
                       help="Trainers to compare")
    parser.add_argument("--dataset", type=str, default="imagenet",
                       help="Dataset name (passed to unified_stats)")
    parser.add_argument("--backbone", type=str, default="ViT-B/16",
                       help="CLIP backbone")
    parser.add_argument("--num-classes", type=int, default=100,
                       help="Number of classes")
    parser.add_argument("--output-dir", type=str, default="./model_comparison",
                       help="Output directory")
    parser.add_argument("--skip-stats", action="store_true",
                       help="Skip running unified_stats; only aggregate existing JSON")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_stats:
        print("\nRunning unified_stats for each trainer...")
        for trainer in args.trainers:
            success = run_statistics(
                trainer, args.dataset, args.backbone, 
                args.num_classes, args.output_dir
            )
            if not success:
                print(f"Warning: skipped {trainer} due to failure")
    
    print("\nLoading statistics JSON...")
    results = load_statistics(args.output_dir, args.trainers)
    
    if not results:
        print("Error: no statistics files found")
        return
    
    print(f"Loaded {len(results)} trainer(s)")
    
    df = create_comparison_table(results)
    
    print_comparison_table(df)
    
    csv_file = os.path.join(args.output_dir, 'model_comparison.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved table: {csv_file}")
    
    print("\nPlotting...")
    plot_comparison(df, args.output_dir)
    
    print("\nPrompt breakdown...")
    create_detailed_comparison(results, args.output_dir)
    
    print("\nSummary report...")
    generate_summary_report(results, args.output_dir)
    
    print("\n" + "="*100)
    print("Done.")
    print(f"Output: {args.output_dir}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

        