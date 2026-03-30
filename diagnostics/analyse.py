"""
Analyze Experiments A and B: comparison plots and summary statistics.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12


class ExperimentAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.results = {
            'ExperimentA': {},
            'ExperimentB': {}
        }

    def load_metrics(self, exp_path):
        """Load training metrics from disk."""
        metrics_file = exp_path / "visualizations_adaptive_bidir" / "training_metrics_adaptive_bidir.json"
        if not metrics_file.exists():
            # Alternate paths
            alt_paths = [
                exp_path / "training_metrics.json",
                exp_path / "metrics.json"
            ]
            for alt in alt_paths:
                if alt.exists():
                    metrics_file = alt
                    break

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Metrics file not found at {metrics_file}")
            return None

    def collect_results(self):
        """Collect results from all experiment runs."""
        print("Collecting results from experiments...")

        # Experiment A
        exp_a_path = self.output_dir / "ExperimentA"
        if exp_a_path.exists():
            for method in ['ParamMatched', 'Adaptive']:
                method_path = exp_a_path / method
                if method_path.exists():
                    self.results['ExperimentA'][method] = {}
                    for dataset_dir in method_path.iterdir():
                        if dataset_dir.is_dir():
                            dataset = dataset_dir.name
                            metrics = self.load_metrics(dataset_dir)
                            if metrics:
                                self.results['ExperimentA'][method][dataset] = metrics

        # Experiment B
        exp_b_path = self.output_dir / "ExperimentB"
        if exp_b_path.exists():
            for method in ['AdaptiveTrain', 'FreezeEarly', 'AlwaysFrozen', 'ExplicitReg']:
                method_path = exp_b_path / method
                if method_path.exists():
                    self.results['ExperimentB'][method] = {}
                    for dataset_dir in method_path.iterdir():
                        if dataset_dir.is_dir():
                            dataset = dataset_dir.name
                            metrics = self.load_metrics(dataset_dir)
                            if metrics:
                                self.results['ExperimentB'][method][dataset] = metrics

        print(f"Collected results for {len(self.results['ExperimentA'])} methods in Exp A")
        print(f"Collected results for {len(self.results['ExperimentB'])} methods in Exp B")

    def plot_experiment_a(self):
        """Plot Experiment A results."""
        print("\nGenerating plots for Experiment A...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment A: Parameter-Count Buffer Hypothesis', fontsize=16, fontweight='bold')

        datasets = ['eurosat', 'imagenet', 'caltech101']

        for idx, dataset in enumerate(datasets):
            # Validation accuracy
            ax1 = axes[0, idx]
            for method in ['ParamMatched', 'Adaptive']:
                if method in self.results['ExperimentA'] and dataset in self.results['ExperimentA'][method]:
                    metrics = self.results['ExperimentA'][method][dataset]
                    if 'val_acc' in metrics:
                        epochs = metrics.get('epochs', list(range(len(metrics['val_acc']))))
                        val_acc = [x for x in metrics['val_acc'] if not np.isnan(x)]
                        epochs = epochs[:len(val_acc)]
                        ax1.plot(epochs, val_acc, label=method, linewidth=2, marker='o', markersize=4)

            ax1.set_title(f'{dataset.upper()} - Validation Accuracy', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Training loss
            ax2 = axes[1, idx]
            for method in ['ParamMatched', 'Adaptive']:
                if method in self.results['ExperimentA'] and dataset in self.results['ExperimentA'][method]:
                    metrics = self.results['ExperimentA'][method][dataset]
                    if 'train_loss' in metrics:
                        epochs = metrics.get('epochs', list(range(len(metrics['train_loss']))))
                        train_loss = [x for x in metrics['train_loss'] if not np.isnan(x)]
                        epochs = epochs[:len(train_loss)]
                        ax2.plot(epochs, train_loss, label=method, linewidth=2, marker='s', markersize=4)

            ax2.set_title(f'{dataset.upper()} - Training Loss', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "experiment_a_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Experiment A plot to {save_path}")
        plt.close()

    def plot_experiment_b(self):
        """Plot Experiment B results."""
        print("\nGenerating plots for Experiment B...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment B: Implicit Regularization Hypothesis', fontsize=16, fontweight='bold')

        datasets = ['eurosat', 'imagenet', 'caltech101']
        methods = ['AdaptiveTrain', 'FreezeEarly', 'AlwaysFrozen', 'ExplicitReg']
        colors = {'AdaptiveTrain': 'blue', 'FreezeEarly': 'orange',
                  'AlwaysFrozen': 'green', 'ExplicitReg': 'red'}

        for idx, dataset in enumerate(datasets):
            # Validation accuracy
            ax1 = axes[0, idx]
            for method in methods:
                if method in self.results['ExperimentB'] and dataset in self.results['ExperimentB'][method]:
                    metrics = self.results['ExperimentB'][method][dataset]
                    if 'val_acc' in metrics:
                        epochs = metrics.get('epochs', list(range(len(metrics['val_acc']))))
                        val_acc = [x for x in metrics['val_acc'] if not np.isnan(x)]
                        epochs = epochs[:len(val_acc)]
                        label = method.replace('Adaptive', 'A-')
                        ax1.plot(epochs, val_acc, label=label, linewidth=2,
                                 color=colors[method], marker='o', markersize=3)

            ax1.set_title(f'{dataset.upper()} - Validation Accuracy', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Regularization loss
            ax2 = axes[1, idx]
            for method in methods:
                if method in self.results['ExperimentB'] and dataset in self.results['ExperimentB'][method]:
                    metrics = self.results['ExperimentB'][method][dataset]
                    if 'reg_loss' in metrics:
                        epochs = metrics.get('epochs', list(range(len(metrics['reg_loss']))))
                        reg_loss = [x for x in metrics['reg_loss'] if not np.isnan(x)]
                        epochs = epochs[:len(reg_loss)]
                        label = method.replace('Adaptive', 'A-')
                        ax2.plot(epochs, reg_loss, label=label, linewidth=2,
                                 color=colors[method], marker='s', markersize=3)

            ax2.set_title(f'{dataset.upper()} - Regularization Loss', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "experiment_b_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Experiment B plot to {save_path}")
        plt.close()

    def generate_summary_table(self):
        """Generate summary tables."""
        print("\nGenerating summary tables...")

        # Experiment A summary
        print("\n" + "=" * 80)
        print("EXPERIMENT A: Parameter-Count Buffer Hypothesis")
        print("=" * 80)

        df_a_data = []
        for method in ['ParamMatched', 'Adaptive']:
            if method not in self.results['ExperimentA']:
                continue
            for dataset in ['eurosat', 'imagenet', 'caltech101']:
                if dataset not in self.results['ExperimentA'][method]:
                    continue
                metrics = self.results['ExperimentA'][method][dataset]
                if 'val_acc' in metrics:
                    val_accs = [x for x in metrics['val_acc'] if not np.isnan(x)]
                    if val_accs:
                        best_acc = max(val_accs)
                        final_acc = val_accs[-1]
                        df_a_data.append({
                            'Method': method,
                            'Dataset': dataset.upper(),
                            'Best Val Acc': f"{best_acc:.2f}",
                            'Final Val Acc': f"{final_acc:.2f}"
                        })

        if df_a_data:
            df_a = pd.DataFrame(df_a_data)
            print("\n", df_a.to_string(index=False))
            df_a.to_csv(self.output_dir / "experiment_a_summary.csv", index=False)

        # Experiment B summary
        print("\n" + "=" * 80)
        print("EXPERIMENT B: Implicit Regularization Hypothesis")
        print("=" * 80)

        df_b_data = []
        methods = ['AdaptiveTrain', 'FreezeEarly', 'AlwaysFrozen', 'ExplicitReg']
        for method in methods:
            if method not in self.results['ExperimentB']:
                continue
            for dataset in ['eurosat', 'imagenet', 'caltech101']:
                if dataset not in self.results['ExperimentB'][method]:
                    continue
                metrics = self.results['ExperimentB'][method][dataset]
                if 'val_acc' in metrics:
                    val_accs = [x for x in metrics['val_acc'] if not np.isnan(x)]
                    if val_accs:
                        best_acc = max(val_accs)
                        final_acc = val_accs[-1]
                        df_b_data.append({
                            'Method': method,
                            'Dataset': dataset.upper(),
                            'Best Val Acc': f"{best_acc:.2f}",
                            'Final Val Acc': f"{final_acc:.2f}"
                        })

        if df_b_data:
            df_b = pd.DataFrame(df_b_data)
            print("\n", df_b.to_string(index=False))
            df_b.to_csv(self.output_dir / "experiment_b_summary.csv", index=False)

    def analyze_hypothesis_a(self):
        """Hypothesis A: parameter-count buffer."""
        print("\n" + "=" * 80)
        print("HYPOTHESIS A ANALYSIS: Parameter-Count Buffer")
        print("=" * 80)

        for dataset in ['eurosat', 'imagenet', 'caltech101']:
            if 'ParamMatched' not in self.results['ExperimentA']:
                continue
            if 'Adaptive' not in self.results['ExperimentA']:
                continue
            if dataset not in self.results['ExperimentA']['ParamMatched']:
                continue
            if dataset not in self.results['ExperimentA']['Adaptive']:
                continue

            pm_metrics = self.results['ExperimentA']['ParamMatched'][dataset]
            adp_metrics = self.results['ExperimentA']['Adaptive'][dataset]

            if 'val_acc' in pm_metrics and 'val_acc' in adp_metrics:
                pm_accs = [x for x in pm_metrics['val_acc'] if not np.isnan(x)]
                adp_accs = [x for x in adp_metrics['val_acc'] if not np.isnan(x)]

                if pm_accs and adp_accs:
                    pm_best = max(pm_accs)
                    adp_best = max(adp_accs)
                    diff = pm_best - adp_best

                    print(f"\n{dataset.upper()}:")
                    print(f"  ParamMatched best: {pm_best:.2f}%")
                    print(f"  Adaptive best:     {adp_best:.2f}%")
                    print(f"  Difference:        {diff:+.2f}%")

                    if abs(diff) < 1.0:
                        print(f"  → Conclusion: Gap <1%; supports parameter-count buffer hypothesis.")
                    else:
                        print(f"  → Conclusion: Gap ≥1%; does not support parameter-count buffer hypothesis.")

    def analyze_hypothesis_b(self):
        """Hypothesis B: implicit regularization."""
        print("\n" + "=" * 80)
        print("HYPOTHESIS B ANALYSIS: Implicit Regularization")
        print("=" * 80)

        methods = ['AdaptiveTrain', 'FreezeEarly', 'AlwaysFrozen', 'ExplicitReg']

        for dataset in ['eurosat', 'imagenet', 'caltech101']:
            print(f"\n{dataset.upper()}:")

            results = {}
            for method in methods:
                if method not in self.results['ExperimentB']:
                    continue
                if dataset not in self.results['ExperimentB'][method]:
                    continue
                metrics = self.results['ExperimentB'][method][dataset]
                if 'val_acc' in metrics:
                    val_accs = [x for x in metrics['val_acc'] if not np.isnan(x)]
                    if val_accs:
                        results[method] = max(val_accs)

            if len(results) >= 2:
                for method, acc in results.items():
                    print(f"  {method:20s}: {acc:.2f}%")

                # FreezeEarly vs train
                if 'FreezeEarly' in results and 'AdaptiveTrain' in results:
                    diff = results['FreezeEarly'] - results['AdaptiveTrain']
                    print(f"\n  FreezeEarly vs Train: {diff:+.2f}%")
                    if diff > 0.5:
                        print(f"  → Early freezing improves accuracy; supports implicit regularization hypothesis.")

                # AlwaysFrozen vs train
                if 'AlwaysFrozen' in results and 'AdaptiveTrain' in results:
                    diff = results['AlwaysFrozen'] - results['AdaptiveTrain']
                    print(f"  AlwaysFrozen vs Train: {diff:+.2f}%")

                # ExplicitReg vs train
                if 'ExplicitReg' in results and 'AdaptiveTrain' in results:
                    diff = results['ExplicitReg'] - results['AdaptiveTrain']
                    print(f"  ExplicitReg vs Train: {diff:+.2f}%")

    def run_full_analysis(self):
        """Run full analysis pipeline."""
        self.collect_results()
        self.plot_experiment_a()
        self.plot_experiment_b()
        self.generate_summary_table()
        self.analyze_hypothesis_a()
        self.analyze_hypothesis_b()

        print("\n" + "=" * 80)
        print("Analysis complete! Results saved to:", self.output_dir)
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze hypothesis experiments")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory containing experiment results")
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.output_dir)
    analyzer.run_full_analysis()
