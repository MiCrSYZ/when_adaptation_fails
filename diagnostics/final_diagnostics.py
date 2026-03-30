"""
Final diagnostic experiments for Adaptive BiDir MaPLe.
Experiment A: dynamic gradient scaling on gates.
Experiment B: entropy regularization on gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from collections import defaultdict
from tqdm import tqdm


class GradientScaler:
    """Experiment A: scale gate gradients inversely with sigmoid derivative magnitude."""

    @staticmethod
    def compute_scale_factor(gate_values, epsilon=1e-6):
        """
        Scale factor: alpha_i = 1 / (sigma(g_i)(1-sigma(g_i)) + eps).

        Args:
            gate_values: Gate logits (before sigmoid).
            epsilon: Floor for numerical stability.

        Returns:
            Per-element scale factors.
        """
        sigmoid_g = torch.sigmoid(gate_values)
        variance_term = sigmoid_g * (1 - sigmoid_g)
        scale_factors = 1.0 / (variance_term + epsilon)
        return scale_factors

    @staticmethod
    def apply_gradient_scaling(model, scale_length_gates=True, scale_depth_weights=True):
        """
        Multiply gate parameter gradients by computed scale factors (clamped).

        Args:
            model: Model with prompt_learner.
            scale_length_gates: Apply to length_gates.
            scale_depth_weights: Apply to depth_weights.
        """
        prompt_learner = model.prompt_learner if hasattr(model, 'prompt_learner') else model.module.prompt_learner

        if scale_length_gates and hasattr(prompt_learner, 'length_gates'):
            if isinstance(prompt_learner.length_gates, nn.Parameter) and prompt_learner.length_gates.grad is not None:
                scale_factors = GradientScaler.compute_scale_factor(prompt_learner.length_gates.data)
                scale_factors = torch.clamp(scale_factors, max=10.0)
                prompt_learner.length_gates.grad *= scale_factors

        if scale_depth_weights and hasattr(prompt_learner, 'depth_weights'):
            if isinstance(prompt_learner.depth_weights, nn.Parameter) and prompt_learner.depth_weights.grad is not None:
                scale_factors = GradientScaler.compute_scale_factor(prompt_learner.depth_weights.data)
                scale_factors = torch.clamp(scale_factors, max=10.0)
                prompt_learner.depth_weights.grad *= scale_factors


class EntropyRegularizer:
    """Experiment B: negative entropy penalty to push gates toward 0 or 1."""

    @staticmethod
    def compute_entropy_loss(gate_values, lambda_entropy=0.01, epsilon=1e-8):
        """
        L_reg = -lambda * sum[ sigma log sigma + (1-sigma) log(1-sigma) ].

        Encourages gates away from 0.5 (lower entropy).

        Args:
            gate_values: Gate logits (before sigmoid).
            lambda_entropy: Weight.
            epsilon: Log stability.

        Returns:
            entropy_loss tensor, scalar entropy for logging.
        """
        sigmoid_g = torch.sigmoid(gate_values)

        log_sigmoid_g = torch.log(sigmoid_g + epsilon)
        log_one_minus_sigmoid_g = torch.log(1 - sigmoid_g + epsilon)

        entropy = -(sigmoid_g * log_sigmoid_g + (1 - sigmoid_g) * log_one_minus_sigmoid_g)
        entropy_value = entropy.mean().item()

        entropy_loss = -lambda_entropy * entropy.sum()

        return entropy_loss, entropy_value


def run_experiment_A(trainer, num_steps=50, compare_baseline=True):
    """
    Experiment A: compare gate/prompt grad norms with and without scaling.

    Logs effective length and mean depth probability over steps.
    """
    print("\n" + "=" * 80)
    print("Experiment A: gate gradient scaling")
    print("=" * 80)

    model = trainer.model
    train_loader = trainer.train_loader_x
    device = trainer.device

    results = {
        'baseline': defaultdict(list) if compare_baseline else None,
        'scaled': defaultdict(list)
    }

    if compare_baseline:
        print("\n[Phase 1] Baseline (no scaling)...")
        model.train()

        for step, batch in enumerate(tqdm(train_loader, total=num_steps)):
            if step >= num_steps:
                break

            image, label = batch["img"].to(device), batch["label"].to(device)

            total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)

            trainer.optim.zero_grad()
            total_loss.backward()

            prompt_learner = model.prompt_learner if hasattr(model, 'prompt_learner') else model.module.prompt_learner

            gate_grad_norm = 0.0
            prompt_grad_norm = 0.0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'length_gates' in name or 'depth_weights' in name:
                        gate_grad_norm += param.grad.norm().item() ** 2
                    elif 'ctx' in name or 'compound_prompts' in name:
                        prompt_grad_norm += param.grad.norm().item() ** 2

            gate_grad_norm = np.sqrt(gate_grad_norm)
            prompt_grad_norm = np.sqrt(prompt_grad_norm)

            results['baseline']['gate_grad_norm'].append(gate_grad_norm)
            results['baseline']['prompt_grad_norm'].append(prompt_grad_norm)
            results['baseline']['grad_ratio'].append(gate_grad_norm / (prompt_grad_norm + 1e-8))

            eff_length = prompt_learner.get_effective_length()
            depth_probs_val = prompt_learner.get_depth_probabilities()

            results['baseline']['effective_length'].append(float(eff_length))
            results['baseline']['avg_depth_prob'].append(float(depth_probs_val.mean().item()))

            trainer.optim.step()

    print("\n[Phase 2] With gradient scaling...")
    model.train()

    for step, batch in enumerate(tqdm(train_loader, total=num_steps)):
        if step >= num_steps:
            break

        image, label = batch["img"].to(device), batch["label"].to(device)

        total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)

        trainer.optim.zero_grad()
        total_loss.backward()

        GradientScaler.apply_gradient_scaling(model, scale_length_gates=True, scale_depth_weights=True)

        prompt_learner = model.prompt_learner if hasattr(model, 'prompt_learner') else model.module.prompt_learner

        gate_grad_norm = 0.0
        prompt_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'length_gates' in name or 'depth_weights' in name:
                    gate_grad_norm += param.grad.norm().item() ** 2
                elif 'ctx' in name or 'compound_prompts' in name:
                    prompt_grad_norm += param.grad.norm().item() ** 2

        gate_grad_norm = np.sqrt(gate_grad_norm)
        prompt_grad_norm = np.sqrt(prompt_grad_norm)

        results['scaled']['gate_grad_norm'].append(gate_grad_norm)
        results['scaled']['prompt_grad_norm'].append(prompt_grad_norm)
        results['scaled']['grad_ratio'].append(gate_grad_norm / (prompt_grad_norm + 1e-8))

        eff_length = prompt_learner.get_effective_length()
        depth_probs_val = prompt_learner.get_depth_probabilities()

        results['scaled']['effective_length'].append(float(eff_length))
        results['scaled']['avg_depth_prob'].append(float(depth_probs_val.mean().item()))

        trainer.optim.step()

    print("\n" + "=" * 80)
    print("Experiment A summary")
    print("=" * 80)

    if compare_baseline:
        print("\n[Baseline vs scaled]")
        print(f"\n1. Gate grad norm:")
        print(f"   Baseline: {np.mean(results['baseline']['gate_grad_norm']):.6f} ± {np.std(results['baseline']['gate_grad_norm']):.6f}")
        print(f"   Scaled:   {np.mean(results['scaled']['gate_grad_norm']):.6f} ± {np.std(results['scaled']['gate_grad_norm']):.6f}")
        print(f"   Ratio: {np.mean(results['scaled']['gate_grad_norm']) / np.mean(results['baseline']['gate_grad_norm']):.2f}x")

        print(f"\n2. Prompt grad norm:")
        print(f"   Baseline: {np.mean(results['baseline']['prompt_grad_norm']):.6f}")
        print(f"   Scaled:   {np.mean(results['scaled']['prompt_grad_norm']):.6f}")

        print(f"\n3. Gate/prompt ratio:")
        print(f"   Baseline: {np.mean(results['baseline']['grad_ratio']):.6f}")
        print(f"   Scaled:   {np.mean(results['scaled']['grad_ratio']):.6f}")
        print(f"   Improvement: {np.mean(results['scaled']['grad_ratio']) / np.mean(results['baseline']['grad_ratio']):.2f}x")

        print(f"\n4. Effective length:")
        print(f"   Baseline: {results['baseline']['effective_length'][0]:.2f} -> {results['baseline']['effective_length'][-1]:.2f} (delta={results['baseline']['effective_length'][-1] - results['baseline']['effective_length'][0]:.2f})")
        print(f"   Scaled:   {results['scaled']['effective_length'][0]:.2f} -> {results['scaled']['effective_length'][-1]:.2f} (delta={results['scaled']['effective_length'][-1] - results['scaled']['effective_length'][0]:.2f})")

        print(f"\n5. Mean depth probability:")
        print(f"   Baseline: {results['baseline']['avg_depth_prob'][0]:.4f} -> {results['baseline']['avg_depth_prob'][-1]:.4f} (delta={results['baseline']['avg_depth_prob'][-1] - results['baseline']['avg_depth_prob'][0]:.4f})")
        print(f"   Scaled:   {results['scaled']['avg_depth_prob'][0]:.4f} -> {results['scaled']['avg_depth_prob'][-1]:.4f} (delta={results['scaled']['avg_depth_prob'][-1] - results['scaled']['avg_depth_prob'][0]:.4f})")
    else:
        print("\n[Scaled only]")
        print(f"Gate grad norm: {np.mean(results['scaled']['gate_grad_norm']):.6f}")
        print(f"Prompt grad norm: {np.mean(results['scaled']['prompt_grad_norm']):.6f}")
        print(f"Grad ratio: {np.mean(results['scaled']['grad_ratio']):.6f}")

    _plot_experiment_A(results, trainer.viz_dir, compare_baseline)

    return results


def run_experiment_B(trainer, num_steps=50, lambda_values=[0.0, 0.01, 0.05, 0.1]):
    """
    Experiment B: sweep entropy regularization weight on gates.
    """
    print("\n" + "=" * 80)
    print("Experiment B: entropy regularization")
    print("=" * 80)

    model = trainer.model
    train_loader = trainer.train_loader_x
    device = trainer.device

    all_results = {}

    for lambda_entropy in lambda_values:
        print(f"\n[Testing lambda={lambda_entropy}]")

        results = defaultdict(list)
        model.train()

        for step, batch in enumerate(tqdm(train_loader, total=num_steps)):
            if step >= num_steps:
                break

            image, label = batch["img"].to(device), batch["label"].to(device)

            total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)

            prompt_learner = model.prompt_learner if hasattr(model, 'prompt_learner') else model.module.prompt_learner

            entropy_loss_length = 0.0
            entropy_loss_depth = 0.0
            entropy_val_length = 0.0
            entropy_val_depth = 0.0

            if lambda_entropy > 0:
                if isinstance(prompt_learner.length_gates, nn.Parameter):
                    ent_loss, ent_val = EntropyRegularizer.compute_entropy_loss(
                        prompt_learner.length_gates, lambda_entropy
                    )
                    entropy_loss_length = ent_loss
                    entropy_val_length = ent_val

                if isinstance(prompt_learner.depth_weights, nn.Parameter):
                    ent_loss, ent_val = EntropyRegularizer.compute_entropy_loss(
                        prompt_learner.depth_weights, lambda_entropy
                    )
                    entropy_loss_depth = ent_loss
                    entropy_val_depth = ent_val

            total_loss_with_entropy = total_loss + entropy_loss_length + entropy_loss_depth

            trainer.optim.zero_grad()
            total_loss_with_entropy.backward()

            gate_grad_norm = 0.0
            prompt_grad_norm = 0.0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'length_gates' in name or 'depth_weights' in name:
                        gate_grad_norm += param.grad.norm().item() ** 2
                    elif 'ctx' in name or 'compound_prompts' in name:
                        prompt_grad_norm += param.grad.norm().item() ** 2

            gate_grad_norm = np.sqrt(gate_grad_norm)
            prompt_grad_norm = np.sqrt(prompt_grad_norm)

            results['gate_grad_norm'].append(gate_grad_norm)
            results['prompt_grad_norm'].append(prompt_grad_norm)
            results['grad_ratio'].append(gate_grad_norm / (prompt_grad_norm + 1e-8))
            results['entropy_length'].append(entropy_val_length)
            results['entropy_depth'].append(entropy_val_depth)

            eff_length = prompt_learner.get_effective_length()
            depth_probs_val = prompt_learner.get_depth_probabilities()

            results['effective_length'].append(float(eff_length))
            results['avg_depth_prob'].append(float(depth_probs_val.mean().item()))

            trainer.optim.step()

        all_results[lambda_entropy] = results

    print("\n" + "=" * 80)
    print("Experiment B summary")
    print("=" * 80)

    for lambda_entropy in lambda_values:
        results = all_results[lambda_entropy]
        print(f"\n[lambda={lambda_entropy}]")
        print(f"  Gate grad norm: {np.mean(results['gate_grad_norm']):.6f}")
        print(f"  Prompt grad norm: {np.mean(results['prompt_grad_norm']):.6f}")
        print(f"  Grad ratio: {np.mean(results['grad_ratio']):.6f}")
        print(f"  Length entropy: {np.mean(results['entropy_length']):.4f}")
        print(f"  Depth entropy: {np.mean(results['entropy_depth']):.4f}")
        print(f"  Effective length: {results['effective_length'][0]:.2f} -> {results['effective_length'][-1]:.2f}")
        print(f"  Depth prob: {results['avg_depth_prob'][0]:.4f} -> {results['avg_depth_prob'][-1]:.4f}")

    _plot_experiment_B(all_results, trainer.viz_dir)

    return all_results


def _plot_experiment_A(results, viz_dir, compare_baseline):
    """Plot Experiment A curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment A: Gradient Scaling', fontsize=16, fontweight='bold')

    if compare_baseline:
        axes[0, 0].plot(results['baseline']['gate_grad_norm'], label='Baseline', alpha=0.7)
        axes[0, 0].plot(results['scaled']['gate_grad_norm'], label='Scaled', alpha=0.7)
        axes[0, 0].set_title('Gate Gradient Norm')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(results['baseline']['prompt_grad_norm'], label='Baseline', alpha=0.7)
        axes[0, 1].plot(results['scaled']['prompt_grad_norm'], label='Scaled', alpha=0.7)
        axes[0, 1].set_title('Prompt Gradient Norm')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Norm')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(results['baseline']['grad_ratio'], label='Baseline', alpha=0.7)
        axes[0, 2].plot(results['scaled']['grad_ratio'], label='Scaled', alpha=0.7)
        axes[0, 2].set_title('Gradient Ratio (Gate/Prompt)')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Ratio')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(results['baseline']['effective_length'], label='Baseline', alpha=0.7)
        axes[1, 0].plot(results['scaled']['effective_length'], label='Scaled', alpha=0.7)
        axes[1, 0].set_title('Effective Length')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(results['baseline']['avg_depth_prob'], label='Baseline', alpha=0.7)
        axes[1, 1].plot(results['scaled']['avg_depth_prob'], label='Scaled', alpha=0.7)
        axes[1, 1].set_title('Average Depth Probability')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        metrics = ['gate_grad_norm', 'prompt_grad_norm', 'grad_ratio', 'effective_length', 'avg_depth_prob']
        baseline_means = [np.mean(results['baseline'][m]) for m in metrics]
        scaled_means = [np.mean(results['scaled'][m]) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        axes[1, 2].bar(x - width / 2, baseline_means, width, label='Baseline', alpha=0.7)
        axes[1, 2].bar(x + width / 2, scaled_means, width, label='Scaled', alpha=0.7)
        axes[1, 2].set_title('Average Metrics Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(['Gate\nGrad', 'Prompt\nGrad', 'Grad\nRatio', 'Eff\nLength', 'Depth\nProb'], fontsize=8)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
    else:
        axes[0, 0].plot(results['scaled']['gate_grad_norm'])
        axes[0, 0].set_title('Gate Gradient Norm (Scaled)')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(results['scaled']['prompt_grad_norm'])
        axes[0, 1].set_title('Prompt Gradient Norm (Scaled)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(results['scaled']['grad_ratio'])
        axes[0, 2].set_title('Gradient Ratio (Scaled)')
        axes[0, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = osp.join(viz_dir, 'experiment_A_gradient_scaling.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


def _plot_experiment_B(all_results, viz_dir):
    """Plot Experiment B curves."""
    lambda_values = list(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment B: Entropy Regularization', fontsize=16, fontweight='bold')

    for lambda_val in lambda_values:
        results = all_results[lambda_val]
        label = f'lambda={lambda_val}'

        axes[0, 0].plot(results['gate_grad_norm'], label=label, alpha=0.7)
        axes[0, 1].plot(results['grad_ratio'], label=label, alpha=0.7)
        axes[0, 2].plot(results['entropy_length'], label=label, alpha=0.7)
        axes[1, 0].plot(results['effective_length'], label=label, alpha=0.7)
        axes[1, 1].plot(results['avg_depth_prob'], label=label, alpha=0.7)

    axes[0, 0].set_title('Gate Gradient Norm')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Norm')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Gradient Ratio (Gate/Prompt)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_title('Length Gates Entropy')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_title('Effective Length')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Average Depth Probability')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    metrics = ['gate_grad_norm', 'grad_ratio', 'entropy_length', 'effective_length', 'avg_depth_prob']
    metric_labels = ['Gate\nGrad', 'Grad\nRatio', 'Length\nEntropy', 'Eff\nLength', 'Depth\nProb']

    x = np.arange(len(metrics))
    width = 0.8 / len(lambda_values)

    for i, lambda_val in enumerate(lambda_values):
        results = all_results[lambda_val]
        means = [np.mean(results[m]) for m in metrics]
        offset = (i - len(lambda_values) / 2 + 0.5) * width
        axes[1, 2].bar(x + offset, means, width, label=f'lambda={lambda_val}', alpha=0.7)

    axes[1, 2].set_title('Average Metrics by lambda')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metric_labels, fontsize=8)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = osp.join(viz_dir, 'experiment_B_entropy_regularization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


def run_all_final_diagnostics(trainer, epoch):
    """Run Experiments A and B and print recommendations."""
    print(f"\n{'=' * 80}")
    print(f"Final diagnostics (epoch {epoch})")
    print(f"{'=' * 80}\n")

    try:
        improvement = 1.0
        results_A = run_experiment_A(trainer, num_steps=50, compare_baseline=True)

        results_B = run_experiment_B(trainer, num_steps=50, lambda_values=[0.0, 0.01, 0.05, 0.1])

        print("\n" + "=" * 80)
        print("Summary and recommendations")
        print("=" * 80)

        if results_A['baseline'] is not None:
            baseline_ratio = np.mean(results_A['baseline']['grad_ratio'])
            scaled_ratio = np.mean(results_A['scaled']['grad_ratio'])
            improvement = scaled_ratio / baseline_ratio

            baseline_length_change = abs(results_A['baseline']['effective_length'][-1] - results_A['baseline']['effective_length'][0])
            scaled_length_change = abs(results_A['scaled']['effective_length'][-1] - results_A['scaled']['effective_length'][0])

            print("\n[Experiment A - gradient scaling]")
            if improvement > 5 and scaled_length_change > baseline_length_change:
                print(f"OK: strong effect (grad ratio x{improvement:.1f})")
                print(f"  Effective length change: {baseline_length_change:.2f} -> {scaled_length_change:.2f}")
                print("  Consider enabling scaling in training (max_scale=10)")
            elif improvement > 2:
                print(f"Partial: grad ratio x{improvement:.1f}")
                print(f"  Limited length change ({scaled_length_change:.2f})")
                print("  Try combining with entropy regularization")
            else:
                print(f"Weak: grad ratio x{improvement:.1f}")
                print("  Gradients may not be the main bottleneck; review losses")

        print("\n[Experiment B - entropy regularization]")
        best_lambda = None
        best_score = -float('inf')

        for lambda_val in [0.01, 0.05, 0.1]:
            if lambda_val in results_B:
                results = results_B[lambda_val]
                baseline_results = results_B[0.0]

                grad_improvement = np.mean(results['grad_ratio']) / np.mean(baseline_results['grad_ratio'])
                length_change = abs(results['effective_length'][-1] - results['effective_length'][0])
                entropy_reduction = np.mean(baseline_results['entropy_length']) - np.mean(results['entropy_length'])

                score = grad_improvement * length_change * (1 + entropy_reduction)

                if score > best_score:
                    best_score = score
                    best_lambda = lambda_val

        if best_lambda is not None:
            print(f"Best lambda: {best_lambda}")
            results = results_B[best_lambda]
            baseline_results = results_B[0.0]

            print(f"  Grad ratio: {np.mean(baseline_results['grad_ratio']):.6f} -> {np.mean(results['grad_ratio']):.6f}")
            print(f"  Length change: {abs(results['effective_length'][-1] - results['effective_length'][0]):.2f}")
            print(f"  Entropy: {np.mean(baseline_results['entropy_length']):.4f} -> {np.mean(results['entropy_length']):.4f}")
            print(f"  Try entropy weight lambda={best_lambda}")
        else:
            print("Entropy regularization did not clearly help in this sweep")
            print("  Try stronger regularization or different losses")

        print("\n[Overall]")
        if improvement > 5 and best_lambda is not None:
            print("1. Combine gradient scaling and entropy regularization")
            print("   - Scaling max_scale=10")
            print(f"   - Entropy lambda={best_lambda}")
            print("2. Consider lower prompt LR and higher gate LR")
            print("3. Longer warmup for gates")
        elif improvement > 2 or best_lambda is not None:
            print("1. Try one intervention first")
            if improvement > best_score / 10:
                print("   Prefer: gradient scaling")
            else:
                print(f"   Prefer: entropy regularization (lambda={best_lambda})")
            print("2. Or use fixed MaPLe")
        else:
            print("Warning: adaptation may be fundamentally hard here")
            print("1. Possible causes:")
            print("   - Flat loss landscape for gates")
            print("   - Optimization conflict between gates and prompts")
            print("   - Extra DOF not useful for this task")
            print("2. Options:")
            print("   - Fall back to fixed MaPLe")
            print("   - Redesign gating (e.g. RL or discrete search)")

        return {
            'experiment_A': results_A,
            'experiment_B': results_B,
            'recommendations': {
                'use_gradient_scaling': improvement > 3,
                'use_entropy_reg': best_lambda is not None,
                'best_lambda': best_lambda,
                'gradient_improvement': improvement if results_A['baseline'] is not None else None
            }
        }

    except Exception as e:
        print(f"\nDiagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        return None
