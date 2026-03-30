"""
Diagnostic experiments for why AdaptiveBiDirMaPLe may fail to learn useful gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os


class DiagnosticTracker:
    """Track key diagnostics during training."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics = defaultdict(list)
        self.gradient_stats = defaultdict(lambda: defaultdict(list))

    def record_gradients(self, model, step):
        """Record per-parameter gradient statistics."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                self.gradient_stats[step][name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'norm': grad.norm().item(),
                    'zeros': (grad.abs() < 1e-8).sum().item() / grad.numel()
                }

    def save(self):
        with open(os.path.join(self.save_dir, 'diagnostics.json'), 'w') as f:
            json.dump({
                'metrics': dict(self.metrics),
                'gradient_stats': {str(k): v for k, v in self.gradient_stats.items()}
            }, f, indent=2)


class ExperimentA_GradientFlow:
    """
    Experiment A: gradient decay along prompt→gate.

    Expected:
    - Gate grad << prompt grad (often 2–3 orders of magnitude)
    - Gate grads near FP16 truncation
    - Longer paths → stronger decay
    """

    @staticmethod
    def hook_gradient_flow(model, tracker):
        """Register backward hooks to trace gradients."""
        hooks = []

        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    tracker.gradient_stats['hooks'][name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.abs().max().item(),
                        'min_nonzero': grad[grad.abs() > 0].abs().min().item() if (grad.abs() > 0).any() else 0
                    }
                return grad

            return hook

        # Trace gradients on key parameters
        prompt_learner = model.prompt_learner

        if hasattr(prompt_learner, 'ctx'):
            h = prompt_learner.ctx.register_hook(make_hook('ctx'))
            hooks.append(h)

        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            h = prompt_learner.length_gates.register_hook(make_hook('length_gates'))
            hooks.append(h)

        if hasattr(prompt_learner, 'depth_weights') and isinstance(prompt_learner.depth_weights, nn.Parameter):
            h = prompt_learner.depth_weights.register_hook(make_hook('depth_weights'))
            hooks.append(h)

        return hooks

    @staticmethod
    def test_gradient_magnitude_ratio(model, image, label):
        """
        Test 1: measure ||∇_gate L|| / ||∇_prompt L||.
        """
        model.train()
        loss, _, _, _, _, _ = model(image, label)

        model.zero_grad()
        loss.backward()

        prompt_grad_norm = 0
        gate_grad_norm = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'ctx' in name or 'compound_prompts' in name:
                    prompt_grad_norm += param.grad.norm().item() ** 2
                elif 'length_gates' in name or 'depth_weights' in name:
                    gate_grad_norm += param.grad.norm().item() ** 2

        prompt_grad_norm = np.sqrt(prompt_grad_norm)
        gate_grad_norm = np.sqrt(gate_grad_norm)

        ratio = gate_grad_norm / (prompt_grad_norm + 1e-8)

        return {
            'prompt_grad_norm': prompt_grad_norm,
            'gate_grad_norm': gate_grad_norm,
            'ratio': ratio,
            'magnitude_gap': np.log10(prompt_grad_norm / (gate_grad_norm + 1e-10))
        }

    @staticmethod
    def test_fp16_truncation(model, image, label):
        """
        Test 2: gradient underflow in FP16; compare FP32 vs FP16 gate gradients.
        """
        results = {}

        # Save original parameter state
        original_state = {}
        for name, param in model.named_parameters():
            original_state[name] = {
                'data': param.data.clone(),
                'dtype': param.dtype
            }

        for dtype_name, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
            try:
                model.train()

                # Cast parameters/buffers to target dtype
                if dtype == torch.float16:
                    model.half()
                    image_input = image.half()
                    label_input = label
                else:
                    model.float()
                    image_input = image.float()
                    label_input = label

                # Forward
                loss, _, _, _, _, _ = model(image_input, label_input)

                # Backward
                model.zero_grad()
                loss.backward()

                # Collect gate grads (float32 for analysis)
                gate_grads = []
                for name, param in model.named_parameters():
                    if ('length_gates' in name or 'depth_weights' in name) and param.grad is not None:
                        gate_grads.append(param.grad.clone().float())

                if gate_grads:
                    gate_grads = torch.cat([g.flatten() for g in gate_grads])
                    results[dtype_name] = {
                        'mean': float(gate_grads.mean().item()),
                        'std': float(gate_grads.std().item()),
                        'min_nonzero': float(gate_grads[gate_grads.abs() > 0].abs().min().item()) if (
                                    gate_grads.abs() > 0).any() else 0.0,
                        'zeros_ratio': float((gate_grads.abs() < 1e-8).sum().item() / gate_grads.numel())
                    }
                else:
                    results[dtype_name] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min_nonzero': 0.0,
                        'zeros_ratio': 1.0
                    }

            except Exception as e:
                print(f"Warning: FP16 truncation test failed for {dtype_name}: {e}")
                results[dtype_name] = {
                    'error': str(e),
                    'mean': 0.0,
                    'std': 0.0,
                    'min_nonzero': 0.0,
                    'zeros_ratio': 1.0
                }
            finally:
                # Restore original state
                for name, param in model.named_parameters():
                    if name in original_state:
                        param.data = original_state[name]['data']
                        if original_state[name]['dtype'] == torch.float16:
                            param.data = param.data.half()
                        else:
                            param.data = param.data.float()

                # Restore model to original dtype
                first_param_dtype = original_state[list(original_state.keys())[0]]['dtype']
                if first_param_dtype == torch.float16:
                    model.half()
                else:
                    model.float()

        # Precision gap between fp32 and fp16
        if 'fp32' in results and 'fp16' in results and 'error' not in results['fp32'] and 'error' not in results[
            'fp16']:
            results['precision_loss'] = {
                'mean_diff': abs(results['fp32']['mean'] - results['fp16']['mean']),
                'zeros_increase': results['fp16']['zeros_ratio'] - results['fp32']['zeros_ratio']
            }

        return results

    @staticmethod
    def test_gradient_chain_length(model, image, label):
        """
        Test 3: effect of backprop depth; shallow vs deep gate gradients.
        """
        model.train()
        loss, _, _, _, _, _ = model(image, label)

        model.zero_grad()
        loss.backward()

        # Per-depth gate gradients
        depth_grads = {}

        prompt_learner = model.prompt_learner
        if hasattr(prompt_learner, 'compound_prompts_text'):
            for idx, param in enumerate(prompt_learner.compound_prompts_text):
                if param.grad is not None:
                    depth_grads[f'layer_{idx}'] = {
                        'norm': param.grad.norm().item(),
                        'mean': param.grad.mean().item()
                    }

        # Fit log-norm decay across depths
        if len(depth_grads) > 1:
            norms = [v['norm'] for v in depth_grads.values()]
            decay_rate = np.polyfit(range(len(norms)), np.log(np.array(norms) + 1e-10), 1)[0]
        else:
            decay_rate = 0

        return {
            'depth_grads': depth_grads,
            'decay_rate': decay_rate
        }


class ExperimentB_LossCompetition:
    """
    Experiment B: gradient competition between loss terms on gates.

    Expected:
    - Misaligned gradients across cls/reg/cycle on gates
    - Angles near 90° or larger (conflict)
    - ||sum|| < sum(||·||) (cancellation)
    """

    @staticmethod
    def compute_gradient_for_loss(model, image, label, loss_type):
        """Compute gradients for one loss component only."""
        model.zero_grad()

        total_loss, cls_loss, reg_loss, cycle_loss, _, _ = model(image, label)

        # Pick loss by type
        if loss_type == 'cls':
            loss = cls_loss
        elif loss_type == 'reg':
            loss = reg_loss
        elif loss_type == 'cycle':
            loss = cycle_loss if isinstance(cycle_loss, torch.Tensor) else torch.tensor(0.0, device=image.device)
        else:
            loss = total_loss

        loss.backward(retain_graph=True)

        # Gate gradients
        gate_grads = {}
        for name, param in model.named_parameters():
            if ('length_gates' in name or 'depth_weights' in name) and param.grad is not None:
                gate_grads[name] = param.grad.clone()

        return gate_grads

    @staticmethod
    def test_gradient_conflict(model, image, label):
        """Test 1: angles between per-loss gate gradients."""
        loss_types = ['cls', 'reg', 'cycle']
        gradients = {}

        for loss_type in loss_types:
            try:
                gradients[loss_type] = ExperimentB_LossCompetition.compute_gradient_for_loss(
                    model, image, label, loss_type
                )
            except Exception as e:
                print(f"Error computing {loss_type} gradient: {e}")
                gradients[loss_type] = {}

        # Pairwise angles
        angles = {}
        for name in gradients['cls'].keys():
            if name in gradients['reg'] and name in gradients['cycle']:
                g_cls = gradients['cls'][name].flatten()
                g_reg = gradients['reg'][name].flatten()
                g_cycle = gradients['cycle'][name].flatten()

                # Pairwise angles
                def cosine_similarity(a, b):
                    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

                angles[name] = {
                    'cls_vs_reg': np.arccos(np.clip(cosine_similarity(g_cls, g_reg), -1, 1)) * 180 / np.pi,
                    'cls_vs_cycle': np.arccos(np.clip(cosine_similarity(g_cls, g_cycle), -1, 1)) * 180 / np.pi,
                    'reg_vs_cycle': np.arccos(np.clip(cosine_similarity(g_reg, g_cycle), -1, 1)) * 180 / np.pi
                }

        return angles

    @staticmethod
    def test_gradient_cancellation(model, image, label):
        """Test 2: cancellation — compare ||∇_total|| vs sum of per-term norms."""
        loss_types = ['cls', 'reg', 'cycle']
        gradients = {}

        for loss_type in loss_types:
            try:
                gradients[loss_type] = ExperimentB_LossCompetition.compute_gradient_for_loss(
                    model, image, label, loss_type
                )
            except:
                gradients[loss_type] = {}

        # Total-loss gradient
        total_grads = ExperimentB_LossCompetition.compute_gradient_for_loss(
            model, image, label, 'total'
        )

        results = {}
        for name in total_grads.keys():
            # Total gradient norm
            total_norm = total_grads[name].norm().item()

            # Sum of per-term norms
            component_norms_sum = sum(
                gradients[lt][name].norm().item()
                for lt in loss_types
                if name in gradients[lt]
            )

            # Cancellation: 1 - (actual / sum of parts)
            cancellation_ratio = 1 - (total_norm / (component_norms_sum + 1e-10))

            results[name] = {
                'total_norm': total_norm,
                'component_sum': component_norms_sum,
                'cancellation_ratio': cancellation_ratio
            }

        return results

    @staticmethod
    def test_regularization_dominance(model, image, label):
        """Test 3: whether regularization dominates gate updates."""
        # Loss magnitudes
        model.train()
        total_loss, cls_loss, reg_loss, cycle_loss, _, _ = model(image, label)

        # Per-loss gradient magnitude on gates
        results = {
            'loss_magnitudes': {
                'cls': cls_loss.item(),
                'reg': reg_loss.item(),
                'cycle': cycle_loss.item() if isinstance(cycle_loss, torch.Tensor) else 0.0
            }
        }

        # Aggregate gate grad contribution per loss type
        gate_grad_contributions = {}
        for loss_type in ['cls', 'reg', 'cycle']:
            grads = ExperimentB_LossCompetition.compute_gradient_for_loss(
                model, image, label, loss_type
            )
            gate_grad_contributions[loss_type] = sum(
                g.abs().mean().item() for g in grads.values()
            ) / (len(grads) + 1e-10)

        results['gate_grad_contributions'] = gate_grad_contributions

        # Reg share of gate grad magnitude
        total_contrib = sum(gate_grad_contributions.values())
        results['reg_dominance'] = gate_grad_contributions['reg'] / (total_contrib + 1e-10)

        return results


class ExperimentC_RedundantDOF:
    """
    Experiment C: redundant degrees of freedom from adaptive parameters.

    Expected:
    - Low sensitivity of logits to gate perturbations
    - Learned pattern similar to fixed schedules
    - Fixed gates can match adaptive accuracy
    """

    @staticmethod
    def test_gate_sensitivity(model, image, label):
        """Measure logit sensitivity to random gate noise."""
        model.eval()

        with torch.no_grad():
            baseline_logits = model(image)

        sensitivities = {}

        prompt_learner = model.prompt_learner

        # length_gates sensitivity
        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            original_gates = prompt_learner.length_gates.clone()

            perturbations = [0.1, 0.5, 1.0]
            for pert in perturbations:
                prompt_learner.length_gates.data = original_gates + torch.randn_like(original_gates) * pert

                with torch.no_grad():
                    perturbed_logits = model(image)

                logit_diff = (perturbed_logits - baseline_logits).abs().mean().item()
                sensitivities[f'length_pert_{pert}'] = logit_diff

            prompt_learner.length_gates.data = original_gates

        # depth_weights sensitivity
        if hasattr(prompt_learner, 'depth_weights') and isinstance(prompt_learner.depth_weights, nn.Parameter):
            original_weights = prompt_learner.depth_weights.clone()
            perturbations = [0.1, 0.5, 1.0]
            for pert in perturbations:
                prompt_learner.depth_weights.data = original_weights + torch.randn_like(original_weights) * pert

                with torch.no_grad():
                    perturbed_logits = model(image)

                logit_diff = (perturbed_logits - baseline_logits).abs().mean().item()
                sensitivities[f'depth_pert_{pert}'] = logit_diff

            prompt_learner.depth_weights.data = original_weights

        return sensitivities

    @staticmethod
    def test_fixed_vs_adaptive(model, val_loader, device):
        """Compare fixed vs learned gate accuracy."""
        model.eval()
        prompt_learner = model.prompt_learner

        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            adaptive_gates = prompt_learner.length_gates.clone()
            adaptive_weights = prompt_learner.depth_weights.clone() if hasattr(prompt_learner,
                                                                               'depth_weights') and isinstance(
                prompt_learner.depth_weights, nn.Parameter) else None
        else:
            return {'error': 'No adaptive gates found'}

        results = {}

        print("    Evaluating adaptive configuration...")
        model.train()
        acc_adaptive = ExperimentC_RedundantDOF._evaluate_accuracy(model, val_loader, device)
        model.eval()
        results['adaptive'] = acc_adaptive

        print("    Evaluating fixed uniform (sigmoid=0.5)...")
        prompt_learner.length_gates = nn.Parameter(torch.zeros_like(adaptive_gates))
        if adaptive_weights is not None:
            prompt_learner.depth_weights = nn.Parameter(torch.zeros_like(adaptive_weights))
        model.train()
        acc_uniform = ExperimentC_RedundantDOF._evaluate_accuracy(model, val_loader, device)
        model.eval()
        results['uniform'] = acc_uniform

        print("    Evaluating fixed full-on (sigmoid~0.88)...")
        prompt_learner.length_gates = nn.Parameter(torch.ones_like(adaptive_gates) * 2.0)
        if adaptive_weights is not None:
            prompt_learner.depth_weights = nn.Parameter(torch.ones_like(adaptive_weights) * 2.0)
        # eval uses hard gating; train uses soft gating for fair comparison across settings
        model.train()
        acc_full = ExperimentC_RedundantDOF._evaluate_accuracy(model, val_loader, device)
        model.eval()
        results['full_active'] = acc_full

        print("    Evaluating fixed decay (shallow-first)...")
        decay_gates = torch.linspace(2.0, -2.0, len(adaptive_gates))  # sigmoid: 0.88 -> 0.12
        prompt_learner.length_gates = nn.Parameter(decay_gates)
        if adaptive_weights is not None:
            decay_weights = torch.linspace(2.0, -2.0, len(adaptive_weights))
            prompt_learner.depth_weights = nn.Parameter(decay_weights)
        model.train()
        acc_decay = ExperimentC_RedundantDOF._evaluate_accuracy(model, val_loader, device)
        model.eval()
        results['decay'] = acc_decay

        print("    Evaluating fixed increase (deep-first)...")
        increase_gates = torch.linspace(-2.0, 2.0, len(adaptive_gates))  # sigmoid: 0.12 -> 0.88
        prompt_learner.length_gates = nn.Parameter(increase_gates)
        if adaptive_weights is not None:
            increase_weights = torch.linspace(-2.0, 2.0, len(adaptive_weights))
            prompt_learner.depth_weights = nn.Parameter(increase_weights)
        model.train()
        acc_increase = ExperimentC_RedundantDOF._evaluate_accuracy(model, val_loader, device)
        model.eval()
        results['increase'] = acc_increase

        prompt_learner.length_gates = nn.Parameter(adaptive_gates)
        if adaptive_weights is not None:
            prompt_learner.depth_weights = nn.Parameter(adaptive_weights)

        results['performance_gap'] = {
            'adaptive_vs_uniform': acc_adaptive - acc_uniform,
            'adaptive_vs_full': acc_adaptive - acc_full,
            'adaptive_vs_decay': acc_adaptive - acc_decay,
            'adaptive_vs_increase': acc_adaptive - acc_increase,
            'best_fixed': max(acc_uniform, acc_full, acc_decay, acc_increase),
            'adaptive_vs_best_fixed': acc_adaptive - max(acc_uniform, acc_full, acc_decay, acc_increase)
        }

        fixed_configs = {
            'uniform': acc_uniform,
            'full_active': acc_full,
            'decay': acc_decay,
            'increase': acc_increase
        }
        best_fixed_name = max(fixed_configs, key=fixed_configs.get)
        results['best_fixed_config'] = best_fixed_name

        return results

    @staticmethod
    def _evaluate_accuracy(model, val_loader, device, max_batches=50):
        """Accuracy on val_loader (subset of batches)."""
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break

                images = batch['img'].to(device)
                labels = batch['label'].to(device)

                logits = model(images)
                _, predicted = logits.max(1)

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total if total > 0 else 0.0

    @staticmethod
    def test_learned_pattern_analysis(model):
        """Summarize learned gate statistics."""
        prompt_learner = model.prompt_learner

        results = {}

        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            gates = torch.sigmoid(prompt_learner.length_gates / prompt_learner.temperature).detach().cpu().numpy()

            results['length_gates'] = {
                'values': gates.tolist(),
                'mean': float(gates.mean()),
                'std': float(gates.std()),
                'entropy': float(
                    -np.sum(gates * np.log(gates + 1e-10) + (1 - gates) * np.log(1 - gates + 1e-10)) / len(gates)),
                'effective_length': int((gates > 0.5).sum()),
                'pattern': 'uniform' if gates.std() < 0.1 else 'varied'
            }

        if hasattr(prompt_learner, 'depth_weights') and isinstance(prompt_learner.depth_weights, nn.Parameter):
            weights = torch.sigmoid(prompt_learner.depth_weights / prompt_learner.temperature).detach().cpu().numpy()

            results['depth_weights'] = {
                'values': weights.tolist(),
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'monotonicity': float(np.corrcoef(range(len(weights)), weights)[0, 1]),  # monotonicity
                'pattern': 'decay' if np.corrcoef(range(len(weights)), weights)[0, 1] < -0.5 else 'uniform'
            }

        return results


class ExperimentRunner:
    """Run all diagnostic experiments and write reports."""

    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.results = {}

    def run_all_experiments(self):
        """Run all diagnostic experiments."""
        print("=" * 80)
        print("Running diagnostic experiments...")
        print("=" * 80)

        batch = next(iter(self.train_loader))
        image = batch['img'].to(self.device)
        label = batch['label'].to(self.device)

        print("\n[Exp A] Gradient scale vs numeric range")
        print("-" * 80)

        print("A1: Gradient magnitude ratio...")
        try:
            self.results['A1_gradient_ratio'] = ExperimentA_GradientFlow.test_gradient_magnitude_ratio(
                self.model, image, label
            )
            print(f"  Prompt grad norm: {self.results['A1_gradient_ratio']['prompt_grad_norm']:.6f}")
            print(f"  Gate grad norm: {self.results['A1_gradient_ratio']['gate_grad_norm']:.6f}")
            print(f"  Ratio: {self.results['A1_gradient_ratio']['ratio']:.6f}")
            print(f"  log10 gap: {self.results['A1_gradient_ratio']['magnitude_gap']:.2f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['A1_gradient_ratio'] = {'error': str(e)}

        print("\nA2: FP16 truncation check...")
        try:
            model_dtype = next(self.model.parameters()).dtype
            print(f"  Model dtype: {model_dtype}")

            self.model.train()
            loss, _, _, _, _, _ = self.model(image, label)
            self.model.zero_grad()
            loss.backward()

            gate_grads = []
            for name, param in self.model.named_parameters():
                if ('length_gates' in name or 'depth_weights' in name) and param.grad is not None:
                    gate_grads.append(param.grad.clone().float())

            if gate_grads:
                gate_grads = torch.cat([g.flatten() for g in gate_grads])
                self.results['A2_fp16_truncation'] = {
                    'current_dtype': str(model_dtype),
                    'mean': float(gate_grads.mean().item()),
                    'std': float(gate_grads.std().item()),
                    'min_nonzero': float(gate_grads[gate_grads.abs() > 0].abs().min().item()) if (
                                gate_grads.abs() > 0).any() else 0.0,
                    'max': float(gate_grads.abs().max().item()),
                    'zeros_ratio': float((gate_grads.abs() < 1e-8).sum().item() / gate_grads.numel()),
                    'note': 'FP16 ~6e-8 floor; min_nonzero near this suggests precision issues'
                }
                print(f"  Min nonzero grad: {self.results['A2_fp16_truncation']['min_nonzero']:.2e}")
                print(f"  Zero fraction: {self.results['A2_fp16_truncation']['zeros_ratio'] * 100:.1f}%")
                if model_dtype == torch.float16 and self.results['A2_fp16_truncation']['min_nonzero'] < 1e-7:
                    print(f"  WARNING: grads near FP16 floor")
            else:
                print(f"  No gate gradients")
                self.results['A2_fp16_truncation'] = {'error': 'No gate gradients found'}

        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['A2_fp16_truncation'] = {'error': str(e)}

        print("\nA3: Chain-length effect...")
        try:
            self.results['A3_chain_length'] = ExperimentA_GradientFlow.test_gradient_chain_length(
                self.model, image, label
            )
            print(f"  Decay slope: {self.results['A3_chain_length']['decay_rate']:.6f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['A3_chain_length'] = {'error': str(e)}

        print("\n[Exp B] Competing regularizers")
        print("-" * 80)

        print("B1: Gradient conflict angles...")
        try:
            self.results['B1_gradient_conflict'] = ExperimentB_LossCompetition.test_gradient_conflict(
                self.model, image, label
            )
            for name, angles in list(self.results['B1_gradient_conflict'].items())[:3]:
                print(f"  {name.split('.')[-1][:20]}:")
                print(f"    cls vs reg: {angles['cls_vs_reg']:.1f}°")
                print(f"    cls vs cycle: {angles['cls_vs_cycle']:.1f}°")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['B1_gradient_conflict'] = {'error': str(e)}

        print("\nB2: Gradient cancellation...")
        try:
            self.results['B2_gradient_cancellation'] = ExperimentB_LossCompetition.test_gradient_cancellation(
                self.model, image, label
            )
            for name, stats in list(self.results['B2_gradient_cancellation'].items())[:3]:
                print(f"  {name.split('.')[-1][:20]}: cancel {stats['cancellation_ratio'] * 100:.1f}%")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['B2_gradient_cancellation'] = {'error': str(e)}

        print("\nB3: Regularization dominance...")
        try:
            self.results['B3_reg_dominance'] = ExperimentB_LossCompetition.test_regularization_dominance(
                self.model, image, label
            )
            print(f"  Reg share: {self.results['B3_reg_dominance']['reg_dominance'] * 100:.1f}%")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['B3_reg_dominance'] = {'error': str(e)}

        print("\n[Exp C] Redundant DOF")
        print("-" * 80)

        print("C1: Gate sensitivity...")
        try:
            self.results['C1_sensitivity'] = ExperimentC_RedundantDOF.test_gate_sensitivity(
                self.model, image, label
            )
            for pert, diff in list(self.results['C1_sensitivity'].items())[:3]:
                print(f"  {pert}: logit delta {diff:.6f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['C1_sensitivity'] = {'error': str(e)}

        print("\nC2: Fixed vs adaptive accuracy...")
        try:
            self.results['C2_fixed_vs_adaptive'] = ExperimentC_RedundantDOF.test_fixed_vs_adaptive(
                self.model, self.val_loader, self.device
            )
            if 'error' not in self.results['C2_fixed_vs_adaptive']:
                c2 = self.results['C2_fixed_vs_adaptive']
                print(f"  Adaptive:       {c2['adaptive']:.2f}%")
                print(f"  Fixed uniform:  {c2['uniform']:.2f}% ({c2['performance_gap']['adaptive_vs_uniform']:+.2f}%)")
                print(f"  Fixed full-on:  {c2['full_active']:.2f}% ({c2['performance_gap']['adaptive_vs_full']:+.2f}%)")
                print(f"  Fixed decay:      {c2['decay']:.2f}% ({c2['performance_gap']['adaptive_vs_decay']:+.2f}%)")
                print(
                    f"  Fixed increase:   {c2['increase']:.2f}% ({c2['performance_gap']['adaptive_vs_increase']:+.2f}%)")
                print(f"  Best fixed:       {c2['best_fixed_config']} ({c2['performance_gap']['best_fixed']:.2f}%)")
                print(f"  Adaptive vs best: {c2['performance_gap']['adaptive_vs_best_fixed']:+.2f}%")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['C2_fixed_vs_adaptive'] = {'error': str(e)}

        print("\nC3: Learned gate patterns...")
        try:
            self.results['C3_pattern_analysis'] = ExperimentC_RedundantDOF.test_learned_pattern_analysis(
                self.model
            )
            if 'length_gates' in self.results['C3_pattern_analysis']:
                print(f"  Length gates pattern: {self.results['C3_pattern_analysis']['length_gates']['pattern']}")
                print(f"  Effective length: {self.results['C3_pattern_analysis']['length_gates']['effective_length']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            self.results['C3_pattern_analysis'] = {'error': str(e)}

        self.save_results()
        self.generate_report()

        print("\n" + "=" * 80)
        print("Done. Results saved to:", self.save_dir)
        print("=" * 80)

    def save_results(self):
        """Write experiment_results.json."""
        with open(os.path.join(self.save_dir, 'experiment_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)

    def generate_report(self):
        report_path = os.path.join(self.save_dir, 'diagnostic_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AdaptiveBiDirMaPLe diagnostic report\n")
            f.write("=" * 80 + "\n\n")

            # Hypothesis A
            f.write("[Hypothesis A] Gradient scale vs numeric range\n")
            f.write("-" * 80 + "\n")

            mag_gap = self.results.get('A1_gradient_ratio', {}).get('magnitude_gap', 0)

            if 'error' not in self.results.get('A1_gradient_ratio', {}):
                if mag_gap > 2:
                    verdict_a = "PASS"
                    f.write(f"{verdict_a}: Gate grads ~{mag_gap:.1f} orders smaller than prompt grads\n")
                else:
                    verdict_a = "NOT SHOWN"
                    f.write(f"{verdict_a}: Log10 gap only {mag_gap:.1f}\n")
            else:
                verdict_a = "FAILED"
                f.write(f"{verdict_a}: {self.results['A1_gradient_ratio'].get('error', 'Unknown error')}\n")

            if 'error' not in self.results.get('A2_fp16_truncation', {}):
                a2_result = self.results['A2_fp16_truncation']
                min_grad = a2_result.get('min_nonzero', 0)
                zeros_ratio = a2_result.get('zeros_ratio', 0)

                if min_grad < 1e-7 or zeros_ratio > 0.2:
                    f.write(f"PASS FP16 issue: min grad {min_grad:.2e}, zero fraction {zeros_ratio * 100:.1f}%\n")
                else:
                    f.write(f"OK precision: min grad {min_grad:.2e}, zero fraction {zeros_ratio * 100:.1f}%\n")

            a_verified = mag_gap > 2 if 'error' not in self.results.get('A1_gradient_ratio', {}) else False
            f.write(f"\nConclusion A: {'supported' if a_verified else 'not supported or inconclusive'}\n")
            if a_verified:
                f.write(f"  - Gate grads much weaker than prompt grads\n")
                f.write(f"  - Try grad scaling or group LR (gate_lr = base_lr * 100)\n\n")
            else:
                f.write(f"  - Magnitude gap is within a tolerable range\n\n")

            f.write("[Hypothesis B] Competing regularizers\n")
            f.write("-" * 80 + "\n")

            high_conflict = False
            if 'error' not in self.results.get('B1_gradient_conflict', {}) and self.results.get('B1_gradient_conflict'):
                for name, angles in self.results['B1_gradient_conflict'].items():
                    if isinstance(angles, dict) and 'cls_vs_reg' in angles:
                        if angles['cls_vs_reg'] > 60 or angles.get('cls_vs_cycle', 0) > 60:
                            high_conflict = True
                            break

            if high_conflict:
                verdict_b1 = "PASS"
                f.write(f"{verdict_b1}: Strong gradient conflict (angle > 60 deg)\n")
            elif 'error' in self.results.get('B1_gradient_conflict', {}):
                verdict_b1 = "FAILED"
                f.write(f"{verdict_b1}: {self.results['B1_gradient_conflict'].get('error', '')}\n")
            else:
                verdict_b1 = "NOT SHOWN"
                f.write(f"{verdict_b1}: No strong conflict\n")

            avg_cancellation = 0
            if 'error' not in self.results.get('B2_gradient_cancellation', {}) and self.results.get(
                    'B2_gradient_cancellation'):
                cancellations = [
                    stats['cancellation_ratio']
                    for stats in self.results['B2_gradient_cancellation'].values()
                    if isinstance(stats, dict) and 'cancellation_ratio' in stats
                ]
                if cancellations:
                    avg_cancellation = np.mean(cancellations)

            if avg_cancellation > 0.3:
                verdict_b2 = "PASS"
                f.write(f"{verdict_b2}: High cancellation rate {avg_cancellation * 100:.1f}%\n")
            elif 'error' in self.results.get('B2_gradient_cancellation', {}):
                verdict_b2 = "FAILED"
                f.write(f"{verdict_b2}\n")
            else:
                verdict_b2 = "NOT SHOWN"
                f.write(f"{verdict_b2}: Cancellation rate {avg_cancellation * 100:.1f}%\n")

            reg_dom = 0
            if 'error' not in self.results.get('B3_reg_dominance', {}):
                reg_dom = self.results.get('B3_reg_dominance', {}).get('reg_dominance', 0)

            if reg_dom > 0.5:
                verdict_b3 = "PASS"
                f.write(f"{verdict_b3}: Regularization dominates gate updates ({reg_dom * 100:.1f}%)\n")
            elif 'error' in self.results.get('B3_reg_dominance', {}):
                verdict_b3 = "FAILED"
                f.write(f"{verdict_b3}\n")
            else:
                verdict_b3 = "NOT SHOWN"
                f.write(f"{verdict_b3}: Reg share of gate grad {reg_dom * 100:.1f}%\n")

            b_verified = high_conflict or avg_cancellation > 0.3 or reg_dom > 0.5
            f.write(f"\nConclusion B: {'supported' if b_verified else 'not supported or inconclusive'}\n")
            if b_verified:
                f.write(f"  - Multiple losses compete or cancel\n")
                f.write(f"  - Regularization may over-constrain gates\n")
                f.write(f"  - Try lower reg (lambda * 0.01) or drop cycle loss\n\n")
            else:
                f.write(f"  - Little evidence of competing losses\n\n")

            f.write("[Hypothesis C] Redundant degrees of freedom\n")
            f.write("-" * 80 + "\n")

            max_sensitivity = 0
            if 'error' not in self.results.get('C1_sensitivity', {}) and self.results.get('C1_sensitivity'):
                sensitivities = [v for v in self.results['C1_sensitivity'].values() if isinstance(v, (int, float))]
                if sensitivities:
                    max_sensitivity = max(sensitivities)

            if max_sensitivity < 0.01:
                verdict_c1 = "PASS"
                f.write(f"{verdict_c1}: Tiny output change under gate noise (max {max_sensitivity:.6f})\n")
            elif 'error' in self.results.get('C1_sensitivity', {}):
                verdict_c1 = "FAILED"
                f.write(f"{verdict_c1}\n")
            else:
                verdict_c1 = "NOT SHOWN"
                f.write(f"{verdict_c1}: Perturbation effect {max_sensitivity:.6f}\n")

            perf_gap = 0
            best_fixed_gap = 0
            if 'error' not in self.results.get('C2_fixed_vs_adaptive', {}):
                c2_result = self.results.get('C2_fixed_vs_adaptive', {})
                perf_gap = c2_result.get('performance_gap', {}).get('adaptive_vs_uniform', 0)
                best_fixed_gap = c2_result.get('performance_gap', {}).get('adaptive_vs_best_fixed', 0)

                f.write(f"Accuracy comparison:\n")
                f.write(f"  Adaptive:           {c2_result.get('adaptive', 0):.2f}%\n")
                f.write(
                    f"  Fixed uniform:      {c2_result.get('uniform', 0):.2f}% (gap {c2_result.get('performance_gap', {}).get('adaptive_vs_uniform', 0):+.2f}%)\n")
                f.write(
                    f"  Fixed full-on:      {c2_result.get('full_active', 0):.2f}% (gap {c2_result.get('performance_gap', {}).get('adaptive_vs_full', 0):+.2f}%)\n")
                f.write(
                    f"  Fixed decay:        {c2_result.get('decay', 0):.2f}% (gap {c2_result.get('performance_gap', {}).get('adaptive_vs_decay', 0):+.2f}%)\n")
                f.write(
                    f"  Fixed increase:     {c2_result.get('increase', 0):.2f}% (gap {c2_result.get('performance_gap', {}).get('adaptive_vs_increase', 0):+.2f}%)\n")
                f.write(
                    f"  Best fixed: {c2_result.get('best_fixed_config', 'unknown')} ({c2_result.get('performance_gap', {}).get('best_fixed', 0):.2f}%)\n")

            if abs(best_fixed_gap) < 2.0:
                verdict_c2 = "PASS"
                if best_fixed_gap < 0:
                    f.write(f"\n{verdict_c2}: Best fixed beats adaptive by {abs(best_fixed_gap):.2f}%\n")
                else:
                    f.write(f"\n{verdict_c2}: Adaptive only +{best_fixed_gap:.2f}% vs best fixed\n")
            elif 'error' in self.results.get('C2_fixed_vs_adaptive', {}):
                verdict_c2 = "FAILED"
                f.write(f"\n{verdict_c2}\n")
            else:
                verdict_c2 = "NOT SHOWN"
                f.write(f"\n{verdict_c2}: Adaptive improves by {best_fixed_gap:.2f}%\n")

            pattern_uniform = False
            if 'error' not in self.results.get('C3_pattern_analysis', {}):
                if 'length_gates' in self.results.get('C3_pattern_analysis', {}):
                    pattern = self.results['C3_pattern_analysis']['length_gates'].get('pattern', '')
                    if pattern == 'uniform':
                        verdict_c3 = "PASS"
                        f.write(f"{verdict_c3}: Gate pattern is uniform (little adaptation)\n")
                        pattern_uniform = True
                    else:
                        verdict_c3 = "NOT SHOWN"
                        f.write(f"{verdict_c3}: Non-uniform gate pattern\n")

            c_verified = max_sensitivity < 0.01 or abs(best_fixed_gap) < 2.0 or pattern_uniform
            f.write(f"\nConclusion C: {'supported' if c_verified else 'not supported or inconclusive'}\n")
            if c_verified:
                f.write(f"  - Adaptive knobs have limited impact on accuracy\n")

                if 'error' not in self.results.get('C2_fixed_vs_adaptive', {}):
                    c2_result = self.results.get('C2_fixed_vs_adaptive', {})
                    best_config = c2_result.get('best_fixed_config', 'unknown')
                    best_fixed_acc = c2_result.get('performance_gap', {}).get('best_fixed', 0)

                    if best_fixed_gap < 0:
                        f.write(f"  - Fixed ({best_config}) can beat adaptive\n")
                        f.write(f"  - Adaptive gates may hurt\n")
                    elif best_fixed_gap < 1.0:
                        f.write(f"  - Adaptive only +{best_fixed_gap:.2f}% vs best fixed ({best_config})\n")
                        f.write(f"  - May not justify training cost\n")

                f.write(f"  - Extra DOF look redundant\n")
                f.write(f"  - Consider fixed MaPLe or remove gates\n\n")
            else:
                f.write(f"  - Some evidence adaptive gates help\n\n")

            f.write("=" * 80 + "\n")
            f.write("Summary\n")
            f.write("=" * 80 + "\n")

            verified_count = sum([a_verified, b_verified, c_verified])

            f.write(f"Hypotheses with support: {verified_count}/3\n\n")

            if verified_count > 0:
                f.write("Issues:\n")
                if a_verified:
                    f.write("1. Gate gradients too small to update effectively\n")
                if b_verified:
                    f.write("2. Multiple losses compete and weaken gate learning\n")
                if c_verified:
                    f.write("3. Adaptive parameters add redundant DOF with little gain\n")

                f.write("\nRoot causes (speculative):\n")
                f.write("AdaptiveBiDirMaPLe may fail when several factors combine:\n")
                f.write("- Architecture: gates are far from the loss; long backprop paths\n")
                f.write("- Optimization: conflicting loss gradients cancel out\n")
                f.write("- Modeling: CLIP attention is already adaptive; explicit gates may be redundant\n")

                f.write("\nSuggestions:\n")
                f.write("[Preferred] Use fixed MaPLe:\n")
                f.write("  - Remove adaptive gates (length_gates, depth_weights)\n")
                f.write("  - Tune n_ctx and prompt_depth by grid search\n")
                f.write("  - Use standard MaPLe or BiDirMaPLe; focus on prompt quality\n")
                f.write("\n[If keeping adaptive]\n")
                f.write("  1. Gradients:\n")
                f.write("     - 100x gate LR vs prompt\n")
                f.write("     - Gradient accumulation or scaling\n")
                f.write("     - Freeze prompts early; train gates alone\n")
                f.write("  2. Losses:\n")
                f.write("     - Drop cycle loss and heavy regularization\n")
                f.write("     - Light sparsity only (lambda=0.0001)\n")
                f.write("     - Dynamic loss balancing\n")
                f.write("  3. DOF:\n")
                f.write("     - Simpler gates, e.g. one global depth scalar\n")
                f.write("     - Discrete gates (Gumbel-Softmax)\n")
                f.write("     - Meta-learn gates from task features\n")

                f.write("\n[Last resort] Accept adaptive may not help:\n")
                f.write("  - Likely a design issue, not a bug\n")
                f.write("  - Strong fixed prompts may be enough\n")
                f.write("  - Prefer multimodal fusion or other directions\n")
            else:
                f.write("No hypothesis clearly supported. Possible reasons:\n")
                f.write("- Training is healthy; adaptive may be working\n")
                f.write("- Snapshot too small to show issues\n")
                f.write("- Need longer training\n")

        print(f"\nDiagnostic report saved: {report_path}")

    def visualize_results(self):
        """Save diagnostic plots to diagnostic_plots.png."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('AdaptiveBiDirMaPLe diagnostic summary', fontsize=16)

        # A1: gradient magnitudes
        ax = axes[0, 0]
        if 'A1_gradient_ratio' in self.results:
            prompt_norm = self.results['A1_gradient_ratio']['prompt_grad_norm']
            gate_norm = self.results['A1_gradient_ratio']['gate_grad_norm']
            ax.bar(['Prompt', 'Gate'], [prompt_norm, gate_norm])
            ax.set_yscale('log')
            ax.set_ylabel('Gradient Norm (log scale)')
            ax.set_title('A1: Gradient Magnitude')
            ax.grid(True, alpha=0.3)

        # B1: angles
        ax = axes[0, 1]
        if 'B1_gradient_conflict' in self.results and self.results['B1_gradient_conflict']:
            angles_data = []
            labels = []
            for name, angles in list(self.results['B1_gradient_conflict'].items())[:3]:
                angles_data.append([angles['cls_vs_reg'], angles['cls_vs_cycle'], angles['reg_vs_cycle']])
                labels.append(name.split('.')[-1][:10])

            if angles_data:
                x = np.arange(len(labels))
                width = 0.25
                ax.bar(x - width, [a[0] for a in angles_data], width, label='cls vs reg')
                ax.bar(x, [a[1] for a in angles_data], width, label='cls vs cycle')
                ax.bar(x + width, [a[2] for a in angles_data], width, label='reg vs cycle')
                ax.set_ylabel('Angle (degrees)')
                ax.set_title('B1: Gradient Conflicts')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Orthogonal')
                ax.legend()
                ax.grid(True, alpha=0.3)

        # B2: cancellation
        ax = axes[0, 2]
        if 'B2_gradient_cancellation' in self.results:
            cancel_ratios = [
                stats['cancellation_ratio']
                for stats in list(self.results['B2_gradient_cancellation'].values())[:5]
            ]
            param_names = [
                name.split('.')[-1][:10]
                for name in list(self.results['B2_gradient_cancellation'].keys())[:5]
            ]
            ax.bar(param_names, cancel_ratios)
            ax.set_ylabel('Cancellation Ratio')
            ax.set_title('B2: Gradient Cancellation')
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

        # C1: sensitivity
        ax = axes[1, 0]
        if 'C1_sensitivity' in self.results:
            pert_levels = []
            sensitivities = []
            for key, val in self.results['C1_sensitivity'].items():
                if 'length' in key:
                    pert_levels.append(key.split('_')[-1])
                    sensitivities.append(val)

            if pert_levels:
                ax.plot(pert_levels, sensitivities, marker='o')
                ax.set_xlabel('Perturbation Level')
                ax.set_ylabel('Output Change')
                ax.set_title('C1: Gate Sensitivity')
                ax.grid(True, alpha=0.3)

        # C2: fixed vs adaptive
        ax = axes[1, 1]
        if 'C2_fixed_vs_adaptive' in self.results:
            configs = ['Adaptive', 'Uniform', 'Decay']
            accs = [
                self.results['C2_fixed_vs_adaptive'].get('adaptive', 0),
                self.results['C2_fixed_vs_adaptive'].get('uniform', 0),
                self.results['C2_fixed_vs_adaptive'].get('decay', 0)
            ]
            ax.bar(configs, accs)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('C2: Fixed vs Adaptive')
            ax.grid(True, alpha=0.3)

        # C3: learned patterns
        ax = axes[1, 2]
        if 'C3_pattern_analysis' in self.results:
            if 'length_gates' in self.results['C3_pattern_analysis']:
                gates = self.results['C3_pattern_analysis']['length_gates']['values']
                ax.plot(gates, marker='o', label='Length Gates')
            if 'depth_weights' in self.results['C3_pattern_analysis']:
                weights = self.results['C3_pattern_analysis']['depth_weights']['values']
                ax.plot(weights, marker='s', label='Depth Weights')
            ax.set_xlabel('Index')
            ax.set_ylabel('Probability')
            ax.set_title('C3: Learned Patterns')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'diagnostic_plots.png'), dpi=150)
        plt.close()

        print(f"Plots saved: {os.path.join(self.save_dir, 'diagnostic_plots.png')}")


def run_diagnostics(trainer, epoch=None):
    """
    Run diagnostics from a trainer (e.g. AdaptiveBiDirMaPLe).

    Args:
        trainer: Trainer instance with model, loaders, device, cfg.
        epoch: Label for output folder (optional).
    """
    save_dir = os.path.join(trainer.cfg.OUTPUT_DIR, f"diagnostics_epoch_{epoch if epoch else 'final'}")

    runner = ExperimentRunner(
        model=trainer.model,
        train_loader=trainer.train_loader_x,
        val_loader=trainer.val_loader,
        device=trainer.device,
        save_dir=save_dir
    )

    runner.run_all_experiments()
    runner.visualize_results()

    return runner.results


def integrate_diagnostics_into_trainer():
    """
    Snippet: call from trainer.after_epoch().

    ```python
    def after_epoch(self):
        super().after_epoch()

        if self.epoch in [1, 5, 10] or self.epoch == self.max_epoch:
            print(f"\\nDiagnostics (epoch {self.epoch})...")
            from diagnostics.diagnostic_experiments import run_diagnostics
            results = run_diagnostics(self, epoch=self.epoch)

            if 'A1_gradient_ratio' in results:
                mag_gap = results['A1_gradient_ratio']['magnitude_gap']
                if mag_gap > 3:
                    print(f"Warning: small gate grads (~{mag_gap:.1f} orders below prompt)")
                    print("Try higher gate LR or lower regularization")
    ```
    """
    pass


if __name__ == "__main__":
    print(__doc__)