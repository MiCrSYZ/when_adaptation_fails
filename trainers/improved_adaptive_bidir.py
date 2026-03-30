"""
Improved Adaptive BiDir MaPLe trainer (experiments A & B):
1. Dynamic gradient scaling on gates
2. Entropy regularization on gates
"""

import os.path as osp
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX

from trainers.adaptive_bidir_maple import (
    AdaptiveBiDirMaPLe,
    CustomCLIP,
    load_clip_to_cpu
)


class ImprovedAdaptiveBiDirMaPLe(AdaptiveBiDirMaPLe):
    """
    Extended trainer with optional:
    - Gradient scaling (Experiment A)
    - Entropy regularization (Experiment B)
    """

    def check_cfg(self, cfg):
        super().check_cfg(cfg)

        cfg.defrost()

        cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_GRADIENT_SCALING = getattr(
            cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, 'USE_GRADIENT_SCALING', False
        )
        cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.GRADIENT_SCALE_MAX = getattr(
            cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, 'GRADIENT_SCALE_MAX', 10.0
        )

        cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_ENTROPY_REG = getattr(
            cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, 'USE_ENTROPY_REG', False
        )
        cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_ENTROPY = getattr(
            cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, 'LAMBDA_ENTROPY', 0.01
        )

        cfg.freeze()

        self.use_gradient_scaling = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_GRADIENT_SCALING
        self.gradient_scale_max = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.GRADIENT_SCALE_MAX
        self.use_entropy_reg = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_ENTROPY_REG
        self.lambda_entropy = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_ENTROPY

        print("\n[Improved Adaptive BiDir MaPLe Config]")
        print(f"  Gradient Scaling: {self.use_gradient_scaling} (max={self.gradient_scale_max})")
        print(f"  Entropy Regularization: {self.use_entropy_reg} (λ={self.lambda_entropy})")

    def _compute_gradient_scale_factors(self, gate_values, epsilon=1e-6):
        """Scale factors: α_i = 1 / (σ(g_i)(1-σ(g_i)) + ε)."""
        sigmoid_g = torch.sigmoid(gate_values)
        variance_term = sigmoid_g * (1 - sigmoid_g)
        scale_factors = 1.0 / (variance_term + epsilon)
        scale_factors = torch.clamp(scale_factors, max=self.gradient_scale_max)
        return scale_factors

    def _apply_gradient_scaling(self):
        """Apply scaling to gate parameter gradients."""
        if not self.use_gradient_scaling:
            return

        prompt_learner = self.model.prompt_learner if hasattr(self.model, 'prompt_learner') else self.model.module.prompt_learner

        if isinstance(prompt_learner.length_gates, nn.Parameter) and prompt_learner.length_gates.grad is not None:
            scale_factors = self._compute_gradient_scale_factors(prompt_learner.length_gates.data)
            prompt_learner.length_gates.grad *= scale_factors

        if isinstance(prompt_learner.depth_weights, nn.Parameter) and prompt_learner.depth_weights.grad is not None:
            scale_factors = self._compute_gradient_scale_factors(prompt_learner.depth_weights.data)
            prompt_learner.depth_weights.grad *= scale_factors

    def _compute_entropy_loss(self, gate_values, epsilon=1e-8):
        """
        Entropy regularizer: L_reg = -λ Σ[σ(g)logσ(g) + (1-σ(g))log(1-σ(g))]

        Returns:
            entropy_loss: scalar loss
            entropy_value: mean entropy (monitoring)
        """
        sigmoid_g = torch.sigmoid(gate_values)

        log_sigmoid_g = torch.log(sigmoid_g + epsilon)
        log_one_minus_sigmoid_g = torch.log(1 - sigmoid_g + epsilon)

        entropy = -(sigmoid_g * log_sigmoid_g + (1 - sigmoid_g) * log_one_minus_sigmoid_g)
        entropy_value = entropy.mean().item()

        entropy_loss = -self.lambda_entropy * entropy.sum()

        return entropy_loss, entropy_value

    def _compute_total_entropy_loss(self):
        """Sum entropy loss over gate tensors."""
        if not self.use_entropy_reg:
            return torch.tensor(0.0, device=self.device), {}

        prompt_learner = self.model.prompt_learner if hasattr(self.model, 'prompt_learner') else self.model.module.prompt_learner

        total_entropy_loss = torch.tensor(0.0, device=self.device)
        entropy_values = {}

        if isinstance(prompt_learner.length_gates, nn.Parameter):
            ent_loss, ent_val = self._compute_entropy_loss(prompt_learner.length_gates)
            total_entropy_loss += ent_loss
            entropy_values['length'] = ent_val

        if isinstance(prompt_learner.depth_weights, nn.Parameter):
            ent_loss, ent_val = self._compute_entropy_loss(prompt_learner.depth_weights)
            total_entropy_loss += ent_loss
            entropy_values['depth'] = ent_val

        return total_entropy_loss, entropy_values

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PREC

        warmup_epochs = 2
        if self.epoch < warmup_epochs:
            for name, param in model.named_parameters():
                if 'length_gates' in name or 'depth_weights' in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if 'length_gates' in name or 'depth_weights' in name:
                    param.requires_grad = True

        if not hasattr(self, 'current_epoch_metrics'):
            self.current_epoch_metrics = {
                'train_loss': [], 'cls_loss': [], 'reg_loss': [], 'cycle_loss': [],
                'entropy_loss': [], 'effective_length': [], 'avg_depth_prob': [],
                'max_depth_prob': [], 'min_depth_prob': [],
                'entropy_length': [], 'entropy_depth': []
            }
        else:
            required_keys = ['train_loss', 'cls_loss', 'reg_loss', 'cycle_loss',
                             'entropy_loss', 'effective_length', 'avg_depth_prob',
                             'max_depth_prob', 'min_depth_prob', 'entropy_length', 'entropy_depth']

            for key in required_keys:
                if key not in self.current_epoch_metrics:
                    self.current_epoch_metrics[key] = []

        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)

                entropy_loss, entropy_values = self._compute_total_entropy_loss()
                total_loss = total_loss + entropy_loss

            optim.zero_grad()
            scaler.scale(total_loss).backward()

            self._apply_gradient_scaling()

            scaler.step(optim)
            scaler.update()
        else:
            total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)

            entropy_loss, entropy_values = self._compute_total_entropy_loss()
            total_loss = total_loss + entropy_loss

            optim.zero_grad()
            total_loss.backward()

            self._apply_gradient_scaling()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if self.batch_idx % 50 == 0 and self.epoch >= warmup_epochs:
                self._print_diagnostic_info(entropy_values)

            optim.step()

        with torch.no_grad():
            prompt_learner = model.prompt_learner if hasattr(model, 'prompt_learner') else model.module.prompt_learner
            eff_length = prompt_learner.get_effective_length()
            depth_probs_v = prompt_learner.get_depth_probabilities()

            def to_scalar(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().mean().item())
                else:
                    return float(x)

            self.current_epoch_metrics['train_loss'].append(to_scalar(total_loss))
            self.current_epoch_metrics['cls_loss'].append(to_scalar(cls_loss))
            self.current_epoch_metrics['reg_loss'].append(to_scalar(reg_loss))
            self.current_epoch_metrics['cycle_loss'].append(to_scalar(cycle_loss))
            self.current_epoch_metrics['entropy_loss'].append(to_scalar(entropy_loss))
            self.current_epoch_metrics['effective_length'].append(to_scalar(eff_length))
            self.current_epoch_metrics['avg_depth_prob'].append(to_scalar(depth_probs_v.mean()))
            self.current_epoch_metrics['max_depth_prob'].append(to_scalar(depth_probs_v.max()))
            self.current_epoch_metrics['min_depth_prob'].append(to_scalar(depth_probs_v.min()))

            if entropy_values:
                self.current_epoch_metrics['entropy_length'].append(entropy_values.get('length', 0.0))
                self.current_epoch_metrics['entropy_depth'].append(entropy_values.get('depth', 0.0))

        loss_summary = {
            "total_loss": to_scalar(total_loss),
            "cls_loss": to_scalar(cls_loss),
            "reg_loss": to_scalar(reg_loss),
            "cycle_loss": to_scalar(cycle_loss),
            "entropy_loss": to_scalar(entropy_loss),
            "effective_length": to_scalar(eff_length),
            "avg_depth_prob": to_scalar(depth_probs_v.mean()),
            "max_depth_prob": to_scalar(depth_probs_v.max()),
            "min_depth_prob": to_scalar(depth_probs_v.min()),
            "length_gates_mean": to_scalar(length_gates),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _print_diagnostic_info(self, entropy_values):
        """Optional debug print for gate vs prompt gradients."""
        model = self.model
        prompt_learner = model.prompt_learner if hasattr(model, 'prompt_learner') else model.module.prompt_learner

        prompt_grad_norm = 0.0
        gate_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'ctx' in name or 'compound_prompts' in name:
                    prompt_grad_norm += param.grad.norm().item() ** 2
                elif 'length_gates' in name or 'depth_weights' in name:
                    gate_grad_norm += param.grad.norm().item() ** 2

        prompt_grad_norm = prompt_grad_norm ** 0.5
        gate_grad_norm = gate_grad_norm ** 0.5

        print(f"\n[Epoch {self.epoch}/{self.max_epoch}] [Batch {self.batch_idx}/{self.num_batches}]")
        print(f"  Prompt Grad Norm: {prompt_grad_norm:.6f}")
        print(f"  Gate Grad Norm:   {gate_grad_norm:.6f}")
        print(f"  Ratio (Gate/Prompt): {gate_grad_norm / (prompt_grad_norm + 1e-8):.6f}")

        if entropy_values:
            print(f"  Length Entropy: {entropy_values.get('length', 0):.4f}")
            print(f"  Depth Entropy:  {entropy_values.get('depth', 0):.4f}")

        if isinstance(prompt_learner.length_gates, nn.Parameter):
            gates_sigmoid = torch.sigmoid(prompt_learner.length_gates)
            print(f"  Length Gates (sigmoid): mean={gates_sigmoid.mean():.4f}, "
                  f"std={gates_sigmoid.std():.4f}, "
                  f"min={gates_sigmoid.min():.4f}, "
                  f"max={gates_sigmoid.max():.4f}")

        if isinstance(prompt_learner.depth_weights, nn.Parameter):
            depth_sigmoid = torch.sigmoid(prompt_learner.depth_weights)
            print(f"  Depth Weights (sigmoid): mean={depth_sigmoid.mean():.4f}, "
                  f"std={depth_sigmoid.std():.4f}, "
                  f"min={depth_sigmoid.min():.4f}, "
                  f"max={depth_sigmoid.max():.4f}")

    def after_epoch(self):
        super().after_epoch()

        if self.use_entropy_reg and hasattr(self, 'current_epoch_metrics'):
            if self.current_epoch_metrics.get('entropy_length'):
                avg_entropy_length = sum(self.current_epoch_metrics['entropy_length']) / len(self.current_epoch_metrics['entropy_length'])
                avg_entropy_depth = sum(self.current_epoch_metrics['entropy_depth']) / len(self.current_epoch_metrics['entropy_depth'])
                print(f"\n[Entropy Reg] Length: {avg_entropy_length:.4f}, Depth: {avg_entropy_depth:.4f}")


@TRAINER_REGISTRY.register()
class ImprovedAdaptiveBiDirMaPLe(ImprovedAdaptiveBiDirMaPLe):
    """Registry entry (same class name as base improved trainer)."""
    pass
