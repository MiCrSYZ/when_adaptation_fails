import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import build_model_adaptive
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import os

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    design_details = {"trainer": 'AdaptiveMaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.ADAPTIVE_MAPLE.N_CTX_MAX,
                      "maple_depth_max": cfg.TRAINER.ADAPTIVE_MAPLE.PROMPT_DEPTH_MAX}
    model = build_model_adaptive(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, depth_weights=None):
        prompts = prompts.type(self.dtype)
        ctx_len = prompts.shape[1]
        pos_len = self.positional_embedding.shape[0]
        if ctx_len < pos_len:
            pad = torch.zeros(
                (prompts.shape[0], pos_len - ctx_len, prompts.shape[2]),
                dtype=prompts.dtype,
                device=prompts.device
            )
            prompts = torch.cat([prompts, pad], dim=1)
        elif ctx_len > pos_len:
            prompts = prompts[:, :pos_len, :]

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Match model.py transformer input format
        combined = [x, compound_prompts_deeper_text or [], 0, depth_weights]

        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AdaptiveMultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.adaptive_type = cfg.TRAINER.ADAPTIVE_MAPLE.TYPE  # ablation: full / length_only / depth_only
        n_cls = len(classnames)
        n_ctx_max = cfg.TRAINER.ADAPTIVE_MAPLE.N_CTX_MAX  # max prompt tokens
        ctx_init = cfg.TRAINER.ADAPTIVE_MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        text_dim = clip_model.ln_final.weight.shape[0]

        vision_dim = None
        if hasattr(clip_model.visual, "transformer") and hasattr(clip_model.visual.transformer, "width"):
            vision_dim = clip_model.visual.transformer.width
        elif hasattr(clip_model.visual, "conv1"):
            vision_dim = clip_model.visual.conv1.weight.shape[0]
        if vision_dim is None:
            vision_dim = 768
            print(f"Warning: Auto-get vision_dim failed, use default {vision_dim} (check your CLIP model type)")

        self.n_cls = n_cls
        self.dtype = dtype
        self.text_dim = text_dim
        self.vision_dim = vision_dim

        self.prompt_depth_max = cfg.TRAINER.ADAPTIVE_MAPLE.PROMPT_DEPTH_MAX
        assert self.prompt_depth_max >= 1, "For AdaptiveMaPLe, PROMPT_DEPTH_MAX should be >= 1"
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # Length gates: learnable or fixed by ablation
        if self.adaptive_type == "depth_only":
            fixed_n_ctx = cfg.TRAINER.ADAPTIVE_MAPLE.FIXED_N_CTX
            assert 1 <= fixed_n_ctx <= n_ctx_max, f"FIXED_N_CTX({fixed_n_ctx})out of range[1, {n_ctx_max}]"
            self.length_gates = torch.zeros(n_ctx_max)
            self.length_gates[:fixed_n_ctx] = 1.0
            print(f"[depth only] prompt length fixed: {fixed_n_ctx}")
        else:
            self.length_gates = nn.Parameter(torch.ones(n_ctx_max))
            print(f"[length only] learnable prompt length(N_CTX_MAX: {n_ctx_max})")

        n_depth_weights = self.prompt_depth_max - 1
        if self.adaptive_type == "length_only":
            fixed_depth = cfg.TRAINER.ADAPTIVE_MAPLE.FIXED_PROMPT_DEPTH
            assert 1 <= fixed_depth <= self.prompt_depth_max, f"FIXED_DEPTH({fixed_depth})out of range[1, {self.prompt_depth_max}]"
            self.depth_weights = torch.zeros(n_depth_weights)
            self.depth_weights[:fixed_depth - 1] = 1.0
            print(f"[length only] depth fixed: {fixed_depth}")
        else:
            self.depth_weights = nn.Parameter(torch.ones(self.prompt_depth_max - 1))
            print(f"[depth only] learnable depth(PROMPT_DEPTH_MAX: {self.prompt_depth_max})")

        # Length / regularization hyperparams
        self.n_ctx_max = n_ctx_max
        self.lambda_sparsity = cfg.TRAINER.ADAPTIVE_MAPLE.LAMBDA_SPARSITY
        self.lambda_depth_smooth = cfg.TRAINER.ADAPTIVE_MAPLE.LAMBDA_DEPTH_SMOOTH
        
        if ctx_init and (n_ctx_max) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx_max, :]
            if ctx_vectors.shape[0] < n_ctx_max:
                remaining = n_ctx_max - ctx_vectors.shape[0]
                random_ctx = torch.empty(remaining, ctx_dim, dtype=dtype)
                nn.init.normal_(random_ctx, std=0.02)
                ctx_vectors = torch.cat([ctx_vectors, random_ctx], dim=0)
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx_max, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx_max)

        print('AdaptiveMaPLe design: Adaptive Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Maximum MaPLe context words (tokens): {n_ctx_max}")
        print(f"Maximum prompt depth: {self.prompt_depth_max}")
        
        self.ctx = nn.Parameter(ctx_vectors)

        self.shared_proj_out = getattr(clip_model.visual, "ln_post", None)
        self.proj = nn.Linear(text_dim, vision_dim, dtype=self.dtype)

        # Deep text prompts
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx_max, 512))
            for _ in range(self.prompt_depth_max - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
            
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.prompt_depth_max - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        device = clip_model.token_embedding.weight.device
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx_max:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def get_effective_length(self):
        """Effective prompt length (soft in train, hard threshold at eval)."""
        if isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
        else:
            gates = self.length_gates
        if self.training and isinstance(self.length_gates, nn.Parameter):
            return gates.sum()
        else:
            return (gates > 0.5).sum().item()

    def get_effective_ctx(self):
        """Masked context vectors."""
        if isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
        else:
            gates = self.length_gates
        if self.training and isinstance(self.length_gates, nn.Parameter):
            effective_ctx = self.ctx * gates.unsqueeze(1)  # [n_ctx_max, ctx_dim]
        else:
            mask = gates > 0.5
            effective_ctx = self.ctx[mask]  # [effective_length, ctx_dim]
        return effective_ctx, gates
    
    def get_depth_probabilities(self):
        """Per-layer depth gate probabilities."""
        if isinstance(self.depth_weights, nn.Parameter):
            return torch.sigmoid(self.depth_weights)
        else:
            return self.depth_weights

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx_eff, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx_eff, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        effective_ctx, length_gates = self.get_effective_ctx()
        effective_ctx = effective_ctx.type(self.dtype)
        depth_probs = self.get_depth_probabilities()

        # Expand to batch
        if effective_ctx.dim() == 2:
            effective_ctx = effective_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(effective_ctx, prefix, suffix)

        # Vision deep prompts
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            if isinstance(self.length_gates, nn.Parameter) and self.training:
                layer_ctx = self.compound_prompts_text[index] * length_gates.unsqueeze(1)
            else:
                mask = length_gates > 0.5
                layer_ctx = self.compound_prompts_text[index][mask]
            visual_deep_prompts.append(layer(layer_ctx))

        # Text encoder expects full compound_prompts_text list
        return (
            prompts,
            self.proj(effective_ctx.mean(0)),  # shared_ctx
            list(self.compound_prompts_text),  # for text side
            visual_deep_prompts,  # for vision side
            depth_probs,
            length_gates
        )

    def compute_regularization_loss(self):
        sparsity_loss = 0.0
        # Sparsity: shorter prompts
        if self.adaptive_type in ["full", "depth_only"] and isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
            sparsity_loss = self.lambda_sparsity * gates.sum()
        
        # Depth smoothness
        depth_smooth_loss = 0.0
        if self.adaptive_type in ["full", "length_only"] and isinstance(self.depth_weights, nn.Parameter):
            if len(self.depth_weights) > 1:
                depth_diffs = self.depth_weights[1:] - self.depth_weights[:-1]
                depth_smooth_loss = self.lambda_depth_smooth * torch.abs(depth_diffs).sum()

        reg_loss = sparsity_loss + depth_smooth_loss
        return reg_loss


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = AdaptiveMultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (prompts, shared_ctx,
         compound_prompts_text, visual_deep_prompts,
         depth_probs, length_gates) = self.prompt_learner()

        image = image.type(self.dtype)
        prompts = prompts.type(self.dtype)

        text_features = self.text_encoder(prompts, tokenized_prompts, compound_prompts_text, depth_probs)
        image_features = self.image_encoder(image, shared_ctx, visual_deep_prompts, depth_probs)

        eps = 1e-8
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            cls_loss = F.cross_entropy(logits, label)
            reg_loss = self.prompt_learner.compute_regularization_loss()
            total_loss = cls_loss + reg_loss

            def to_scalar(x):
                if isinstance(x, torch.Tensor):
                    if x.numel() == 1:
                        return x.item()
                    else:
                        return x.mean().item()
                elif isinstance(x, np.ndarray):
                    if x.size == 1:
                        return float(x.item())
                    else:
                        return float(x.mean())
                elif isinstance(x, (float, int)):
                    return float(x)
                else:
                    return float(x)

            return total_loss, cls_loss, reg_loss, to_scalar(length_gates), to_scalar(depth_probs)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class AdaptiveMaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTIVE_MAPLE.PREC in ["fp16", "fp32", "amp"]

        # Metric buffers
        self.metrics = {
            'train_loss': [],
            'cls_loss': [],
            'reg_loss': [],
            'val_acc': [],
            'effective_length': [],
            'avg_depth_prob': [],
            'max_depth_prob': [],
            'min_depth_prob': [],
            'length_gates': [],
            'depth_probs': [],
            'epochs': []
        }

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADAPTIVE_MAPLE.PREC == "fp32" or cfg.TRAINER.ADAPTIVE_MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP with Adaptive MaPLe")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        # Freeze CLIP backbone; train prompts only
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        # Move model to device
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("AdaptiveMultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ADAPTIVE_MAPLE.PREC == "amp" else None

        # Viz output dir
        self.viz_dir = osp.join(cfg.OUTPUT_DIR, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

        #device_count = torch.cuda.device_count()
        #if device_count > 1:
        #    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #    self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.ADAPTIVE_MAPLE.PREC
        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, length_gates, depth_probs = model(image, label)
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            total_loss, cls_loss, reg_loss, length_gates, depth_probs = model(image, label)
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        loss_summary = {
            "total_loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "reg_loss": reg_loss.item()
        }
        
        # Extra logging
        if hasattr(self.model, 'module'):
            prompt_learner = self.model.module.prompt_learner
        else:
            prompt_learner = self.model.prompt_learner
            
        with torch.no_grad():
            eff_length = prompt_learner.get_effective_length()
            depth_probs = prompt_learner.get_depth_probabilities()
            loss_summary.update({
                "effective_length": float(eff_length if isinstance(eff_length, (float, int)) else eff_length.item()),
                "avg_depth_prob": float(depth_probs.mean().item()),
                "max_depth_prob": float(depth_probs.max().item()),
                "min_depth_prob": float(depth_probs.min().item()),
                "length_gates": length_gates if isinstance(length_gates, (float, int)) else float(np.mean(length_gates)),
                "depth_probs": float(depth_probs.mean().item())
            })

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Drop non-tensor keys
            keys_to_remove = [
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix"
            ]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        """Aggregate epoch metrics and validate."""
        super().after_epoch()

        # Epoch means
        avg_metrics = {}
        for key, values in self.current_epoch_metrics.items():
            if values:
                if key in ['length_gates', 'depth_probs']:
                    if all(isinstance(v, (list, np.ndarray)) for v in values):
                        avg_metrics[key] = np.mean(values, axis=0)
                    else:
                        avg_metrics[key] = np.mean(values)
                else:
                    avg_metrics[key] = np.mean(values)

        # Append to history
        for key in avg_metrics:
            if key in self.metrics:
                self.metrics[key].append(avg_metrics[key])

        # Validation accuracy
        if self.epoch % getattr(self.cfg.TRAIN, "EVAL_PERIOD", 1) == 0:
            try:
                val_acc = self.test(split="val")
                self.metrics['val_acc'].append(val_acc)
            except Exception as e:
                print(f"Validation failed: {e}")
                self.metrics['val_acc'].append(float('nan'))
        else:
            self.metrics['val_acc'].append(float('nan'))

        # Epoch index
        self.metrics['epochs'].append(self.epoch)
        self.save_metrics()
        if self.epoch % 5 == 0 or self.epoch == self.max_epoch:
            self.plot_metrics()
            
    def after_train(self):
        """Print final gate stats and save plots."""
        print("\n" + "="*50)
        print("Final Adaptive Parameters:")
        print("="*50)
        
        if hasattr(self.model, 'module'):
            prompt_learner = self.model.module.prompt_learner
        else:
            prompt_learner = self.model.prompt_learner
            
        with torch.no_grad():
            gates = torch.sigmoid(prompt_learner.length_gates)
            depth_probs = torch.sigmoid(prompt_learner.depth_weights)
            
            print(f"Length gates (sigmoid): {gates.cpu().numpy()}")
            print(f"Effective length: {prompt_learner.get_effective_length()}")
            print(f"Depth probabilities: {depth_probs.cpu().numpy()}")
            
            # Hard gates at inference
            effective_tokens = (gates > 0.5).sum().item()
            active_depths = (depth_probs > 0.5).sum().item()
            print(f"Inference effective tokens: {effective_tokens}")
            print(f"Inference active depths: {active_depths}")
            
        print("="*50 + "\n")

        self.plot_metrics(final=True)

    def save_metrics(self):
        """Save metrics JSON."""
        metrics_path = osp.join(self.viz_dir, "training_metrics.json")
        # JSON-serializable
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [v.tolist() for v in value]
            else:
                serializable_metrics[key] = value

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

    def plot_metrics(self, final=False):
        """Plot training curves."""
        if not self.metrics['epochs']:
            return

        epochs = self.metrics['epochs']

        # Figure grid
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(f'Training Metrics - {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)

        # Loss
        if self.metrics['train_loss']:
            axes[0, 0].plot(epochs, self.metrics['train_loss'], label='Total Loss')
            axes[0, 0].plot(epochs, self.metrics['cls_loss'], label='Classification Loss')
            axes[0, 0].plot(epochs, self.metrics['reg_loss'], label='Regularization Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # Val accuracy
        if self.metrics['val_acc']:
            axes[0, 1].plot(epochs, self.metrics['val_acc'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].grid(True)

        # Effective length
        if self.metrics['effective_length']:
            axes[1, 0].plot(epochs, self.metrics['effective_length'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Effective Length')
            axes[1, 0].set_title('Effective Prompt Length')
            axes[1, 0].grid(True)

        # Depth probs
        if self.metrics['depth_probs'] and len(self.metrics['depth_probs'][0]) > 0:
            depth_probs = np.array(self.metrics['depth_probs'])
            for i in range(depth_probs.shape[1]):
                axes[1, 1].plot(epochs, depth_probs[:, i], label=f'Layer {i + 1}')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Probability')
            axes[1, 1].set_title('Depth Probabilities')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # Length gates
        if self.metrics['length_gates'] and len(self.metrics['length_gates'][0]) > 0:
            length_gates = np.array(self.metrics['length_gates'])
            for i in range(length_gates.shape[1]):
                axes[2, 0].plot(epochs, length_gates[:, i], label=f'Token {i + 1}')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Gate Value')
            axes[2, 0].set_title('Length Gate Values')
            axes[2, 0].legend()
            axes[2, 0].grid(True)

        # Depth prob stats
        if self.metrics['avg_depth_prob']:
            axes[2, 1].plot(epochs, self.metrics['avg_depth_prob'], label='Average')
            axes[2, 1].plot(epochs, self.metrics['max_depth_prob'], label='Max')
            axes[2, 1].plot(epochs, self.metrics['min_depth_prob'], label='Min')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Probability')
            axes[2, 1].set_title('Depth Probability Statistics')
            axes[2, 1].legend()
            axes[2, 1].grid(True)

        plt.tight_layout()

        # Save figure
        if final:
            plt.savefig(osp.join(self.viz_dir, 'final_metrics.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(osp.join(self.viz_dir, f'metrics_epoch_{self.epoch}.png'), dpi=150, bbox_inches='tight')

        plt.close()

    def before_epoch(self):
        """Reset per-epoch metric buffers."""
        super().before_epoch()

        # Fresh lists for this epoch
        self.current_epoch_metrics = {
            'train_loss': [],
            'cls_loss': [],
            'reg_loss': [],
            'effective_length': [],
            'avg_depth_prob': [],
            'max_depth_prob': [],
            'min_depth_prob': [],
            'length_gates': [],
            'depth_probs': []
        }

    def after_step(self, output):
        """Accumulate batch metrics."""
        super().after_step(output)

        # Batch metrics
        self.current_epoch_metrics['train_loss'].append(output['total_loss'])
        self.current_epoch_metrics['cls_loss'].append(output['cls_loss'])
        self.current_epoch_metrics['reg_loss'].append(output['reg_loss'])
        self.current_epoch_metrics['effective_length'].append(output['effective_length'])
        self.current_epoch_metrics['avg_depth_prob'].append(output['avg_depth_prob'])
        self.current_epoch_metrics['max_depth_prob'].append(output['max_depth_prob'])
        self.current_epoch_metrics['min_depth_prob'].append(output['min_depth_prob'])
        if 'length_gates' in output:
            if isinstance(output['length_gates'], (list, np.ndarray)):
                self.current_epoch_metrics['length_gates'].append(np.mean(output['length_gates']))
            else:
                self.current_epoch_metrics['length_gates'].append(output['length_gates'])

        if 'depth_probs' in output:
            if isinstance(output['depth_probs'], (list, np.ndarray)):
                self.current_epoch_metrics['depth_probs'].append(np.mean(output['depth_probs']))
            else:
                self.current_epoch_metrics['depth_probs'].append(output['depth_probs'])
