"""
Experiment B: implicit regularization — four standalone trainer variants.
"""
import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import build_model_adaptive

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, trainer_prefix='ADAPTIVE_BIDIR_MAPLE'):
    """Load CLIP to CPU (shared by variants)."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    trainer_cfg = getattr(cfg.TRAINER, trainer_prefix)
    
    design_details = {
        "trainer": 'AdaptiveBiDirMaPLe',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": trainer_cfg.N_CTX_MAX,
        "maple_depth_max": trainer_cfg.PROMPT_DEPTH_MAX
    }

    model = build_model_adaptive(state_dict or model.state_dict(), design_details)
    return model


class CrossModalMappingNetwork(nn.Module):
    """Small cross-modal MLP."""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mapping(x)


class AdaptiveBiDirPromptLearner(nn.Module):
    """Adaptive bidirectional prompt learner (base)."""
    def __init__(self, cfg, classnames, clip_model, trainer_prefix='ADAPTIVE_BIDIR_MAPLE'):
        super().__init__()
        
        trainer_cfg = getattr(cfg.TRAINER, trainer_prefix)
        n_ctx_max = trainer_cfg.N_CTX_MAX
        prompt_depth_max = trainer_cfg.PROMPT_DEPTH_MAX
        prec = trainer_cfg.PREC
        lambda_sparsity = trainer_cfg.LAMBDA_SPARSITY
        lambda_depth_smooth = trainer_cfg.LAMBDA_DEPTH_SMOOTH
        adaptive_type = trainer_cfg.TYPE
        enable_bidir = getattr(trainer_cfg, "BIDIRECTIONAL", True)
        cycle_loss_weight = getattr(trainer_cfg, "CYCLE_LOSS_WEIGHT", 0.1)
        
        classnames = [name.replace("_", " ") for name in classnames]
        n_cls = len(classnames)
        dtype = clip_model.dtype
        text_dim = clip_model.ln_final.weight.shape[0]
        
        if hasattr(clip_model.visual, "transformer") and hasattr(clip_model.visual.transformer, "width"):
            vision_dim = clip_model.visual.transformer.width
        elif hasattr(clip_model.visual, "conv1"):
            vision_dim = clip_model.visual.conv1.weight.shape[0]
        else:
            vision_dim = 768
        
        self.n_cls = n_cls
        self.dtype = dtype
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.n_ctx_max = n_ctx_max
        self.prompt_depth_max = prompt_depth_max
        self.adaptive_type = adaptive_type
        self.lambda_sparsity = lambda_sparsity
        self.lambda_depth_smooth = lambda_depth_smooth
        
        # LENGTH gates
        if self.adaptive_type == "depth_only":
            fixed_n_ctx = getattr(trainer_cfg, "FIXED_N_CTX", 4)
            assert 1 <= fixed_n_ctx <= self.n_ctx_max
            self.length_gates = torch.zeros(self.n_ctx_max)
            self.length_gates[:fixed_n_ctx] = 1.0
        else:
            self.length_gates = nn.Parameter(torch.ones(self.n_ctx_max))
        
        # DEPTH weights
        n_depth_weights = self.prompt_depth_max - 1
        if self.adaptive_type == "length_only":
            fixed_depth = getattr(trainer_cfg, "FIXED_PROMPT_DEPTH", 6)
            assert 1 <= fixed_depth <= self.prompt_depth_max
            self.depth_weights = torch.zeros(n_depth_weights)
            self.depth_weights[:fixed_depth - 1] = 1.0
        else:
            self.depth_weights = nn.Parameter(torch.ones(n_depth_weights))
        
        # Context vectors
        ctx_init = getattr(trainer_cfg, "CTX_INIT", None)
        if ctx_init and (self.n_ctx_max) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            device = clip_model.token_embedding.weight.device
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx_max, :]
            if ctx_vectors.shape[0] < self.n_ctx_max:
                remaining = self.n_ctx_max - ctx_vectors.shape[0]
                random_ctx = torch.empty(remaining, text_dim, dtype=dtype)
                nn.init.normal_(random_ctx, std=0.02)
                ctx_vectors = torch.cat([ctx_vectors, random_ctx], dim=0)
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(self.n_ctx_max, text_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx_max)
        
        self.ctx = nn.Parameter(ctx_vectors)
        self.proj = nn.Linear(text_dim, vision_dim, dtype=self.dtype)
        
        # Deeper prompts
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_ctx_max, text_dim))
            for _ in range(self.prompt_depth_max - 1)
        ])
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)
        
        single_layer = nn.Linear(text_dim, self.vision_dim)
        self.compound_prompt_projections = _get_clones(single_layer, self.prompt_depth_max - 1)
        
        # Bidirectional
        self.enable_bidirectional = enable_bidir
        self.cycle_loss_weight = cycle_loss_weight
        
        if self.enable_bidirectional:
            self.l2v_mappings = nn.ModuleList([
                CrossModalMappingNetwork(text_dim, max(text_dim // 2, 64), self.vision_dim)
                for _ in range(self.prompt_depth_max - 1)
            ])
            self.v2l_mappings = nn.ModuleList([
                CrossModalMappingNetwork(self.vision_dim, max(self.vision_dim // 2, 64), text_dim)
                for _ in range(self.prompt_depth_max - 1)
            ])
            self.alpha_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(self.prompt_depth_max - 1)
            ])
            self.beta_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(self.prompt_depth_max - 1)
            ])
        
        # Tokenized prompts
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        device = clip_model.token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx_max:, :])
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = [len(_tokenizer.encode(n)) for n in classnames]
    
    def get_effective_length(self):
        if isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
        else:
            gates = self.length_gates
        if self.training and isinstance(self.length_gates, nn.Parameter):
            return gates.sum()
        else:
            return (gates > 0.5).sum().item()
    
    def get_effective_ctx(self):
        if isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
        else:
            gates = self.length_gates
        if self.training and isinstance(self.length_gates, nn.Parameter):
            effective_ctx = self.ctx * gates.unsqueeze(1)
        else:
            mask = gates > 0.5
            effective_ctx = self.ctx[mask]
        return effective_ctx, gates
    
    def get_depth_probabilities(self):
        if isinstance(self.depth_weights, nn.Parameter):
            return torch.sigmoid(self.depth_weights)
        else:
            return self.depth_weights
    
    def compute_cycle_consistency_loss(self, text_prompts, vision_prompts):
        if not self.enable_bidirectional or len(text_prompts) == 0:
            return torch.tensor(0.0, device=self.ctx.device, dtype=self.dtype)
        cycle_loss = 0.0
        for i in range(len(text_prompts)):
            l2v = self.l2v_mappings[i](text_prompts[i])
            v2l = self.v2l_mappings[i](l2v)
            cycle_loss = cycle_loss + F.mse_loss(text_prompts[i], v2l)
        return cycle_loss / len(text_prompts)
    
    def apply_bidirectional_coupling(self, text_prompts, vision_prompts):
        if not self.enable_bidirectional or len(text_prompts) == 0:
            return text_prompts, vision_prompts
        
        coupled_text_prompts = []
        coupled_vision_prompts = []
        
        for i in range(len(text_prompts)):
            t = text_prompts[i]
            v = vision_prompts[i]
            l2v_mapped = self.l2v_mappings[i](t)
            v2l_mapped = self.v2l_mappings[i](v)
            alpha = torch.sigmoid(self.alpha_params[i])
            beta = torch.sigmoid(self.beta_params[i])
            coupled_text = alpha * t + (1 - alpha) * v2l_mapped
            l2v_updated = self.l2v_mappings[i](coupled_text)
            coupled_vision = beta * v + (1 - beta) * l2v_updated
            coupled_text_prompts.append(coupled_text)
            coupled_vision_prompts.append(coupled_vision)
        
        return coupled_text_prompts, coupled_vision_prompts
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def forward(self):
        effective_ctx, length_gates = self.get_effective_ctx()
        effective_ctx = effective_ctx.type(self.dtype)
        depth_probs = self.get_depth_probabilities()
        
        if effective_ctx.dim() == 2:
            effective_ctx = effective_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = self.construct_prompts(effective_ctx, prefix, suffix)
        prompts = prompts.type(self.dtype)
        
        visual_deep_prompts = []
        for idx, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[idx]))
        
        if self.enable_bidirectional:
            coupled_text_prompts, coupled_vision_prompts = self.apply_bidirectional_coupling(
                self.compound_prompts_text, visual_deep_prompts
            )
            shared_ctx_proj = self.proj(effective_ctx.mean(1))
            return prompts, shared_ctx_proj, coupled_text_prompts, coupled_vision_prompts, depth_probs, length_gates
        else:
            shared_ctx_proj = self.proj(effective_ctx.mean(1))
            text_prompts_list = [param for param in self.compound_prompts_text]
            return prompts, shared_ctx_proj, text_prompts_list, visual_deep_prompts, depth_probs, length_gates
    
    def compute_regularization_loss(self):
        sparsity_loss = 0.0
        if self.adaptive_type in ["full", "depth_only"] and isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
            sparsity_loss = self.lambda_sparsity * gates.sum()
        
        depth_smooth_loss = 0.0
        if self.adaptive_type in ["full", "length_only"] and isinstance(self.depth_weights, nn.Parameter):
            if len(self.depth_weights) > 1:
                depth_diffs = self.depth_weights[1:] - self.depth_weights[:-1]
                depth_smooth_loss = self.lambda_depth_smooth * torch.abs(depth_diffs).sum()
        
        reg_loss = sparsity_loss + depth_smooth_loss
        return reg_loss


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
                dtype=prompts.dtype, device=prompts.device
            )
            prompts = torch.cat([prompts, pad], dim=1)
        elif ctx_len > pos_len:
            prompts = prompts[:, :pos_len, :]
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        combined = [x, compound_prompts_deeper_text, 0]
        if depth_weights is not None:
            combined.append(depth_weights.type(self.dtype))
        
        outputs = self.transformer(combined)
        x = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, trainer_prefix='ADAPTIVE_BIDIR_MAPLE'):
        super().__init__()
        self.prompt_learner = AdaptiveBiDirPromptLearner(cfg, classnames, clip_model, trainer_prefix)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        trainer_cfg = getattr(cfg.TRAINER, trainer_prefix)
        self.cycle_loss_weight = getattr(trainer_cfg, "CYCLE_LOSS_WEIGHT", 0.1)
    
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        prompts, shared_ctx, deep_text, deep_vision, depth_probs, length_gates = self.prompt_learner()
        image = image.type(self.dtype)
        prompts = prompts.type(self.dtype)
        
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_text, depth_probs)
        image_features = self.image_encoder(image, shared_ctx, deep_vision, depth_probs)
        
        eps = 1e-8
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:
            cls_loss = F.cross_entropy(logits, label)
            reg_loss = self.prompt_learner.compute_regularization_loss()
            cycle_loss = self.prompt_learner.compute_cycle_consistency_loss(
                deep_text if isinstance(deep_text, list) else [],
                deep_vision if isinstance(deep_vision, list) else []
            )
            total_loss = cls_loss + reg_loss + (self.cycle_loss_weight * cycle_loss)
            
            def to_scalar(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().mean().item())
                return float(x)
            
            return total_loss, cls_loss, reg_loss, cycle_loss, to_scalar(length_gates), to_scalar(depth_probs)
        
        return logits
    
    def evaluate(self, image):
        with torch.no_grad():
            return self.forward(image)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# ============================================================
# Trainer 1: AdaptiveTrain (train gates normally)
# ============================================================
@TRAINER_REGISTRY.register()
class AdaptiveTrain(TrainerX):
    """Experiment B-1: baseline — gates trained throughout."""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTIVE_TRAIN.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"[AdaptiveTrain] Loading CLIP")
        clip_model = load_clip_to_cpu(cfg, 'ADAPTIVE_TRAIN')
        
        prec = cfg.TRAINER.ADAPTIVE_TRAIN.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()
        
        print("[AdaptiveTrain] Building model with trainable gates")
        self.model = CustomCLIP(cfg, classnames, clip_model, 'ADAPTIVE_TRAIN')
        
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("AdaptiveTrainPromptLearner", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if prec == "amp" else None
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[AdaptiveTrain] Total trainable params: {total_params:,}")
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.ADAPTIVE_TRAIN.PREC
        
        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        def to_scalar(x):
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)

        loss_summary = {
            "total_loss": to_scalar(total_loss),
            "cls_loss": to_scalar(cls_loss),
            "reg_loss": to_scalar(reg_loss),
            "cycle_loss": to_scalar(cycle_loss)
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load best checkpoint by default
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

            # Safe val_result access
            val_result = checkpoint.get("val_result", None)

            # Drop problematic buffer keys
            keys_to_remove = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]

            # Print format depends on val_result
            if val_result is not None:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result={val_result:.1f})"
                )
            else:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result=N/A)"
                )

            # load_state_dict
            try:
                self._models[name].load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
                missing_keys, unexpected_keys = self._models[name].load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")


# ============================================================
# Trainer 2: AdaptiveFreezeEarly (freeze gates early)
# ============================================================
@TRAINER_REGISTRY.register()
class AdaptiveFreezeEarly(TrainerX):
    """Experiment B-2: freeze gates after a few epochs."""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.PREC in ["fp16", "fp32", "amp"]
        self.freeze_epoch = getattr(cfg.TRAINER.ADAPTIVE_FREEZE_EARLY, "FREEZE_EPOCH", 1)
        self.gates_frozen = False
        print(f"[FreezeEarly] Will freeze gates at epoch {self.freeze_epoch}")
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"[AdaptiveFreezeEarly] Loading CLIP")
        clip_model = load_clip_to_cpu(cfg, 'ADAPTIVE_FREEZE_EARLY')
        
        prec = cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()
        
        print("[AdaptiveFreezeEarly] Building model")
        self.model = CustomCLIP(cfg, classnames, clip_model, 'ADAPTIVE_FREEZE_EARLY')
        
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("AdaptiveFreezeEarlyLearner", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if prec == "amp" else None
    
    def before_epoch(self):
        super().before_epoch()
        
        if self.epoch == self.freeze_epoch and not self.gates_frozen:
            print(f"\n{'='*80}")
            print(f"[FreezeEarly] Freezing gates at epoch {self.epoch}")
            print(f"{'='*80}\n")
            
            prompt_learner = self.model.prompt_learner
            if isinstance(prompt_learner.length_gates, nn.Parameter):
                gates_val = torch.sigmoid(prompt_learner.length_gates).detach()
                print(f"Length gates before freeze: {gates_val.cpu().numpy()}")
            if isinstance(prompt_learner.depth_weights, nn.Parameter):
                depth_val = torch.sigmoid(prompt_learner.depth_weights).detach()
                print(f"Depth weights before freeze: {depth_val.cpu().numpy()}")
            
            for name, param in self.model.named_parameters():
                if 'length_gates' in name or 'depth_weights' in name:
                    param.requires_grad_(False)
                    print(f"Frozen: {name}")
            
            self.gates_frozen = True
            self.optim = build_optimizer(self.model, self.cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.PREC
        
        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        def to_scalar(x):
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)

        loss_summary = {
            "total_loss": to_scalar(total_loss),
            "cls_loss": to_scalar(cls_loss),
            "reg_loss": to_scalar(reg_loss),
            "cycle_loss": to_scalar(cycle_loss)
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load best checkpoint by default
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

            # Safe val_result access
            val_result = checkpoint.get("val_result", None)

            # Drop problematic buffer keys
            keys_to_remove = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]

            # Print format depends on val_result
            if val_result is not None:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result={val_result:.1f})"
                )
            else:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result=N/A)"
                )

            # load_state_dict
            try:
                self._models[name].load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
                missing_keys, unexpected_keys = self._models[name].load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")


# ============================================================
# Trainer 3: AdaptiveAlwaysFrozen (gates always frozen)
# ============================================================
@TRAINER_REGISTRY.register()
class AdaptiveAlwaysFrozen(TrainerX):
    """Experiment B-3: gates frozen for all training."""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.PREC in ["fp16", "fp32", "amp"]
        self.fixed_length = getattr(cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN, "FIXED_LENGTH", 4)
        self.fixed_depth = getattr(cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN, "FIXED_DEPTH", 6)
        print(f"[AlwaysFrozen] Using fixed length={self.fixed_length}, depth={self.fixed_depth}")
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"[AdaptiveAlwaysFrozen] Loading CLIP")
        clip_model = load_clip_to_cpu(cfg, 'ADAPTIVE_ALWAYS_FROZEN')
        
        prec = cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()
        
        print("[AdaptiveAlwaysFrozen] Building model with frozen gates")
        self.model = CustomCLIP(cfg, classnames, clip_model, 'ADAPTIVE_ALWAYS_FROZEN')
        
        prompt_learner = self.model.prompt_learner
        
        if hasattr(prompt_learner, 'length_gates'):
            with torch.no_grad():
                prompt_learner.length_gates.zero_()
                prompt_learner.length_gates[:self.fixed_length] = 5.0
            prompt_learner.length_gates.requires_grad_(False)
            print(f"Length gates set to: {torch.sigmoid(prompt_learner.length_gates).cpu().numpy()}")
        
        if hasattr(prompt_learner, 'depth_weights'):
            with torch.no_grad():
                prompt_learner.depth_weights.zero_()
                prompt_learner.depth_weights[:self.fixed_depth-1] = 5.0
            prompt_learner.depth_weights.requires_grad_(False)
            print(f"Depth weights set to: {torch.sigmoid(prompt_learner.depth_weights).cpu().numpy()}")
        
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
            elif 'length_gates' in name or 'depth_weights' in name:
                param.requires_grad_(False)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {len(enabled)} params")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self.register_model("AdaptiveAlwaysFrozenLearner", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if prec == "amp" else None
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.PREC
        
        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        def to_scalar(x):
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)

        loss_summary = {
            "total_loss": to_scalar(total_loss),
            "cls_loss": to_scalar(cls_loss),
            "reg_loss": to_scalar(reg_loss),
            "cycle_loss": to_scalar(cycle_loss)
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load best checkpoint by default
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

            # Safe val_result access
            val_result = checkpoint.get("val_result", None)

            # Drop problematic buffer keys
            keys_to_remove = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]

            # Print format depends on val_result
            if val_result is not None:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result={val_result:.1f})"
                )
            else:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result=N/A)"
                )

            # load_state_dict
            try:
                self._models[name].load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
                missing_keys, unexpected_keys = self._models[name].load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")


# ============================================================
# Trainer 4: AdaptiveExplicitReg (explicit dropout + WD)
# ============================================================
@TRAINER_REGISTRY.register()
class AdaptiveExplicitReg(TrainerX):
    """Experiment B-4: dropout on prompts + stronger weight decay."""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.PREC in ["fp16", "fp32", "amp"]
        self.dropout_rate = getattr(cfg.TRAINER.ADAPTIVE_EXPLICIT_REG, "DROPOUT_RATE", 0.1)
        print(f"[ExplicitReg] Using dropout={self.dropout_rate}")
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"[AdaptiveExplicitReg] Loading CLIP")
        clip_model = load_clip_to_cpu(cfg, 'ADAPTIVE_EXPLICIT_REG')
        
        prec = cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()
        
        print("[AdaptiveExplicitReg] Building model with explicit regularization")
        self.model = CustomCLIP(cfg, classnames, clip_model, 'ADAPTIVE_EXPLICIT_REG')
        
        # Dropout on prompt path
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        
        # Stronger L2
        original_wd = cfg.OPTIM.WEIGHT_DECAY
        cfg.defrost()
        cfg.OPTIM.WEIGHT_DECAY = 0.01
        cfg.freeze()
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("AdaptiveExplicitRegLearner", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if prec == "amp" else None
        print(f"[ExplicitReg] Weight decay increased from {original_wd} to 0.01")
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.PREC
        
        # Dropout on ctx before forward
        if self.model.training:
            prompt_learner = self.model.prompt_learner
            # Dropout on context rows
            original_ctx = prompt_learner.ctx.data.clone()
            prompt_learner.ctx.data = self.dropout(prompt_learner.ctx.data)
        
        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss, cls_loss, reg_loss, cycle_loss, _, _ = self.model(image, label)
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        def to_scalar(x):
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)

        # Restore ctx after stochastic forward
        if self.model.training:
            prompt_learner.ctx.data = original_ctx

        loss_summary = {
            "total_loss": to_scalar(total_loss),
            "cls_loss": to_scalar(cls_loss),
            "reg_loss": to_scalar(reg_loss),
            "cycle_loss": to_scalar(cycle_loss)
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load best checkpoint by default
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

            # Safe val_result access
            val_result = checkpoint.get("val_result", None)

            # Drop problematic buffer keys
            keys_to_remove = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]

            # Print format depends on val_result
            if val_result is not None:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result={val_result:.1f})"
                )
            else:
                print(
                    f"Loading weights to {name} from {model_path} "
                    f"(epoch={epoch}, val_result=N/A)"
                )

            # load_state_dict
            try:
                self._models[name].load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
                missing_keys, unexpected_keys = self._models[name].load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")