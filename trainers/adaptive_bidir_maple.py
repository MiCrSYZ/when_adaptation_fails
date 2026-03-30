import os.path as osp
import copy
import math
from collections import OrderedDict
from datetime import datetime
import json
import os
import types
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import build_model_adaptive

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

    #trainer_cfg = getattr(cfg.TRAINER, trainer_name)

    # design_details defaults; adaptive map length / depth max from cfg.ADAPTIVE
    design_details = {
        "trainer": 'AdaptiveBiDirMaPLe',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.N_CTX_MAX,
        "maple_depth_max": cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PROMPT_DEPTH_MAX
    }

    model = build_model_adaptive(state_dict or model.state_dict(), design_details)

    # Monkey-patch transformer.forward to accept extra *args/**kwargs and ignore them, avoiding
    # TypeError when transformer(...) is called with named args (e.g. depth_weights=...).
    try:
        def _transformer_forward(self, x, *args, **kwargs):
            # x may be a list: [x, deep_prompts, counter, depth_probs]; pass through to resblocks as before.
            return self.resblocks(x)

        model.transformer.forward = types.MethodType(_transformer_forward, model.transformer)
    except Exception as e:
        print("Warning: failed to patch transformer.forward to accept kwargs:", e)

    return model


class CrossModalMappingNetwork(nn.Module):
    #Bidirectional coupling function
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, depth_weights=None):
        # note: transformer for Adaptive variant expects combined inputs (see model.py)
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
        combined = [x]
        if compound_prompts_deeper_text is not None:
            compound_prompts_deeper_text = [p.type(x.dtype) for p in compound_prompts_deeper_text]
            combined.append(compound_prompts_deeper_text)
        combined.append(0)  # placeholder counter
        if depth_weights is not None:
            depth_weights = depth_weights.type(self.dtype)

        outputs = self.transformer(combined, depth_weights=depth_weights)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AdaptiveBiDirPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx_max = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.N_CTX_MAX
        prompt_depth_max = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PROMPT_DEPTH_MAX
        prec = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PREC
        lambda_sparsity = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_SPARSITY
        lambda_depth_smooth = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_DEPTH_SMOOTH
        adaptive_type = cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.TYPE
        enable_bidir = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "BIDIRECTIONAL", True)
        cycle_loss_weight = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "CYCLE_LOSS_WEIGHT", 0.1)

        classnames = [name.replace("_", " ") for name in classnames]
        n_cls = len(classnames)
        dtype = clip_model.dtype
        # text dimension from clip transformer width
        text_dim = clip_model.ln_final.weight.shape[0]

        vision_dim = None
        # Prefer ViT width; else conv1 out channels; else default CLIP vision dim.
        if hasattr(clip_model.visual, "transformer") and hasattr(clip_model.visual.transformer, "width"):
            vision_dim = clip_model.visual.transformer.width
        elif hasattr(clip_model.visual, "conv1"):
            vision_dim = clip_model.visual.conv1.weight.shape[0]
        if vision_dim is None:
            vision_dim = 768
            print(f"Warning: Auto-get vision_dim failed, use default {vision_dim} (check your CLIP model type)")

        # store
        self.n_cls = n_cls
        self.dtype = dtype
        self.text_dim = text_dim
        self.vision_dim = vision_dim

        # adaptive settings
        self.n_ctx_max = n_ctx_max
        self.prompt_depth_max = prompt_depth_max
        assert self.prompt_depth_max >= 1, "PROMPT_DEPTH_MAX must be >= 1"
        self.adaptive_type = adaptive_type

        self.lambda_sparsity = lambda_sparsity
        self.lambda_depth_smooth = lambda_depth_smooth

        # LENGTH handling
        if self.adaptive_type == "depth_only":
            # fixed number of ctx
            fixed_n_ctx = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "FIXED_N_CTX", 4)
            assert 1 <= fixed_n_ctx <= self.n_ctx_max
            self.length_gates = torch.zeros(self.n_ctx_max)
            self.length_gates[:fixed_n_ctx] = 1.0
            print(f"[depth_only] using fixed length {fixed_n_ctx}")
        else:
            # learnable gates
            self.length_gates = nn.Parameter(torch.ones(self.n_ctx_max))
            print(f"[length/adaptive] learnable length gates (n_ctx_max={self.n_ctx_max})")

        # DEPTH handling (probabilities for inserting deeper prompts)
        n_depth_weights = self.prompt_depth_max - 1
        if self.adaptive_type == "length_only":
            fixed_depth = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "FIXED_PROMPT_DEPTH", 6)
            assert 1 <= fixed_depth <= self.prompt_depth_max
            self.depth_weights = torch.zeros(n_depth_weights)
            # set first (fixed_depth - 1) weights to 1
            self.depth_weights[:fixed_depth - 1] = 1.0
            print(f"[length_only] using fixed depth {fixed_depth}")
        else:
            self.depth_weights = nn.Parameter(torch.ones(n_depth_weights))
            print(f"[depth/adaptive] learnable depth weights (prompt_depth_max={self.prompt_depth_max})")

        # initialize ctx vectors (token-level context)
        ctx_init = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "CTX_INIT", None)
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

        print("AdaptiveBiDirMaPLe design initialized")
        print(f"Initial context prefix: '{prompt_prefix}' | n_ctx_max={self.n_ctx_max} | prompt_depth_max={self.prompt_depth_max}")
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx_max, text_dim)

        # projection for shared ctx (to vision-transformer internal projection if needed)
        # project from text_dim -> vision internal dim (use a safe default 768 if unknown)
        self.shared_proj_out = getattr(clip_model.visual, "ln_post", None)
        # linear projection used for sharing (text->vision shared ctx)
        self.proj = nn.Linear(text_dim, vision_dim, dtype=self.dtype)

        # compound prompts: textual deeper prompts (list of param matrices of size [n_ctx_max, some_dim])
        # We choose to keep textual deeper prompt dim = text_dim (consistent)
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_ctx_max, text_dim))
            for _ in range(self.prompt_depth_max - 1)
        ])
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        # compound prompt projections to obtain visual deep prompts (map textual contexts to vision space)
        # mapping from text_dim -> vision_dim
        single_layer = nn.Linear(text_dim, self.vision_dim)
        self.compound_prompt_projections = _get_clones(single_layer, self.prompt_depth_max - 1)

        # Bidirectional mapping networks (if enabled)
        self.enable_bidirectional = enable_bidir
        self.cycle_loss_weight = cycle_loss_weight

        if self.enable_bidirectional:
            # Language->Vision: text_dim -> vision_dim
            self.l2v_mappings = nn.ModuleList([
                CrossModalMappingNetwork(text_dim, max(text_dim // 2, 64), self.vision_dim)
                for _ in range(self.prompt_depth_max - 1)
            ])
            # Vision->Language: vision_dim -> text_dim
            self.v2l_mappings = nn.ModuleList([
                CrossModalMappingNetwork(self.vision_dim, max(self.vision_dim // 2, 64), text_dim)
                for _ in range(self.prompt_depth_max - 1)
            ])
            # learnable mixing params per depth layer (scalar)
            self.alpha_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(self.prompt_depth_max - 1)
            ])
            self.beta_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(self.prompt_depth_max - 1)
            ])

        # prepare tokenized templates for classes (prefix/suffix buffers)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        device = clip_model.token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx_max:, :])  # rest after ctx tokens

        self.tokenized_prompts = tokenized_prompts
        self.name_lens = [len(_tokenizer.encode(n)) for n in classnames]

    # --------- adaptive helpers ----------
    def get_effective_length(self):
        if isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
        else:
            gates = self.length_gates
        if self.training and isinstance(self.length_gates, nn.Parameter):
            # soft count during training
            return gates.sum()
        else:
            return (gates > 0.5).sum().item()

    def get_effective_ctx(self):
        if isinstance(self.length_gates, nn.Parameter):
            gates = torch.sigmoid(self.length_gates)
        else:
            gates = self.length_gates
        # Ensure gates and ctx live on the same device.
        # Some diagnostic code paths may temporarily move/replace tensors,
        # and we want training/eval to be robust.
        gates = gates.to(self.ctx.device)
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

    # --------- bidir utilities ----------
    def compute_cycle_consistency_loss(self, text_prompts, vision_prompts):
        if not self.enable_bidirectional or len(text_prompts) == 0:
            return torch.tensor(0.0, device=self.ctx.device, dtype=self.dtype)
        cycle_loss = 0.0
        for i in range(len(text_prompts)):
            # map text->vision->text
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
            t = text_prompts[i]  # (n_ctx_max, text_dim)
            v = vision_prompts[i]  # mapped visual context (vision_dim)
            # map t->vision, v->text
            l2v_mapped = self.l2v_mappings[i](t)  # (n_ctx_max, vision_dim)
            v2l_mapped = self.v2l_mappings[i](v)  # (n_ctx_max, text_dim)
            alpha = torch.sigmoid(self.alpha_params[i])
            beta = torch.sigmoid(self.beta_params[i])
            coupled_text = alpha * t + (1 - alpha) * v2l_mapped
            # re-map to vision side
            l2v_updated = self.l2v_mappings[i](coupled_text)
            coupled_vision = beta * v + (1 - beta) * l2v_updated
            coupled_text_prompts.append(coupled_text)
            coupled_vision_prompts.append(coupled_vision)
        return coupled_text_prompts, coupled_vision_prompts

    # --------- main forward ----------
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

        # expand ctx to classes
        if effective_ctx.dim() == 2:
            effective_ctx = effective_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # main prompts used by text encoder: prefix + effective_ctx + suffix
        prompts = self.construct_prompts(effective_ctx, prefix, suffix)
        prompts = prompts.type(self.dtype)

        # produce visual deep prompts from textual deeper params
        visual_deep_prompts = []
        for idx, layer in enumerate(self.compound_prompt_projections):
            # each compound_prompts_text[idx] shape: (n_ctx_max, text_dim)
            visual_deep_prompts.append(layer(self.compound_prompts_text[idx]))  # to vision_dim

        if self.enable_bidirectional:
            coupled_text_prompts, coupled_vision_prompts = self.apply_bidirectional_coupling(
                self.compound_prompts_text, visual_deep_prompts
            )
            shared_ctx_proj = self.proj(effective_ctx.mean(1))  # project per-class shared ctx (n_cls, text_dim) -> (n_cls, text_dim)
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


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = AdaptiveBiDirPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # loss weights (cycle)
        self.cycle_loss_weight = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "CYCLE_LOSS_WEIGHT", 0.1)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, depth_probs, length_gates = self.prompt_learner()
        image = image.type(self.dtype)
        prompts = prompts.type(self.dtype)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text, depth_probs)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, depth_probs)

        eps = 1e-8
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)
        logits = logit_scale * image_features @ text_features.t()

        # Training branch returns losses, but for diagnostics we sometimes call
        # `model(images)` in `model.train()` mode *without* labels. In that case
        # we should still return logits for accuracy computation.
        if self.prompt_learner.training and label is not None:
            cls_loss = F.cross_entropy(logits, label)
            reg_loss = self.prompt_learner.compute_regularization_loss()

            if self.prompt_learner.enable_bidirectional:
                cycle_loss = self.prompt_learner.compute_cycle_consistency_loss(
                    deep_compound_prompts_text if isinstance(deep_compound_prompts_text, list) else [],
                    deep_compound_prompts_vision if isinstance(deep_compound_prompts_vision, list) else []
                )
            else:
                cycle_loss = torch.tensor(0.0, device=logits.device)

            total_loss = cls_loss + reg_loss + (self.cycle_loss_weight * cycle_loss if cycle_loss is not None else 0.0)

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

            return total_loss, cls_loss, reg_loss, cycle_loss, to_scalar(length_gates), to_scalar(depth_probs)

        return logits

    def evaluate(self, image):
        with torch.no_grad():
            return self.forward(image)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class AdaptiveBiDirMaPLe(TrainerX):
    def check_cfg(self, cfg):
        prec_ok = False
        if hasattr(cfg.TRAINER, "ADAPTIVE_BIDIR_MAPLE") and cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PREC in ["fp16", "fp32", "amp"]:
            prec_ok = True
        assert prec_ok, "Precision must be set in cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE"

        self.metrics = {
            'train_loss': [],
            'cls_loss': [],
            'reg_loss': [],
            'cycle_loss': [],
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

        prec = getattr(cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE, "PREC", None)
        if prec in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP with Adaptive + BiDir MaPLe")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Freezing backbone encoders, enabling prompt learner params")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # show which params are trainable
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        gate_params = []
        mapping_params = []
        prompt_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if any(x in name for x in ['length_gates', 'depth_weights']):
                gate_params.append(param)
                print(f"[Gate] {name}")
            elif any(x in name for x in ['l2v_mappings', 'v2l_mappings', 'alpha_params', 'beta_params']):
                mapping_params.append(param)
                print(f"[Mapping] {name}")
            else:
                prompt_params.append(param)
                print(f"[Prompt] {name}")

        # Build param groups
        base_lr = cfg.OPTIM.LR
        param_groups = [
            {'params': prompt_params, 'lr': base_lr, 'name': 'prompt'},
            {'params': mapping_params, 'lr': base_lr * 1, 'name': 'mapping'},
            {'params': gate_params, 'lr': base_lr * 1, 'name': 'gate'}
        ]

        param_groups = [g for g in param_groups if len(g['params']) > 0]

        # Build optimizer manually so param groups are preserved
        if cfg.OPTIM.NAME.lower() == "sgd":
            self.optim = torch.optim.SGD(
                param_groups,
                momentum=getattr(cfg.OPTIM, 'MOMENTUM', 0.9),
                weight_decay=getattr(cfg.OPTIM, 'WEIGHT_DECAY', 5e-4),
                nesterov=True
            )
        elif cfg.OPTIM.NAME.lower() == "adam":
            self.optim = torch.optim.Adam(
                param_groups,
                betas=(0.9, 0.999),
                weight_decay=getattr(cfg.OPTIM, 'WEIGHT_DECAY', 5e-4)
            )
        elif cfg.OPTIM.NAME.lower() == "adamw":
            self.optim = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                weight_decay=getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.01)
            )
        else:
            self.optim = build_optimizer(self.model, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("AdaptiveBiDirPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if prec == "amp" else None
        self.viz_dir = osp.join(cfg.OUTPUT_DIR, "visualizations_adaptive_bidir")
        os.makedirs(self.viz_dir, exist_ok=True)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PREC

        # Freeze gate params for the first two epochs
        warmup_epochs = 2
        if self.epoch < warmup_epochs:
            for name, param in self.model.named_parameters():
                if 'length_gates' in name or 'depth_weights' in name:
                    param.grad = None

        if prec == "amp":
            with autocast():
                total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            total_loss, cls_loss, reg_loss, cycle_loss, length_gates, depth_probs = model(image, label)
            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optim.step()

        with torch.no_grad():
            prompt_learner = self.model.prompt_learner
            eff_length = prompt_learner.get_effective_length()
            depth_probs_v = prompt_learner.get_depth_probabilities()

            def to_scalar(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().mean().item())
                elif isinstance(x, (np.ndarray, np.generic)):
                    return float(np.mean(x))
                else:
                    return float(x)

            if not hasattr(self, 'current_epoch_metrics'):
                self.current_epoch_metrics = {
                    'train_loss': [],
                    'cls_loss': [],
                    'reg_loss': [],
                    'cycle_loss': [],
                    'effective_length': [],
                    'avg_depth_prob': [],
                    'max_depth_prob': [],
                    'min_depth_prob': []
                }

            self.current_epoch_metrics['train_loss'].append(to_scalar(total_loss))
            self.current_epoch_metrics['cls_loss'].append(to_scalar(cls_loss))
            self.current_epoch_metrics['reg_loss'].append(to_scalar(reg_loss))
            self.current_epoch_metrics['cycle_loss'].append(to_scalar(cycle_loss))
            self.current_epoch_metrics['effective_length'].append(to_scalar(eff_length))
            self.current_epoch_metrics['avg_depth_prob'].append(to_scalar(depth_probs_v.mean()))
            self.current_epoch_metrics['max_depth_prob'].append(to_scalar(depth_probs_v.max()))
            self.current_epoch_metrics['min_depth_prob'].append(to_scalar(depth_probs_v.min()))

        loss_summary = {
            "total_loss": to_scalar(total_loss),
            "cls_loss": to_scalar(cls_loss),
            "reg_loss": to_scalar(reg_loss),
            "cycle_loss": to_scalar(cycle_loss),
            "effective_length": to_scalar(eff_length),
            "avg_depth_prob": to_scalar(depth_probs_v.mean()),
            "max_depth_prob": to_scalar(depth_probs_v.max()),
            "min_depth_prob": to_scalar(depth_probs_v.min()),
            "length_gates_mean": to_scalar(length_gates),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def before_epoch(self):
        super().before_epoch()
        self.current_epoch_metrics = {
            'train_loss': [],
            'cls_loss': [],
            'reg_loss': [],
            'cycle_loss': [],
            'effective_length': [],
            'avg_depth_prob': [],
            'max_depth_prob': [],
            'min_depth_prob': []
        }

    def after_step(self):
        if hasattr(self, 'outputs') and self.outputs:
            for key in ['total_loss', 'cls_loss', 'reg_loss', 'cycle_loss',
                        'effective_length', 'avg_depth_prob', 'max_depth_prob', 'min_depth_prob']:
                if key in self.outputs:
                    metric_key = 'train_loss' if key == 'total_loss' else key
                    value = self.outputs[key]

                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.current_epoch_metrics[metric_key].append(float(value))
                    else:
                        print(f"Warning: Invalid value for {key}: {value}")

    def after_epoch(self):
        super().after_epoch()

        diagnostic_epochs = [1, 4, 9]

        if self.epoch in diagnostic_epochs:
            print(f"\n{'=' * 80}")
            print(f"Running diagnostics (Epoch {self.epoch}/{self.max_epoch})")
            print(f"{'=' * 80}\n")

            try:
                from diagnostics.diagnostic_experiments import run_diagnostics
                diagnostic_results = run_diagnostics(self, epoch=self.epoch)
                self._print_diagnostic_warnings(diagnostic_results)

            except Exception as e:
                print(f"Diagnostics failed: {e}")
                import traceback
                traceback.print_exc()

            if self.epoch in [4, 9]:
                from diagnostics.advanced_experiments import run_advanced_diagnostics
                print(f"\n{'=' * 80}")
                print(f"Running phase-2 advanced diagnostics (Epoch {self.epoch})")
                print(f"{'=' * 80}\n")
                run_advanced_diagnostics(self, epoch=self.epoch)

        self._aggregate_epoch_metrics()

    def _print_diagnostic_warnings(self, results):
        """Print warnings from diagnostic results."""
        print("\n" + "=" * 80)
        print("Diagnostic summary")
        print("=" * 80)

        warnings = []

        if 'A1_gradient_ratio' in results:
            mag_gap = results['A1_gradient_ratio'].get('magnitude_gap', 0)
            if mag_gap > 2:
                warnings.append(
                    f"Severe gradient imbalance: gate grad ~{mag_gap:.1f} orders smaller than prompt"
                )
                warnings.append(f"Suggestion: raise gate LR toward {self.cfg.OPTIM.LR * 100:.4f}")

        if 'B2_gradient_cancellation' in results:
            cancellations = [
                s['cancellation_ratio']
                for s in results['B2_gradient_cancellation'].values()
                if isinstance(s, dict) and 'cancellation_ratio' in s
            ]
            if cancellations:
                avg_cancel = sum(cancellations) / len(cancellations)
                if avg_cancel > 0.3:
                    warnings.append(f"Strong gradient cancellation: avg rate {avg_cancel * 100:.1f}%")
                    warnings.append("Suggestion: lower regularization or remove cycle loss")

        if 'C2_fixed_vs_adaptive' in results and 'performance_gap' in results['C2_fixed_vs_adaptive']:
            gap = results['C2_fixed_vs_adaptive']['performance_gap'].get('adaptive_vs_uniform', 0)
            if abs(gap) < 1.0:
                warnings.append(f"Limited benefit from adaptation: gap vs uniform ~{gap:.2f}%")
                warnings.append("Suggestion: consider fixed MaPLe")

        if warnings:
            print("\nIssues detected:")
            for w in warnings:
                print(w)
        else:
            print("\nNo major issues; training looks normal.")

        print("=" * 80 + "\n")

    def _aggregate_epoch_metrics(self):
        avg = {}
        for k, v in self.current_epoch_metrics.items():
            if not v:
                avg[k] = float('nan')
            else:
                try:
                    cleaned = [float(x) for x in v if not np.isnan(x)]
                    if cleaned:
                        avg[k] = float(np.mean(cleaned))
                    else:
                        avg[k] = float('nan')
                except Exception as e:
                    print(f"[ERROR] processing metric {k}: {e}")
                    avg[k] = float('nan')

        eval_period = getattr(self.cfg.TRAIN, "EVAL_PERIOD", 1)
        if self.epoch % eval_period == 0:
            try:
                val_acc = self.test(split="val")
                avg['val_acc'] = val_acc
            except Exception as e:
                print(f"[ERROR] Validation failed: {e}")
                avg['val_acc'] = float('nan')
        else:
            avg['val_acc'] = float('nan')

        for k, val in avg.items():
            if k in self.metrics:
                self.metrics[k].append(val)

        self.metrics['epochs'].append(self.epoch)

        self.save_metrics()
        if self.epoch % 5 == 0 or self.epoch == self.max_epoch:
            self.plot_metrics()

    def save_metrics(self):
        metrics_path = osp.join(self.viz_dir, "training_metrics_adaptive_bidir.json")
        serializable = {}
        for k, v in self.metrics.items():
            if isinstance(v, list):
                serializable[k] = []
                for item in v:
                    if isinstance(item, (np.integer, np.int64)):
                        serializable[k].append(int(item))
                    elif isinstance(item, (np.floating, np.float32, np.float64)):
                        serializable[k].append(float(item))
                    elif isinstance(item, (int, float)):
                        serializable[k].append(item)
                    else:
                        serializable[k].append(float('nan'))
            else:
                serializable[k] = v
        with open(metrics_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def plot_metrics(self):
        if not self.metrics['epochs']:
            return
        epochs = self.metrics['epochs']
        try:
            plt.figure(figsize=(12, 10))

            # Loss plot
            if self.metrics['train_loss'] and not all(np.isnan(self.metrics['train_loss'])):
                plt.subplot(3, 1, 1)
                plt.plot(epochs, self.metrics['train_loss'], label='train_loss')
                plt.plot(epochs, self.metrics['cls_loss'], label='cls_loss')
                plt.plot(epochs, self.metrics['reg_loss'], label='reg_loss')
                plt.plot(epochs, self.metrics['cycle_loss'], label='cycle_loss')
                plt.legend()
                plt.grid(True)
                plt.title('Training Losses')

            # Effective length plot
            if self.metrics['effective_length'] and not all(np.isnan(self.metrics['effective_length'])):
                plt.subplot(3, 1, 2)
                plt.plot(epochs, self.metrics['effective_length'], label='effective_length')
                plt.legend()
                plt.grid(True)
                plt.title('Effective Length')

            # Validation accuracy plot
            if self.metrics['val_acc'] and not all(np.isnan(self.metrics['val_acc'])):
                plt.subplot(3, 1, 3)
                plt.plot(epochs, self.metrics['val_acc'], label='val_acc')
                plt.legend()
                plt.grid(True)
                plt.title('Validation Accuracy')

            plt.tight_layout()
            plot_path = osp.join(self.viz_dir, f'metrics_epoch_{self.epoch}.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in self.dm.val_loader:
                images, labels = self.parse_batch_train(batch)
                logits = self.model.evaluate(images)
                loss = F.cross_entropy(logits, labels)
                acc = compute_accuracy(logits, labels)[0].item()
                total_loss += loss.item() * labels.size(0)
                total_acc += acc * labels.size(0)
                total_count += labels.size(0)
        self.model.train()
        return total_loss / total_count, total_acc / total_count

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model skipped (no directory provided)")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):

                raise FileNotFoundError(f"Model not found at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            # remove prefix buffers to avoid mismatch
            keys_to_remove = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]

            print(f"Loading weights to {name} from {model_path} (epoch={checkpoint.get('epoch', -1)})")
            try:
                self._models[name].load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
                missing_keys, unexpected_keys = self._models[name].load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")