"""
Experiment A: parameter-count buffer baseline.
ParamMatched variant: no gates, matched parameter count to adaptive model.
"""
import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import build_model_adaptive

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    """Load CLIP like adaptive_bidir_maple."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'AdaptiveBiDirMaPLe',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.PARAM_MATCHED.FIXED_N_CTX,
        "maple_depth_max": cfg.TRAINER.PARAM_MATCHED.FIXED_PROMPT_DEPTH
    }

    model = build_model_adaptive(state_dict or model.state_dict(), design_details)
    return model


class ParamMatchedPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        trainer_cfg = getattr(cfg.TRAINER, 'PARAM_MATCHED')
        self.n_ctx = trainer_cfg.FIXED_N_CTX
        self.prompt_depth = trainer_cfg.FIXED_PROMPT_DEPTH
        prec = trainer_cfg.PREC

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

        # Base parameters (same role as Adaptive)

        # Context vectors
        ctx_init = getattr(trainer_cfg, "CTX_INIT", None)
        if ctx_init and self.n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            device = clip_model.token_embedding.weight.device
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :]
            if ctx_vectors.shape[0] < self.n_ctx:
                remaining = self.n_ctx - ctx_vectors.shape[0]
                random_ctx = torch.empty(remaining, text_dim, dtype=dtype)
                nn.init.normal_(random_ctx, std=0.02)
                ctx_vectors = torch.cat([ctx_vectors, random_ctx], dim=0)
        else:
            ctx_vectors = torch.empty(self.n_ctx, text_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        self.proj = nn.Linear(text_dim, vision_dim)
        self.proj = self.proj.to(dtype)

        # Deeper prompts
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_ctx, text_dim, dtype=dtype))
            for _ in range(self.prompt_depth - 1)
        ])
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer = nn.Linear(text_dim, self.vision_dim)
        single_layer = single_layer.to(dtype)
        self.compound_prompt_projections = _get_clones(single_layer, self.prompt_depth - 1)

        # Bidirectional
        self.enable_bidirectional = getattr(trainer_cfg, "BIDIRECTIONAL", True)

        if self.enable_bidirectional:
            self.l2v_mappings = nn.ModuleList([
                CrossModalMappingNetwork(text_dim, max(text_dim // 2, 64), self.vision_dim, dtype=dtype)
                for _ in range(self.prompt_depth - 1)
            ])
            self.v2l_mappings = nn.ModuleList([
                CrossModalMappingNetwork(self.vision_dim, max(self.vision_dim // 2, 64), text_dim, dtype=dtype)
                for _ in range(self.prompt_depth - 1)
            ])
            self.alpha_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5, dtype=dtype)) for _ in range(self.prompt_depth - 1)
            ])
            self.beta_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5, dtype=dtype)) for _ in range(self.prompt_depth - 1)
            ])

        # Parameter budget to match Adaptive

        adaptive_n_ctx_max = 4
        adaptive_depth_max = 8

        # Count Adaptive-style parameter budget
        adaptive_params = 0
        adaptive_params += adaptive_n_ctx_max * text_dim
        adaptive_params += text_dim * vision_dim
        adaptive_params += (adaptive_depth_max - 1) * adaptive_n_ctx_max * text_dim
        adaptive_params += (adaptive_depth_max - 1) * text_dim * vision_dim
        adaptive_params += (adaptive_depth_max - 1) * (text_dim * (text_dim // 2) + (text_dim // 2) * vision_dim)
        adaptive_params += (adaptive_depth_max - 1) * (vision_dim * (vision_dim // 2) + (vision_dim // 2) * text_dim)
        adaptive_params += 2 * (adaptive_depth_max - 1)
        adaptive_params += adaptive_n_ctx_max
        adaptive_params += (adaptive_depth_max - 1)

        # ParamMatched params before buffer
        current_params = 0
        current_params += self.n_ctx * text_dim
        current_params += text_dim * vision_dim
        current_params += (self.prompt_depth - 1) * self.n_ctx * text_dim
        current_params += (self.prompt_depth - 1) * text_dim * vision_dim
        if self.enable_bidirectional:
            current_params += (self.prompt_depth - 1) * (text_dim * (text_dim // 2) + (text_dim // 2) * vision_dim)
            current_params += (self.prompt_depth - 1) * (vision_dim * (vision_dim // 2) + (vision_dim // 2) * text_dim)
            current_params += 2 * (self.prompt_depth - 1)

        params_to_compensate = adaptive_params - current_params

        print(f"[ParamMatched] Adaptive params: {adaptive_params:,}")
        print(f"[ParamMatched] Current params (no buffer): {current_params:,}")
        print(f"[ParamMatched] Need to compensate: {params_to_compensate:,}")

        params_per_extra_layer = self.n_ctx * text_dim + text_dim * vision_dim
        if self.enable_bidirectional:
            params_per_extra_layer += (text_dim * (text_dim // 2) + (text_dim // 2) * vision_dim)
            params_per_extra_layer += (vision_dim * (vision_dim // 2) + (vision_dim // 2) * text_dim)
            params_per_extra_layer += 2

        num_extra_layers = params_to_compensate // params_per_extra_layer

        if num_extra_layers > 0:
            print(f"[ParamMatched] Adding {num_extra_layers} extra prompt layers")

            self.extra_compound_prompts_text = nn.ParameterList([
                nn.Parameter(torch.empty(self.n_ctx, text_dim, dtype=dtype))
                for _ in range(num_extra_layers)
            ])
            for p in self.extra_compound_prompts_text:
                nn.init.normal_(p, std=0.02)

            extra_base_layer = nn.Linear(text_dim, self.vision_dim)
            extra_base_layer = extra_base_layer.to(dtype)
            self.extra_compound_prompt_projections = _get_clones(extra_base_layer, num_extra_layers)

            if self.enable_bidirectional:
                self.extra_l2v_mappings = nn.ModuleList([
                    CrossModalMappingNetwork(text_dim, max(text_dim // 2, 64), self.vision_dim, dtype=dtype)
                    for _ in range(num_extra_layers)
                ])
                self.extra_v2l_mappings = nn.ModuleList([
                    CrossModalMappingNetwork(self.vision_dim, max(self.vision_dim // 2, 64), text_dim, dtype=dtype)
                    for _ in range(num_extra_layers)
                ])
                self.extra_alpha_params = nn.ParameterList([
                    nn.Parameter(torch.tensor(0.5, dtype=dtype)) for _ in range(num_extra_layers)
                ])
                self.extra_beta_params = nn.ParameterList([
                    nn.Parameter(torch.tensor(0.5, dtype=dtype)) for _ in range(num_extra_layers)
                ])

            params_to_compensate -= num_extra_layers * params_per_extra_layer

        if params_to_compensate > 0:
            hidden_dim = max(64, int(params_to_compensate / (2 * text_dim)))
            print(f"[ParamMatched] Adding MLP with hidden_dim={hidden_dim} to compensate remaining {params_to_compensate:,} params")

            mlp_seq = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, text_dim)
            )
            self.param_buffer_mlp = mlp_seq.to(dtype)
        else:
            self.param_buffer_mlp = None

        # Tokenized prompts
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        device = clip_model.token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])
        self.tokenized_prompts = tokenized_prompts

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ParamMatched] Final total trainable params: {total_params:,}")
        print(f"[ParamMatched] Target (Adaptive) params: {adaptive_params:,}")
        print(f"[ParamMatched] Difference: {abs(total_params - adaptive_params):,} ({abs(total_params - adaptive_params) / adaptive_params * 100:.2f}%)")

    def forward(self):
        ctx = self.ctx.type(self.dtype)

        # Optional param-buffer MLP
        if self.param_buffer_mlp is not None:
            ctx_processed = self.param_buffer_mlp(ctx)
            ctx = ctx + ctx_processed

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        prompts = prompts.type(self.dtype)

        visual_deep_prompts = []
        for idx, layer in enumerate(self.compound_prompt_projections):
            text_prompt = self.compound_prompts_text[idx].type(self.dtype)
            visual_deep_prompts.append(layer(text_prompt))

        if self.enable_bidirectional:
            coupled_text_prompts, coupled_vision_prompts = self.apply_bidirectional_coupling(
                self.compound_prompts_text, visual_deep_prompts
            )
            shared_ctx_proj = self.proj(ctx.mean(1))
            return prompts, shared_ctx_proj, coupled_text_prompts, coupled_vision_prompts, None
        else:
            shared_ctx_proj = self.proj(ctx.mean(1))
            text_prompts_list = [param for param in self.compound_prompts_text]
            return prompts, shared_ctx_proj, text_prompts_list, visual_deep_prompts, None

    def apply_bidirectional_coupling(self, compound_prompts_text, visual_deep_prompts):
        """Bidirectional coupling (dtype alignment)."""
        text_list = [p.type(self.dtype) if isinstance(p, torch.Tensor) else p for p in compound_prompts_text]
        vision_list = [v.type(self.dtype) if isinstance(v, torch.Tensor) else v for v in visual_deep_prompts]
        return text_list, vision_list

    def compute_cycle_consistency_loss(self, deep_text_list, deep_vision_list):
        """Cycle-consistency loss for bidirectional path."""
        if not self.enable_bidirectional:
            return torch.tensor(0.0, dtype=self.dtype, device=self.ctx.device)
        if not deep_text_list or not deep_vision_list:
            return torch.tensor(0.0, dtype=self.dtype, device=self.ctx.device)
        if len(deep_text_list) != len(deep_vision_list):
            return torch.tensor(0.0, dtype=self.dtype, device=self.ctx.device)

        losses = []
        for i, (t, v) in enumerate(zip(deep_text_list, deep_vision_list)):
            try:
                t = t.type(self.dtype)
                v = v.type(self.dtype)
                l2v = self.l2v_mappings[i](t)
                recon_t = self.v2l_mappings[i](l2v)
                losses.append(F.mse_loss(recon_t, t))
            except Exception:
                losses.append(torch.tensor(0.0, dtype=self.dtype, device=self.ctx.device))

        if len(losses) == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.ctx.device)
        return sum(losses) / len(losses)


class CrossModalMappingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dtype=torch.float16):
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
        self.mapping = self.mapping.to(dtype)

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

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        prompts = prompts.type(self.dtype)
        ctx_len = prompts.shape[1]
        pos_len = self.positional_embedding.shape[0]

        if ctx_len < pos_len:
            pad = torch.zeros(
                (prompts.shape[0], pos_len - ctx_len, prompts.shape[2]),
                dtype=self.dtype,
                device=prompts.device
            )
            prompts = torch.cat([prompts, pad], dim=1)
        elif ctx_len > pos_len:
            prompts = prompts[:, :pos_len, :]

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        if compound_prompts_deeper_text is not None:
            compound_prompts_deeper_text = [p.type(self.dtype) for p in compound_prompts_deeper_text]

        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]

        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = ParamMatchedPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cycle_loss_weight = getattr(cfg.TRAINER.PARAM_MATCHED, "CYCLE_LOSS_WEIGHT", 0.1)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_text, deep_vision, _ = self.prompt_learner()

        image = image.type(self.dtype)
        prompts = prompts.type(self.dtype)
        shared_ctx = shared_ctx.type(self.dtype)

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
        image_features = self.image_encoder(image, shared_ctx, deep_vision, None)

        eps = 1e-8
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)

        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            cls_loss = F.cross_entropy(logits, label)
            cycle_loss = self.prompt_learner.compute_cycle_consistency_loss(
                deep_text if isinstance(deep_text, list) else [],
                deep_vision if isinstance(deep_vision, list) else []
            )
            total_loss = cls_loss + (self.cycle_loss_weight * cycle_loss)
            return total_loss, cls_loss, cycle_loss

        return logits

    def evaluate(self, image):
        with torch.no_grad():
            return self.forward(image)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class ParamMatched(TrainerX):
    """Experiment A: parameter-matched baseline (no gates)."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PARAM_MATCHED.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        prec = getattr(cfg.TRAINER.PARAM_MATCHED, "PREC", None)
        if prec in ["fp32", "amp"]:
            clip_model.float()

        print("Building ParamMatched CLIP (Fixed structure + param buffer)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        name_to_update = "prompt_learner"
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

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("ParamMatchedPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if prec == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PARAM_MATCHED.PREC

        if prec == "amp":
            with autocast():
                total_loss, cls_loss, cycle_loss = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss, cls_loss, cycle_loss = self.model(image, label)
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        loss_summary = {
            "total_loss": float(total_loss.item()),
            "cls_loss": float(cls_loss.item()),
            "cycle_loss": float(cycle_loss.item())
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

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint.get("val_result", None)

            keys_to_remove = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
            for k in keys_to_remove:
                if k in state_dict:
                    del state_dict[k]

            if val_result is not None:
                print(f"Loading weights to {name} from {model_path} (epoch={epoch}, val_result={val_result:.1f})")
            else:
                print(f"Loading weights to {name} from {model_path} (epoch={epoch}, val_result=N/A)")

            try:
                self._models[name].load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
                missing_keys, unexpected_keys = self._models[name].load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")