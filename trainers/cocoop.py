import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoCoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        # Gating options
        self.use_gate = bool(getattr(cfg.TRAINER.COCOOP, "USE_GATE", False))
        self.gate_type = getattr(cfg.TRAINER.COCOOP, "GATE_TYPE", "scalar")
        self.gate_init_std = float(getattr(cfg.TRAINER.COCOOP, "GATE_INIT_STD", 0.02))
        self.gate_mode = getattr(cfg.TRAINER.COCOOP, "GATE_MODE", "static")  # static|conditional
        self.param_matched = bool(getattr(cfg.TRAINER.COCOOP, "PARAM_MATCHED", False))

        if self.use_gate and self.gate_mode == "static":
            if self.gate_type == "scalar":
                gate_logits = torch.zeros(n_ctx, dtype=dtype)
                if self.gate_init_std > 0:
                    nn.init.normal_(gate_logits, mean=0.0, std=self.gate_init_std)
                self.gate_logits = nn.Parameter(gate_logits)
            elif self.gate_type == "vector":
                gate_logits = torch.zeros(n_ctx, ctx_dim, dtype=dtype)
                if self.gate_init_std > 0:
                    nn.init.normal_(gate_logits, mean=0.0, std=self.gate_init_std)
                self.gate_logits = nn.Parameter(gate_logits)
            else:
                raise ValueError("Unsupported GATE_TYPE for CoCoOp: {}".format(self.gate_type))

        if self.use_gate and self.gate_mode == "conditional":
            if self.gate_type == "scalar":
                self.gate_head = nn.Linear(vis_dim, n_ctx)
            else:
                self.gate_head = nn.Linear(vis_dim, n_ctx * ctx_dim)
            if cfg.TRAINER.COCOOP.PREC == "fp16":
                self.gate_head.half()

        if self.param_matched and not self.use_gate:
            # parameter-matched bias
            if self.gate_type == "scalar":
                self.pm_bias = nn.Parameter(torch.zeros(n_ctx, 1, dtype=dtype))
            else:
                self.pm_bias = nn.Parameter(torch.zeros(n_ctx, ctx_dim, dtype=dtype))

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        # Optimized: process one sample at a time to reduce memory
        batch_size = im_features.size(0)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        
        # Compute bias once for all samples
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        
        # Apply gating or parameter-matched baseline
        if self.use_gate and self.gate_mode == "static":
            if self.gate_type == "scalar":
                p = torch.sigmoid(self.gate_logits).unsqueeze(-1)  # [L,1]
            else:
                p = torch.sigmoid(self.gate_logits)  # [L,D]
            ctx_mod = p * self.ctx  # [L,D]
            ctx_base = ctx_mod  # (n_ctx, ctx_dim)
        elif self.use_gate and self.gate_mode == "conditional":
            # Conditional gating: compute gate logits for all samples at once for logging
            if self.gate_type == "scalar":
                gate_logits = self.gate_head(im_features)  # (batch, L)
                # Save for logging: average across batch for statistics
                self._last_conditional_gate_logits = gate_logits.detach()
            else:
                gate_logits = self.gate_head(im_features)  # (batch, L*D)
                self._last_conditional_gate_logits = gate_logits.detach().view(batch_size, self.n_ctx, -1)
            ctx_base = ctx
        elif self.param_matched and not self.use_gate:
            bias_pm = self.pm_bias if self.pm_bias.shape[-1] == ctx.shape[-1] else self.pm_bias.repeat(1, ctx.shape[-1])
            ctx_base = self.ctx + bias_pm
        else:
            ctx_base = ctx

        # Process one sample at a time to reduce peak memory
        prompts = []
        for i in range(batch_size):
            # Get bias for this sample
            bias_i = bias[i:i+1]  # (1, 1, ctx_dim)
            
            # Apply conditional gating if needed
            if self.use_gate and self.gate_mode == "conditional":
                imf_i = im_features[i:i+1]  # (1, vis_dim)
                if self.gate_type == "scalar":
                    logits_i = self.gate_head(imf_i)  # (1, L)
                    p_i = torch.sigmoid(logits_i).unsqueeze(-1)  # (1, L, 1)
                else:
                    logits_i = self.gate_head(imf_i)  # (1, L*D)
                    p_i = torch.sigmoid(logits_i.view(1, self.n_ctx, -1))  # (1, L, D)
                ctx_i = (ctx_base.unsqueeze(0) + bias_i) * p_i  # (1, n_ctx, ctx_dim)
            else:
                # Static gating or no gating: ctx_base already has gating applied if needed
                ctx_i = ctx_base.unsqueeze(0) + bias_i  # (1, n_ctx, ctx_dim)
            
            # Expand to all classes and construct prompts (n_cls, n_tkn, ctx_dim)
            ctx_i_expanded = ctx_i.expand(self.n_cls, -1, -1)  # (n_cls, n_ctx, ctx_dim)
            pts_i = self.construct_prompts(ctx_i_expanded, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        
        # Stack only when needed (lazy evaluation could further help, but this is simpler)
        prompts = torch.stack(prompts)  # (batch, n_cls, n_tkn, ctx_dim)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # Memory optimization: use gradient checkpointing for text encoder
        self.use_checkpoint = getattr(cfg.TRAINER.COCOOP, "USE_CHECKPOINT", True)
        # Memory optimization: process samples sequentially instead of batched
        self.sequential_processing = getattr(cfg.TRAINER.COCOOP, "SEQUENTIAL_PROCESSING", True)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        # Memory-efficient: process one sample at a time for text encoding
        # This avoids creating large intermediate tensors (batch, n_cls, n_tkn, dim)
        if self.sequential_processing:
            batch_size = image.size(0)
            logits = []
            
            for i in range(batch_size):
                pts_i = prompts[i]  # (n_cls, n_tkn, ctx_dim)
                imf_i = image_features[i:i+1]  # (1, dim)
                
                # Text encoding: this is the memory-intensive part
                # Use gradient checkpointing if training to reduce memory
                if self.prompt_learner.training and self.use_checkpoint:
                    try:
                        # Checkpoint the text encoder to save memory
                        text_features = torch.utils.checkpoint.checkpoint(
                            self.text_encoder, pts_i, tokenized_prompts, use_reentrant=False
                        )
                    except Exception:
                        # Fallback: if checkpointing fails, use normal forward
                        text_features = self.text_encoder(pts_i, tokenized_prompts)
                else:
                    text_features = self.text_encoder(pts_i, tokenized_prompts)
                
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()  # (1, n_cls)
                logits.append(l_i.squeeze(0))  # (n_cls,)
            
            logits = torch.stack(logits)  # (batch, n_cls)
        else:
            # Original batched processing (uses more memory but faster)
            logits = []
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)

        if self.prompt_learner.training and label is not None:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # Build optimizer with optional gate lr
        base_lr = float(getattr(cfg.OPTIM, "LR", 1e-3))
        gate_lr_mul = float(getattr(cfg.TRAINER.COCOOP, "GATE_LR_MUL", 1.0))
        param_groups = []
        gate_like = []
        others = []
        for n, p in self.model.prompt_learner.named_parameters():
            if not p.requires_grad:
                continue
            if ("gate_logits" in n) or ("gate_head" in n):
                gate_like.append(p)
            else:
                others.append(p)
        if gate_like:
            param_groups.append({"params": gate_like, "lr": base_lr * gate_lr_mul})
        if others:
            param_groups.append({"params": others, "lr": base_lr})
        optim_target = param_groups if param_groups else self.model.prompt_learner
        self.optim = build_optimizer(optim_target, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
        self._gate_reg_lambda = float(getattr(cfg.TRAINER.COCOOP, "GATE_REG_LAMBDA", 0.0))
        self._equalize_grad = bool(getattr(cfg.TRAINER.COCOOP, "EQUALIZE_GRAD", False))
        self._alpha_max = float(getattr(cfg.TRAINER.COCOOP, "ALPHA_MAX", 10.0))

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
                loss = loss + self._compute_gate_reg()
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            self._maybe_equalize_gate_gradients()
            # Save gradient before step (gradients will be cleared/modified by step)
            gate_grad_norm = self._get_gate_grad_norm()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            loss = loss + self._compute_gate_reg()
            optim.zero_grad()
            loss.backward()
            self._maybe_equalize_gate_gradients()
            # Save gradient before step (gradients will be cleared/modified by step)
            gate_grad_norm = self._get_gate_grad_norm()
            optim.step()

        # compute train accuracy by getting logits in no_grad
        # Temporarily set to eval mode to get logits without label
        # Handle DataParallel wrapper if present
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        was_training = actual_model.training
        with torch.no_grad():
            actual_model.eval()
            logits = model(image)
            acc = compute_accuracy(logits, label)[0].item()
            actual_model.train(was_training)
        loss_summary = {"loss": loss.item(), "acc": acc}

        # Log gate metrics to TensorBoard and return them for logging
        gate_metrics = self._log_gate_metrics(gate_grad_norm=gate_grad_norm)
        if gate_metrics:
            loss_summary.update(gate_metrics)

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

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    # ---------- Gating helpers ----------
    def _gate_params_and_p(self):
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        if hasattr(pl, "gate_logits"):
            p = torch.sigmoid(pl.gate_logits)
            return pl.gate_logits, p
        if hasattr(pl, "gate_head"):
            # cannot cache per-batch p here reliably; skip direct p logging if absent
            return pl.gate_head, None
        return None, None

    def _compute_gate_reg(self) -> torch.Tensor:
        if self._gate_reg_lambda <= 0:
            return torch.zeros([], device=self.device)
        _, p = self._gate_params_and_p()
        if p is None:
            return torch.zeros([], device=self.device)
        eps = 1e-6
        ent = - (p * (p + eps).log() + (1 - p) * (1 - p + eps).log()).sum()
        return self._gate_reg_lambda * ent

    def _maybe_equalize_gate_gradients(self):
        if not self._equalize_grad:
            return
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        if hasattr(pl, "gate_logits"):
            p = torch.sigmoid(pl.gate_logits).detach()
            eps = 1e-6
            alpha = 1.0 / (p * (1 - p) + eps)
            alpha = torch.clamp(alpha, 1.0, self._alpha_max)
            if pl.gate_logits.grad is not None:
                pl.gate_logits.grad.mul_(alpha)
        # if conditional, we scale gradients of gate_head uniformly by mean alpha
        elif hasattr(pl, "gate_head"):
            # best-effort: use 0.25 as p midpoint to compute alpha=1/(p*(1-p))≈4
            alpha_mean = max(1.0, min(self._alpha_max, 4.0))
            for p in pl.gate_head.parameters():
                if p.grad is not None:
                    p.grad.mul_(alpha_mean)

    def _get_gate_grad_norm(self):
        """Get gate gradient norm before optimizer.step() clears gradients."""
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        # For static mode
        if hasattr(pl, "gate_logits") and pl.gate_logits.grad is not None:
            grad_norm = pl.gate_logits.grad.detach().pow(2).sum().sqrt().item() / pl.gate_logits.numel()
            return grad_norm
        # For conditional mode: compute average gradient norm across gate_head parameters
        if hasattr(pl, "gate_head"):
            total_norm_sq = 0.0
            total_params = 0
            for param in pl.gate_head.parameters():
                if param.grad is not None:
                    param_norm_sq = param.grad.detach().pow(2).sum().item()
                    total_norm_sq += param_norm_sq
                    total_params += param.numel()
            if total_params > 0:
                grad_norm = (total_norm_sq ** 0.5) / total_params
                return grad_norm
        return 0.0

    def _log_gate_metrics(self, gate_grad_norm=None):
        """Log gate metrics to TensorBoard and return them for printing in logs."""
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        p_detached = None
        
        # Static mode: use gate_logits
        if hasattr(pl, "gate_logits"):
            p_detached = torch.sigmoid(pl.gate_logits).detach()
        # Conditional mode: use saved gate logits from last forward
        elif hasattr(pl, "gate_head") and hasattr(pl, "_last_conditional_gate_logits"):
            gate_logits = pl._last_conditional_gate_logits  # (batch, L) or (batch, L, D)
            if gate_logits is not None:
                if gate_logits.dim() == 2:  # scalar gate: (batch, L)
                    p_detached = torch.sigmoid(gate_logits)  # (batch, L)
                else:  # vector gate: (batch, L, D)
                    p_detached = torch.sigmoid(gate_logits)  # (batch, L, D)
                # Average across batch for summary statistics
                p_detached = p_detached.mean(dim=0)  # (L,) or (L, D)
        
        if p_detached is not None:
            d_eff = p_detached.sum().item()
            mean_p = p_detached.mean().item()
            std_p = p_detached.std().item()
            eps = 1e-6
            ent = - (p_detached * (p_detached + eps).log() + (1 - p_detached) * (1 - p_detached + eps).log()).sum().item()
            
            # Use saved gradient norm if provided, otherwise try to get from grad (may be 0 after step)
            if gate_grad_norm is None:
                if hasattr(pl, "gate_logits") and pl.gate_logits.grad is not None:
                    gate_grad_norm = pl.gate_logits.grad.detach().pow(2).sum().sqrt().item() / pl.gate_logits.numel()
                elif hasattr(pl, "gate_head"):
                    # Compute average gradient norm for gate_head
                    total_norm_sq = 0.0
                    total_params = 0
                    for param in pl.gate_head.parameters():
                        if param.grad is not None:
                            param_norm_sq = param.grad.detach().pow(2).sum().item()
                            total_norm_sq += param_norm_sq
                            total_params += param.numel()
                    if total_params > 0:
                        gate_grad_norm = (total_norm_sq ** 0.5) / total_params
                    else:
                        gate_grad_norm = 0.0
                else:
                    gate_grad_norm = 0.0
            
            step = self.epoch * self.num_batches + self.batch_idx
            self.tb_writer.add_scalar("gate/d_eff", d_eff, step)
            self.tb_writer.add_scalar("gate/mean_p", mean_p, step)
            self.tb_writer.add_scalar("gate/std_p", std_p, step)
            self.tb_writer.add_scalar("gate/entropy", ent, step)
            self.tb_writer.add_scalar("gate/grad_norm", gate_grad_norm, step)
            
            # Return metrics for logging
            return {
                "gate/d_eff": d_eff,
                "gate/mean_p": mean_p,
                "gate/std_p": std_p,
                "gate/entropy": ent,
                "gate/grad_norm": gate_grad_norm,
            }
        return {}
