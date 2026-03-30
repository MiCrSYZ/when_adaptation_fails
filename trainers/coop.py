import os.path as osp

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
    design_details = {"trainer": 'CoOp',
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
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
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
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # Gating configs
        self.use_gate = bool(getattr(cfg.TRAINER.COOP, "USE_GATE", False))
        self.gate_type = getattr(cfg.TRAINER.COOP, "GATE_TYPE", "scalar")  # scalar|vector
        self.gate_init_std = float(getattr(cfg.TRAINER.COOP, "GATE_INIT_STD", 0.02))
        self.param_matched = bool(getattr(cfg.TRAINER.COOP, "PARAM_MATCHED", False))

        gate_param_list = []
        bias_param_list = []
        if self.use_gate:
            if self.gate_type == "scalar":
                # g in R^{L}
                gate_logits = torch.zeros(n_ctx, dtype=dtype)
                if self.gate_init_std > 0:
                    nn.init.normal_(gate_logits, mean=0.0, std=self.gate_init_std)
                self.gate_logits = nn.Parameter(gate_logits)
            elif self.gate_type == "vector":
                # g in R^{L x D}
                gate_logits = torch.zeros(n_ctx, ctx_dim, dtype=dtype)
                if self.gate_init_std > 0:
                    nn.init.normal_(gate_logits, mean=0.0, std=self.gate_init_std)
                self.gate_logits = nn.Parameter(gate_logits)
            else:
                raise ValueError("Unsupported GATE_TYPE for CoOp: {}".format(self.gate_type))
            gate_param_list.append("gate_logits")

        # Parameter-matched baseline: add per-token bias with parameter count ≈ gating variant
        if self.param_matched and not self.use_gate:
            if self.gate_type == "scalar":
                # per-token scalar bias, broadcast along dim
                bias = torch.zeros(n_ctx, 1, dtype=dtype)
                self.pm_bias = nn.Parameter(bias)
            else:
                bias = torch.zeros(n_ctx, ctx_dim, dtype=dtype)
                self.pm_bias = nn.Parameter(bias)
            bias_param_list.append("pm_bias")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Apply gating or parameter-matched baseline on context tokens before composing prompts
        if hasattr(self, "gate_logits") and self.use_gate:
            # p = sigmoid(g) \odot C
            if self.gate_type == "scalar":
                p = torch.sigmoid(self.gate_logits).unsqueeze(-1)  # [L,1]
            else:
                p = torch.sigmoid(self.gate_logits)  # [L,D]
            gated_ctx = p * self.ctx  # [L,D]
            ctx = gated_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            # cache for logging
            self._last_gate_p = p.detach()
        elif hasattr(self, "pm_bias") and self.param_matched and not self.use_gate:
            # ctx' = ctx + bias (bias broadcast if scalar)
            biased_ctx = self.ctx + (self.pm_bias if self.pm_bias.shape[-1] == ctx.shape[-1] else self.pm_bias.repeat(1, ctx.shape[-1]))
            ctx = biased_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

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

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # Build optimizer with optional different lr for gate params
        base_lr = float(getattr(cfg.OPTIM, "LR", 1e-3))
        gate_lr_mul = float(getattr(cfg.TRAINER.COOP, "GATE_LR_MUL", 1.0))
        param_groups = []
        gate_params = []
        other_params = []
        for n, p in self.model.prompt_learner.named_parameters():
            if not p.requires_grad:
                continue
            if "gate_logits" in n:
                gate_params.append(p)
            else:
                other_params.append(p)
        if gate_params:
            param_groups.append({"params": gate_params, "lr": base_lr * gate_lr_mul})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})
        optim_target = param_groups if param_groups else self.model.prompt_learner
        self.optim = build_optimizer(optim_target, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
        self._gate_reg_lambda = float(getattr(cfg.TRAINER.COOP, "GATE_REG_LAMBDA", 0.0))
        self._equalize_grad = bool(getattr(cfg.TRAINER.COOP, "EQUALIZE_GRAD", False))
        self._alpha_max = float(getattr(cfg.TRAINER.COOP, "ALPHA_MAX", 10.0))

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                # optional gate regularization (entropy on p)
                loss = loss + self._compute_gate_reg()
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            # unscale for safe grad ops
            self.scaler.unscale_(self.optim)
            self._maybe_equalize_gate_gradients()
            # Save gradient before step (gradients will be cleared/modified by step)
            gate_grad_norm = self._get_gate_grad_norm()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            loss = loss + self._compute_gate_reg()
            self.optim.zero_grad()
            loss.backward()
            self._maybe_equalize_gate_gradients()
            # Save gradient before step (gradients will be cleared/modified by step)
            gate_grad_norm = self._get_gate_grad_norm()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

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
    def _current_gate_p(self):
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        if hasattr(pl, "gate_logits"):
            g = pl.gate_logits
            p = torch.sigmoid(g)
            return p
        return None

    def _compute_gate_reg(self) -> torch.Tensor:
        if self._gate_reg_lambda <= 0:
            return torch.zeros([], device=self.device)
        p = self._current_gate_p()
        if p is None:
            return torch.zeros([], device=self.device)
        # Entropy regularization: H(p) = -sum(p log p + (1-p) log(1-p))
        eps = 1e-6
        ent = - (p * (p + eps).log() + (1 - p) * (1 - p + eps).log()).sum()
        return self._gate_reg_lambda * ent

    def _maybe_equalize_gate_gradients(self):
        if not self._equalize_grad:
            return
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        if not hasattr(pl, "gate_logits"):
            return
        p = torch.sigmoid(pl.gate_logits).detach()
        eps = 1e-6
        alpha = 1.0 / (p * (1 - p) + eps)
        alpha = torch.clamp(alpha, 1.0, self._alpha_max)
        if pl.gate_logits.grad is not None:
            pl.gate_logits.grad.mul_(alpha)

    def _get_gate_grad_norm(self):
        """Get gate gradient norm before optimizer.step() clears gradients."""
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        if not hasattr(pl, "gate_logits") or pl.gate_logits.grad is None:
            return 0.0
        grad_norm = pl.gate_logits.grad.detach().pow(2).sum().sqrt().item() / pl.gate_logits.numel()
        return grad_norm

    def _log_gate_metrics(self, gate_grad_norm=None):
        """Log gate metrics to TensorBoard and return them for printing in logs."""
        pl = self.model.module.prompt_learner if isinstance(self.model, nn.DataParallel) else self.model.prompt_learner
        if not hasattr(pl, "gate_logits"):
            return {}
        
        p = torch.sigmoid(pl.gate_logits).detach()
        d_eff = p.sum().item()
        mean_p = p.mean().item()
        std_p = p.std().item()
        eps = 1e-6
        ent = - (p * (p + eps).log() + (1 - p) * (1 - p + eps).log()).sum().item()
        
        # Use saved gradient norm if provided, otherwise try to get from grad (may be 0 after step)
        if gate_grad_norm is None:
            if pl.gate_logits.grad is not None:
                gate_grad_norm = pl.gate_logits.grad.detach().pow(2).sum().sqrt().item() / pl.gate_logits.numel()
            else:
                gate_grad_norm = 0.0
        
        global_step = self.epoch * self.num_batches + self.batch_idx
        self.tb_writer.add_scalar("gate/d_eff", d_eff, global_step)
        self.tb_writer.add_scalar("gate/mean_p", mean_p, global_step)
        self.tb_writer.add_scalar("gate/std_p", std_p, global_step)
        self.tb_writer.add_scalar("gate/entropy", ent, global_step)
        self.tb_writer.add_scalar("gate/grad_norm", gate_grad_norm, global_step)
        
        # Return metrics for logging
        return {
            "gate/d_eff": d_eff,
            "gate/mean_p": mean_p,
            "gate/std_p": std_p,
            "gate/entropy": ent,
            "gate/grad_norm": gate_grad_norm,
        }
