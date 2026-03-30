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
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

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
    design_details = {"trainer": 'BiDirMaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.BIDIR_MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class CrossModalMappingNetwork(nn.Module):
    """Network for cross-modal mapping with two-layer MLP + ReLU"""

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

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        # MaPLe: full 77-token prompts; deep prompts swap inside the transformer
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Combined input to transformer
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIDIR_MAPLE.N_CTX
        ctx_init = cfg.TRAINER.BIDIR_MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg.TRAINER.BIDIR_MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.BIDIR_MAPLE.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.enable_bidirectional = getattr(cfg.TRAINER.BIDIR_MAPLE, 'BIDIRECTIONAL', True)
        self.cycle_loss_weight = getattr(cfg.TRAINER.BIDIR_MAPLE, 'CYCLE_LOSS_WEIGHT', 0.1)

        # Init context vectors
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning with Bidirectional Coupling')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        print(f"Bidirectional coupling enabled: {self.enable_bidirectional}")

        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

        # Deeper prompts
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, ctx_dim))  # use ctx_dim, not hard-coded 512
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        if self.enable_bidirectional:
            # Dimension: ctx_dim
            self.l2v_mappings = nn.ModuleList([
                CrossModalMappingNetwork(ctx_dim, ctx_dim // 2, 768)
                for _ in range(self.compound_prompts_depth - 1)
            ])

            self.v2l_mappings = nn.ModuleList([
                CrossModalMappingNetwork(768, 384, ctx_dim)
                for _ in range(self.compound_prompts_depth - 1)
            ])

            self.alpha_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5))
                for _ in range(self.compound_prompts_depth - 1)
            ])
            self.beta_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5))
                for _ in range(self.compound_prompts_depth - 1)
            ])

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def compute_cycle_consistency_loss(self, text_prompts, vision_prompts):
        if not self.enable_bidirectional or len(text_prompts) == 0:
            return torch.tensor(0.0).to(text_prompts[0].device if text_prompts else vision_prompts[0].device)

        cycle_loss = 0.0
        for i in range(len(text_prompts)):
            l2v = self.l2v_mappings[i](text_prompts[i])
            v2l = self.v2l_mappings[i](l2v)
            cycle_loss += F.mse_loss(text_prompts[i], v2l)

        return cycle_loss / len(text_prompts)

    def apply_bidirectional_coupling(self, text_prompts, vision_prompts):
        if not self.enable_bidirectional:
            return text_prompts, vision_prompts

        coupled_text_prompts = []
        coupled_vision_prompts = []

        for i in range(len(text_prompts)):
            l2v_mapped = self.l2v_mappings[i](text_prompts[i])
            v2l_mapped = self.v2l_mappings[i](vision_prompts[i])

            alpha = torch.sigmoid(self.alpha_params[i])
            beta = torch.sigmoid(self.beta_params[i])

            coupled_text = alpha * text_prompts[i] + (1 - alpha) * v2l_mapped
            l2v_updated = self.l2v_mappings[i](coupled_text)
            coupled_vision = beta * vision_prompts[i] + (1 - beta) * l2v_updated

            coupled_text_prompts.append(coupled_text)
            coupled_vision_prompts.append(coupled_vision)

        return coupled_text_prompts, coupled_vision_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # Build initial prompts (length 77)
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Vision deep prompts
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        # Bidirectional coupling
        if self.enable_bidirectional:
            coupled_text_prompts, coupled_vision_prompts = self.apply_bidirectional_coupling(
                self.compound_prompts_text, visual_deep_prompts
            )
            return prompts, self.proj(self.ctx), coupled_text_prompts, coupled_vision_prompts
        else:
            return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts

    def get_cycle_loss(self):
        if not self.enable_bidirectional:
            return torch.tensor(0.0)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        return self.compute_cycle_consistency_loss(self.compound_prompts_text, visual_deep_prompts)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Store cycle loss weight
        self.cycle_loss_weight = getattr(cfg.TRAINER.BIDIR_MAPLE, 'CYCLE_LOSS_WEIGHT', 0.1)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        eps = 1e-8
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, label)

            # Cycle consistency loss
            cycle_loss = self.prompt_learner.get_cycle_loss()

            # Total loss
            #total_loss = ce_loss + self.cycle_loss_weight * cycle_loss

            return ce_loss, cycle_loss

        return logits

    def evaluate(self, image):
        with torch.no_grad():
            return self.forward(image)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CycleLossScheduler:
    """Cycle loss weight schedule."""
    def __init__(self, base_weight, warmup_epochs=2, schedule_type='linear'):
        self.base_weight = base_weight
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type

    def get_weight(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            if self.schedule_type == 'linear':
                # Linear warmup to base_weight
                return self.base_weight * (current_epoch / self.warmup_epochs)
            elif self.schedule_type == 'cosine':
                # Cosine warmup
                import math
                progress = current_epoch / self.warmup_epochs
                return self.base_weight * (1 - math.cos(progress * math.pi)) / 2
        return self.base_weight


class ModelEMA:
    """EMA shadow weights."""
    def __init__(self, model, decay=0.999, start_epoch=None):
        self.model = model
        self.decay = decay
        self.start_epoch = start_epoch
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register params to average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, current_epoch=None):
        """EMA update step."""
        if self.start_epoch is not None and current_epoch < self.start_epoch:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + \
                              self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Swap in EMA weights for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore live weights after EMA eval."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


@TRAINER_REGISTRY.register()
class BiDirMaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIDIR_MAPLE.PREC in ["fp16", "fp32", "amp"]

        cfg.defrost()
        cfg.TRAINER.BIDIR_MAPLE.VISUALIZE = getattr(cfg.TRAINER.BIDIR_MAPLE, "VISUALIZE", True)
        cfg.TRAINER.BIDIR_MAPLE.VAL_FREQ = getattr(cfg.TRAINER.BIDIR_MAPLE, "VAL_FREQ", 1)
        cfg.freeze()

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BIDIR_MAPLE.PREC == "fp32" or cfg.TRAINER.BIDIR_MAPLE.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP with bidirectional coupling")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_ce_loss': [],
            'train_cycle_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        # Output dirs
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = osp.join(cfg.OUTPUT_DIR, f"logs_{current_time}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Cycle loss scheduler
        base_cycle_weight = cfg.TRAINER.BIDIR_MAPLE.CYCLE_LOSS_WEIGHT
        warmup_epochs = cfg.TRAINER.BIDIR_MAPLE.CYCLE_WARMUP_EPOCHS
        self.cycle_scheduler = CycleLossScheduler(
            base_weight=base_cycle_weight,
            warmup_epochs=warmup_epochs,
            schedule_type='linear'
        )

        if cfg.TRAINER.BIDIR_MAPLE.USE_EMA:
            self.model_ema = ModelEMA(
                self.model,
                decay=cfg.TRAINER.BIDIR_MAPLE.EMA_DECAY,
                start_epoch=cfg.TRAINER.BIDIR_MAPLE.EMA_START_EPOCH
            )
            print(f"EMA enabled: decay={self.model_ema.decay}, "
                  f"start_epoch={self.model_ema.start_epoch}")
        else:
            self.model_ema = None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        current_cycle_weight = self.cycle_scheduler.get_weight(self.epoch)

        prec = self.cfg.TRAINER.BIDIR_MAPLE.PREC
        if prec == "amp":
            with autocast():
                ce_loss, cycle_loss = model(image, label)
            total_loss = ce_loss + current_cycle_weight * cycle_loss
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optim)
            scaler.update()
        else:
            ce_loss, cycle_loss = model(image, label)
            total_loss = ce_loss + current_cycle_weight * cycle_loss
            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optim.step()

        if self.model_ema is not None:
            self.model_ema.update(current_epoch=self.epoch)

        loss_summary = {
            "loss": float(total_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "cycle_loss": float(cycle_loss.item())
        }

        if hasattr(self, 'current_epoch_metrics'):
            self.current_epoch_metrics['train_loss'].append(loss_summary["loss"])
            self.current_epoch_metrics['train_ce_loss'].append(loss_summary["ce_loss"])
            self.current_epoch_metrics['train_cycle_loss'].append(loss_summary["cycle_loss"])

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def test(self, split=None, **kwargs):
        """Use EMA weights for testing."""
        if self.model_ema is not None and split in ['val', 'test']:
            # EMA forward
            self.model_ema.apply_shadow()
            try:
                result = super().test(split=split, **kwargs)
            finally:
                self.model_ema.restore()
            return result
        else:
            return super().test(split=split, **kwargs)

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def before_epoch(self):
        """Reset epoch metric buffers."""
        super().before_epoch()
        self.current_epoch_metrics = {
            'train_loss': [],
            'train_ce_loss': [],
            'train_cycle_loss': []
        }

    def after_step(self):
        """Accumulate batch metrics."""
        if hasattr(self, 'outputs') and self.outputs and hasattr(self, 'current_epoch_metrics'):
            if 'train_loss' in self.current_epoch_metrics and len(
                    self.current_epoch_metrics['train_loss']) <= self.batch_idx:
                # Fallback if not accumulated in forward_backward
                self.current_epoch_metrics['train_loss'].append(self.outputs["loss"])
                self.current_epoch_metrics['train_ce_loss'].append(self.outputs["ce_loss"])
                self.current_epoch_metrics['train_cycle_loss'].append(self.outputs["cycle_loss"])

    def after_epoch(self):
        """End of epoch: log, validate, plot."""
        # Mean train loss
        avg_train_loss = np.mean(self.current_epoch_metrics['train_loss']) if self.current_epoch_metrics[
            'train_loss'] else 0.0
        avg_ce_loss = np.mean(self.current_epoch_metrics['train_ce_loss']) if self.current_epoch_metrics[
            'train_ce_loss'] else 0.0
        avg_cycle_loss = np.mean(self.current_epoch_metrics['train_cycle_loss']) if self.current_epoch_metrics[
            'train_cycle_loss'] else 0.0

        # Validation
        if (self.epoch + 1) % self.cfg.TRAINER.BIDIR_MAPLE.VAL_FREQ == 0 or self.epoch == self.max_epoch - 1:
            val_loss, val_acc = self.validate()
        else:
            val_loss, val_acc = float('nan'), float('nan')

        # Log metrics
        self.train_history['epoch'].append(self.epoch + 1)
        self.train_history['train_loss'].append(avg_train_loss)
        self.train_history['train_ce_loss'].append(avg_ce_loss)
        self.train_history['train_cycle_loss'].append(avg_cycle_loss)
        self.train_history['val_loss'].append(val_loss)
        self.train_history['val_acc'].append(val_acc)

        current_weight = self.cycle_scheduler.get_weight(self.epoch)
        if self.epoch < self.cycle_scheduler.warmup_epochs:
            print(f"[Warmup] Cycle loss weight: {current_weight:.4f}")

        if self.model_ema is not None and \
                self.epoch >= self.model_ema.start_epoch:
            print(f"[EMA Active] decay={self.model_ema.decay}")

        # Print epoch summary
        print(f"\nEpoch [{self.epoch + 1}/{self.max_epoch}] Metrics:")
        print(f"Train Loss: {avg_train_loss:.4f} | CE Loss: {avg_ce_loss:.4f} | Cycle Loss: {avg_cycle_loss:.4f}")
        if not math.isnan(val_loss):
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        # Periodic plots
        if self.cfg.TRAINER.BIDIR_MAPLE.VISUALIZE and (self.epoch + 1) % 5 == 0:
            self.visualize_metrics()

        # Parent after_epoch (checkpoints, etc.)
        super().after_epoch()

    def after_train(self):
        super().after_train()
        # Final plot
        if self.cfg.TRAINER.BIDIR_MAPLE.VISUALIZE:
            self.visualize_metrics(save=True)
            print(f"Training metrics saved to {self.output_dir}")

    def validate(self):
        """Evaluate on val set."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dm.val_loader, desc="Validating")):
                images, labels = self.parse_batch_train(batch)

                # Forward
                logits = self.model.evaluate(images)

                # Loss / accuracy
                loss = F.cross_entropy(logits, labels)
                acc_percent = compute_accuracy(logits, labels)[0].item()
                acc_decimal = acc_percent / 100.0
                total_loss += loss.item() * labels.size(0)
                total_acc += acc_decimal * labels.size(0)
                total_count += labels.size(0)

        avg_loss = total_loss / total_count
        avg_acc = total_acc / total_count

        self.model.train()
        return avg_loss, avg_acc

    def visualize_metrics(self, save=False):
        """Plot loss and accuracy curves."""
        plt.figure(figsize=(15, 10))

        # Loss plot
        plt.subplot(2, 1, 1)
        epochs = self.train_history['epoch']
        plt.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, self.train_history['train_ce_loss'], 'g--', label='Train CE Loss')
        plt.plot(epochs, self.train_history['train_cycle_loss'], 'c-.', label='Train Cycle Loss')

        # Drop NaNs
        val_epochs = [e for e, v in zip(epochs, self.train_history['val_loss']) if not math.isnan(v)]
        val_losses = [v for v in self.train_history['val_loss'] if not math.isnan(v)]
        if val_losses:
            plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(2, 1, 2)
        val_acc_epochs = [e for e, a in zip(epochs, self.train_history['val_acc']) if not math.isnan(a)]
        val_accs = [a for a in self.train_history['val_acc'] if not math.isnan(a)]
        if val_accs:
            plt.plot(val_acc_epochs, val_accs, 'm-', label='Validation Accuracy')
            plt.title('Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if save:
            # Save figure
            fig_path = osp.join(self.output_dir, "training_metrics.png")
            plt.savefig(fig_path, dpi=300)
            print(f"Metrics visualization saved to {fig_path}")

            # Save metrics npz
            data_path = osp.join(self.output_dir, "metrics_data.npz")
            np.savez(
                data_path,
                epochs=np.array(self.train_history['epoch']),
                train_loss=np.array(self.train_history['train_loss']),
                train_ce_loss=np.array(self.train_history['train_ce_loss']),
                train_cycle_loss=np.array(self.train_history['train_cycle_loss']),
                val_loss=np.array(self.train_history['val_loss']),
                val_acc=np.array(self.train_history['val_acc'])
            )
        else:
            plt.show()

        plt.close()

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

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)