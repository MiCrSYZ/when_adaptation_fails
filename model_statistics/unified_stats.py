"""
Unified model statistics for MaPLe, BiDirMaPLe, AdaptiveMaPLe, AdaptiveBiDirMaPLe.
Run from repo root, e.g.:
python -m model_statistics.unified_stats --trainer AdaptiveBiDirMaPLe --dataset imagenet
python -m model_statistics.unified_stats --trainer MaPLe --dataset caltech101
python -m model_statistics.unified_stats --trainer BiDirMaPLe --dataset food101
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import json
import os

from dassl.config import get_cfg_default
from clip import clip
from clip.model import build_model, build_model_adaptive

import trainers.maple
import trainers.bidir_maple
import trainers.adaptive_maple
import trainers.adaptive_bidir_maple


def extend_cfg(cfg):
    """Extend default cfg for all supported trainers."""
    from yacs.config import CfgNode as CN
    
    # MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    # BiDirMaPLe
    cfg.TRAINER.BIDIR_MAPLE = CN()
    cfg.TRAINER.BIDIR_MAPLE.N_CTX = 2
    cfg.TRAINER.BIDIR_MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIDIR_MAPLE.PREC = "fp16"
    cfg.TRAINER.BIDIR_MAPLE.PROMPT_DEPTH = 9
    cfg.TRAINER.BIDIR_MAPLE.BIDIRECTIONAL = True  # Enable/disable bidirectional coupling
    cfg.TRAINER.BIDIR_MAPLE.CYCLE_LOSS_WEIGHT = 0.1  # Weight for cycle consistency loss λ
    cfg.TRAINER.BIDIR_MAPLE.CYCLE_WARMUP_EPOCHS = 2
    cfg.TRAINER.BIDIR_MAPLE.USE_EMA = False
    cfg.TRAINER.BIDIR_MAPLE.EMA_DECAY = 0.999
    cfg.TRAINER.BIDIR_MAPLE.EMA_START_EPOCH = 5
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    # AdaptiveMaPLe
    cfg.TRAINER.ADAPTIVE_MAPLE = CN()
    cfg.TRAINER.ADAPTIVE_MAPLE.N_CTX_MAX = 8
    cfg.TRAINER.ADAPTIVE_MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.ADAPTIVE_MAPLE.PREC = "fp16"
    cfg.TRAINER.ADAPTIVE_MAPLE.PROMPT_DEPTH_MAX = 12
    cfg.TRAINER.ADAPTIVE_MAPLE.LAMBDA_SPARSITY = 0.01
    cfg.TRAINER.ADAPTIVE_MAPLE.LAMBDA_DEPTH_SMOOTH = 0.001
    cfg.TRAINER.ADAPTIVE_MAPLE.TYPE = "full"
    cfg.TRAINER.ADAPTIVE_MAPLE.FIXED_N_CTX = 4
    cfg.TRAINER.ADAPTIVE_MAPLE.FIXED_PROMPT_DEPTH = 6
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    # AdaptiveBiDirMaPLe
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE = CN()
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.N_CTX_MAX = 8
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PREC = "fp16"
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PROMPT_DEPTH_MAX = 12
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_SPARSITY = 0.01
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_DEPTH_SMOOTH = 0.001
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.TYPE = "full"  # full / length_only / depth_only
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.FIXED_N_CTX = 4
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.FIXED_PROMPT_DEPTH = 6
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.BIDIRECTIONAL = True
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.CYCLE_LOSS_WEIGHT = 0.1
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.MAPPING_HIDDEN_DIM_RATIO = 0.5
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def load_clip_model(cfg, trainer_name):
    """Load CLIP backbone."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    if trainer_name == "MaPLe":
        design_details = {
            "trainer": 'MaPLe',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.TRAINER.MAPLE.N_CTX
        }
        model = build_model(state_dict or model.state_dict(), design_details)
    
    elif trainer_name == "BiDirMaPLe":
        design_details = {
            "trainer": 'BiDirMaPLe',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.TRAINER.BIDIR_MAPLE.N_CTX
        }
        model = build_model(state_dict or model.state_dict(), design_details)
    
    elif trainer_name == "AdaptiveMaPLe":
        design_details = {
            "trainer": 'AdaptiveMaPLe',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.TRAINER.ADAPTIVE_MAPLE.N_CTX_MAX,
            "maple_depth_max": cfg.TRAINER.ADAPTIVE_MAPLE.PROMPT_DEPTH_MAX
        }
        model = build_model_adaptive(state_dict or model.state_dict(), design_details)
    
    elif trainer_name == "AdaptiveBiDirMaPLe":
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
    
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}")
    
    return model


def create_custom_model(cfg, clip_model, trainer_name, num_classes=100):
    """Build CustomCLIP for the given trainer."""
    dummy_classnames = [f"class_{i}" for i in range(num_classes)]
    
    if trainer_name == "MaPLe":
        from trainers.maple import CustomCLIP
        model = CustomCLIP(cfg, dummy_classnames, clip_model)
    
    elif trainer_name == "BiDirMaPLe":
        from trainers.bidir_maple import CustomCLIP
        model = CustomCLIP(cfg, dummy_classnames, clip_model)
    
    elif trainer_name == "AdaptiveMaPLe":
        from trainers.adaptive_maple import CustomCLIP
        model = CustomCLIP(cfg, dummy_classnames, clip_model)
    
    elif trainer_name == "AdaptiveBiDirMaPLe":
        from trainers.adaptive_bidir_maple import CustomCLIP
        model = CustomCLIP(cfg, dummy_classnames, clip_model)
    
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}")
    
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    return model


def count_parameters_detailed(model):
    """Count parameters by module and prompt components."""
    results = OrderedDict()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    results['frozen_params'] = frozen_params
    results['trainable_ratio'] = f"{trainable_params/total_params*100:.2f}%"
    
    module_stats = OrderedDict()
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_stats[name] = {
            'total': module_total,
            'trainable': module_trainable,
            'frozen': module_total - module_trainable
        }
    
    results['module_params'] = module_stats
    
    if hasattr(model, 'prompt_learner'):
        prompt_stats = count_prompt_parameters(model.prompt_learner)
        results['prompt_params'] = prompt_stats
    
    return results


def count_prompt_parameters(prompt_learner):
    """Break down prompt-related parameter counts."""
    prompt_stats = OrderedDict()
    
    if hasattr(prompt_learner, 'ctx'):
        prompt_stats['shallow_ctx'] = prompt_learner.ctx.numel()
    
    if hasattr(prompt_learner, 'compound_prompts_text'):
        text_prompts = sum(p.numel() for p in prompt_learner.compound_prompts_text)
        prompt_stats['deep_text_prompts'] = text_prompts
    
    if hasattr(prompt_learner, 'compound_prompt_projections'):
        proj_params = sum(p.numel() for p in prompt_learner.compound_prompt_projections.parameters())
        prompt_stats['visual_proj'] = proj_params
    
    if hasattr(prompt_learner, 'length_gates'):
        if isinstance(prompt_learner.length_gates, nn.Parameter):
            prompt_stats['length_gates'] = prompt_learner.length_gates.numel()
    
    if hasattr(prompt_learner, 'depth_weights'):
        if isinstance(prompt_learner.depth_weights, nn.Parameter):
            prompt_stats['depth_weights'] = prompt_learner.depth_weights.numel()
    
    if hasattr(prompt_learner, 'l2v_mappings'):
        l2v_params = sum(p.numel() for p in prompt_learner.l2v_mappings.parameters())
        prompt_stats['lang_to_vis'] = l2v_params
    
    if hasattr(prompt_learner, 'v2l_mappings'):
        v2l_params = sum(p.numel() for p in prompt_learner.v2l_mappings.parameters())
        prompt_stats['vis_to_lang'] = v2l_params
    
    if hasattr(prompt_learner, 'alpha_params'):
        prompt_stats['alpha'] = sum(p.numel() for p in prompt_learner.alpha_params)
    
    if hasattr(prompt_learner, 'beta_params'):
        prompt_stats['beta'] = sum(p.numel() for p in prompt_learner.beta_params)
    
    total_prompt = sum(v for k, v in prompt_stats.items())
    prompt_stats['prompt_total'] = total_prompt
    
    return prompt_stats


def format_number(num):
    """Human-readable large numbers."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_statistics(results, trainer_name):
    """Pretty-print statistics."""
    print("\n" + "="*100)
    print(f"Model statistics — {trainer_name}")
    print("="*100)
    
    print(f"\n{'Total params':<20}: {format_number(results['total_params']):>12} ({results['total_params']:>15,})")
    print(f"{'Trainable':<20}: {format_number(results['trainable_params']):>12} ({results['trainable_params']:>15,})")
    print(f"{'Frozen':<20}: {format_number(results['frozen_params']):>12} ({results['frozen_params']:>15,})")
    print(f"{'Trainable ratio':<20}: {results['trainable_ratio']:>12}")
    
    if 'module_params' in results:
        print("\n" + "-"*100)
        print("Parameters by module:")
        print("-"*100)
        print(f"{'Module':<30} | {'Total':>15} | {'Trainable':>15} | {'Frozen':>15}")
        print("-"*100)
        
        for name, params in results['module_params'].items():
            print(f"{name:<30} | "
                  f"{format_number(params['total']):>15} | "
                  f"{format_number(params['trainable']):>15} | "
                  f"{format_number(params['frozen']):>15}")
    
    if 'prompt_params' in results:
        print("\n" + "-"*100)
        print("Prompt parameter breakdown:")
        print("-"*100)
        for name, count in results['prompt_params'].items():
            print(f"{name:<30}: {format_number(count):>12} ({count:>12,})")
    
    print("="*100 + "\n")


def compute_flops_thop(model, input_size=(1, 3, 224, 224), num_classes=100):
    """FLOPs via thop."""
    try:
        from thop import profile, clever_format
        
        dummy_input = torch.randn(input_size)
        dummy_label = torch.randint(0, num_classes, (input_size[0],))
        
        model.eval()
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception:
                pass
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        return {
            'flops': flops,
            'flops_str': flops_str,
            'params': params,
            'params_str': params_str
        }
    except ImportError:
        print("thop not installed. pip install thop")
        return None
    except Exception as e:
        print(f"THOP failed: {e}")
        return None


def compute_flops_fvcore(model, input_size=(1, 3, 224, 224)):
    """FLOPs via fvcore."""
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        
        dummy_input = torch.randn(input_size)
        
        model.eval()
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception:
                pass
        
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        
        if total_flops > 1e12:
            flops_str = f"{total_flops / 1e12:.3f} T"
        elif total_flops > 1e9:
            flops_str = f"{total_flops / 1e9:.3f} G"
        elif total_flops > 1e6:
            flops_str = f"{total_flops / 1e6:.3f} M"
        elif total_flops > 1e3:
            flops_str = f"{total_flops / 1e3:.3f} K"
        else:
            flops_str = f"{total_flops}"
        
        return {
            'flops': total_flops,
            'flops_str': flops_str
        }
    except ImportError:
        print("fvcore not installed. pip install fvcore")
        return None
    except Exception as e:
        print(f"FVCore failed: {e}")
        return None


def estimate_flops_manual(model, input_size=(1, 3, 224, 224)):
    """Rough FLOP estimate without extra deps."""
    total_flops = 0
    
    def count_conv2d(m):
        # FLOPs = 2 * Cin * Cout * K^2 * H * W
        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        
        h_out = input_size[2] // (m.stride[0] if isinstance(m.stride, tuple) else m.stride)
        w_out = input_size[3] // (m.stride[1] if isinstance(m.stride, tuple) else m.stride)
        
        flops = 2 * cin * cout * kh * kw * h_out * w_out
        return flops
    
    def count_linear(m):
        flops = 2 * m.in_features * m.out_features
        return flops
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total_flops += count_conv2d(module)
        elif isinstance(module, nn.Linear):
            total_flops += count_linear(module)
    
    if total_flops > 1e12:
        flops_str = f"{total_flops / 1e12:.3f} T"
    elif total_flops > 1e9:
        flops_str = f"{total_flops / 1e9:.3f} G"
    elif total_flops > 1e6:
        flops_str = f"{total_flops / 1e6:.3f} M"
    else:
        flops_str = f"{total_flops / 1e3:.3f} K"
    
    return {
        'flops': total_flops,
        'flops_str': flops_str
    }


def estimate_memory_usage(model, input_size=(1, 3, 224, 224)):
    """Rough GPU memory estimate (MB)."""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    grad_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    dummy_input = torch.randn(input_size)
    activation_memory = dummy_input.numel() * dummy_input.element_size() * 10
    
    total_memory = param_memory + grad_memory + activation_memory
    
    return {
        'params_mb': param_memory / (1024**2),
        'grads_mb': grad_memory / (1024**2),
        'activations_est_mb': activation_memory / (1024**2),
        'total_est_mb': total_memory / (1024**2)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, required=True,
                       choices=["MaPLe", "BiDirMaPLe", "AdaptiveMaPLe", "AdaptiveBiDirMaPLe"],
                       help="Trainer name")
    parser.add_argument("--dataset", type=str, default="imagenet",
                       help="Dataset name (for logging)")
    parser.add_argument("--backbone", type=str, default="ViT-B/16",
                       help="CLIP backbone")
    parser.add_argument("--num-classes", type=int, default=100,
                       help="Number of classes")
    parser.add_argument("--output-dir", type=str, default="./statistics",
                       help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.TRAINER.NAME = args.trainer
    cfg.freeze()
    
    print(f"\nLoading {args.trainer}...")
    print(f"Backbone: {args.backbone}")
    print(f"Dataset: {args.dataset}")
    
    clip_model = load_clip_model(cfg, args.trainer)
    model = create_custom_model(cfg, clip_model, args.trainer, args.num_classes)
    
    print("\nCounting parameters...")
    results = count_parameters_detailed(model)
    print_statistics(results, args.trainer)
    
    print("Estimated memory (MB):")
    print("-"*100)
    memory_stats = estimate_memory_usage(model)
    for key, value in memory_stats.items():
        print(f"{key:<30}: {value:>10.2f} MB")
    print("-"*100 + "\n")
    
    output_file = os.path.join(args.output_dir, f"{args.trainer}_statistics.json")
    save_results = {
        'trainer': args.trainer,
        'backbone': args.backbone,
        'dataset': args.dataset,
        'total_params': results['total_params'],
        'trainable_params': results['trainable_params'],
        'frozen_params': results['frozen_params'],
        'trainable_ratio': results['trainable_ratio'],
        'memory_usage_mb': memory_stats,
        'prompt_params': results.get('prompt_params', {})
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"Saved statistics to: {output_file}\n")


if __name__ == "__main__":
    main()
