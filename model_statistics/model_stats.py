"""
Parameter and FLOP counting for a single AdaptiveBiDirMaPLe-style model.
Run from repo root, e.g.:
python -m model_statistics.model_stats --config-file <trainer.yaml> --dataset-config-file configs/datasets/imagenet.yaml
"""

import argparse
import torch
import torch.nn as nn
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count

from dassl.config import get_cfg_default
from clip import clip
from clip.model import build_model_adaptive
import trainers.adaptive_bidir_maple


def extend_cfg(cfg):
    """Match train.py defaults for AdaptiveBiDirMaPLe."""
    from yacs.config import CfgNode as CN

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


def load_clip_model(cfg):
    """Load CLIP backbone."""
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
        "maple_length": cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.N_CTX_MAX,
        "maple_depth_max": cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PROMPT_DEPTH_MAX
    }

    model = build_model_adaptive(state_dict or model.state_dict(), design_details)
    return model


def count_parameters(model, trainable_only=False):
    """Return total parameter count."""
    if trainable_only:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_type = "trainable"
    else:
        params = sum(p.numel() for p in model.parameters())
        param_type = "all"
    return params, param_type


def count_parameters_detailed(model):
    """Per-module breakdown."""
    results = {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    results['frozen_params'] = frozen_params

    module_params = {}
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = {
            'total': module_total,
            'trainable': module_trainable
        }

    results['module_params'] = module_params

    return results


def compute_flops_thop(model, input_size=(1, 3, 224, 224), num_classes=100):
    """FLOPs via thop."""
    try:
        dummy_input = torch.randn(input_size)
        dummy_label = torch.randint(0, num_classes, (input_size[0],))

        try:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        except Exception:
            flops, params = profile(model, inputs=(dummy_input, dummy_label), verbose=False)

        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except Exception as e:
        print(f"THOP failed: {e}")
        return None, None


def compute_flops_fvcore(model, input_size=(1, 3, 224, 224)):
    """FLOPs via fvcore."""
    try:
        dummy_input = torch.randn(input_size)
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()

        if total_flops > 1e9:
            flops_str = f"{total_flops / 1e9:.3f} G"
        elif total_flops > 1e6:
            flops_str = f"{total_flops / 1e6:.3f} M"
        elif total_flops > 1e3:
            flops_str = f"{total_flops / 1e3:.3f} K"
        else:
            flops_str = f"{total_flops}"

        return flops_str, total_flops
    except Exception as e:
        print(f"FVCore failed: {e}")
        return None, None


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


def print_statistics(results):
    """Print breakdown."""
    print("\n" + "="*80)
    print("Model statistics")
    print("="*80)

    print(f"\nTotal params:   {format_number(results['total_params'])} ({results['total_params']:,})")
    print(f"Trainable:      {format_number(results['trainable_params'])} ({results['trainable_params']:,})")
    print(f"Frozen:         {format_number(results['frozen_params'])} ({results['frozen_params']:,})")
    print(f"Trainable %:    {results['trainable_params']/results['total_params']*100:.2f}%")

    print("\nBy module:")
    print("-"*80)
    for module_name, params in results['module_params'].items():
        print(f"{module_name:30s} | total: {format_number(params['total']):>10s} | "
              f"trainable: {format_number(params['trainable']):>10s}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True,
                       help="Trainer config YAML")
    parser.add_argument("--dataset-config-file", type=str, required=True,
                       help="Dataset config YAML")
    parser.add_argument("--input-size", type=int, nargs=4,
                       default=[1, 3, 224, 224],
                       help="Input shape [batch, C, H, W]")
    parser.add_argument("--compute-flops", action="store_true",
                       help="Also compute FLOPs (needs thop or fvcore)")
    args = parser.parse_args()

    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.freeze()

    print("Loading CLIP...")
    clip_model = load_clip_model(cfg)

    dummy_classnames = [f"class_{i}" for i in range(100)]

    from trainers.adaptive_bidir_maple import CustomCLIP
    model = CustomCLIP(cfg, dummy_classnames, clip_model)

    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            param.requires_grad_(False)

    print("\nCounting parameters...")
    results = count_parameters_detailed(model)
    print_statistics(results)

    if args.compute_flops:
        print("\nComputing FLOPs...")
        input_size = tuple(args.input_size)

        print("\nTHOP:")
        try:
            flops_thop, params_thop = compute_flops_thop(model, input_size)
            if flops_thop:
                print(f"FLOPs: {flops_thop}")
                print(f"Params (thop): {params_thop}")
        except ImportError:
            print("thop not installed. pip install thop")

        print("\nFVCore:")
        try:
            flops_fvcore, total_flops = compute_flops_fvcore(model, input_size)
            if flops_fvcore:
                print(f"FLOPs: {flops_fvcore}")
        except ImportError:
            print("fvcore not installed. pip install fvcore")

    print("\nDone.")
    print("="*80)


if __name__ == "__main__":
    main()
