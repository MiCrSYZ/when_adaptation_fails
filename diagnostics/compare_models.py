import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import os
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from diagnostics.advanced_experiments import CrossModelComparison
from yacs.config import CfgNode as CN
import trainers.adaptive_bidir_maple
import trainers.bidir_maple
import trainers.maple


def extend_cfg(cfg):

    # MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    # AdaptiveBiDirMaPLe
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE = CN()
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.N_CTX_MAX = 4
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PREC = "fp16"
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.PROMPT_DEPTH_MAX = 8
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_SPARSITY = 0.001
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_DEPTH_SMOOTH = 0.0002
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.TYPE = "full"
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.FIXED_N_CTX = 4
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.FIXED_PROMPT_DEPTH = 8
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.BIDIRECTIONAL = True
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.CYCLE_LOSS_WEIGHT = 0.1
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.MAPPING_HIDDEN_DIM_RATIO = 0.5
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    # BiDirMaPLe
    cfg.TRAINER.BIDIR_MAPLE = CN()
    cfg.TRAINER.BIDIR_MAPLE.N_CTX = 4
    cfg.TRAINER.BIDIR_MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIDIR_MAPLE.PREC = "fp16"
    cfg.TRAINER.BIDIR_MAPLE.PROMPT_DEPTH = 9
    cfg.TRAINER.BIDIR_MAPLE.BIDIRECTIONAL = True
    cfg.TRAINER.BIDIR_MAPLE.CYCLE_LOSS_WEIGHT = 0.2
    cfg.TRAINER.BIDIR_MAPLE.CYCLE_WARMUP_EPOCHS = 2
    cfg.TRAINER.BIDIR_MAPLE.USE_EMA = False
    cfg.TRAINER.BIDIR_MAPLE.EMA_DECAY = 0.999
    cfg.TRAINER.BIDIR_MAPLE.EMA_START_EPOCH = 5
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def load_model_from_checkpoint(trainer_name, config_path, model_dir, dataset_config, epoch=None):
    """
    Load a trained model from checkpoint.

    Args:
        trainer_name: Trainer name, e.g. 'AdaptiveBiDirMaPLe'.
        config_path: Trainer YAML config path.
        model_dir: Checkpoint directory.
        dataset_config: Dataset YAML path.
        epoch: Epoch to load (None = best checkpoint).
    """
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. Merge dataset config
    cfg.merge_from_file(dataset_config)

    # 2. Merge trainer config
    if config_path:
        cfg.merge_from_file(config_path)

    # 3. Set trainer name explicitly
    cfg.TRAINER.NAME = trainer_name

    # 4. Required output dirs and seed
    cfg.OUTPUT_DIR = model_dir
    cfg.SEED = 1

    # 5. Build trainer
    trainer = build_trainer(cfg)

    # 6. Load weights
    trainer.load_model(model_dir, epoch=epoch)

    return trainer.model


def main():
    """Entry point."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset YAML
    dataset_config = 'configs/datasets/eurosat.yaml'

    # Models to load
    models_to_load = [
        {
            'name': 'adaptive_epoch_10',
            'trainer': 'AdaptiveBiDirMaPLe',
            'config': 'configs/trainers/AdaptiveBiDirMaPLe/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple.yaml',
            'dir': 'output/base2new/train_base/eurosat/shots_16/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple/seed1_10epoch_LR50',
            'epoch': 10
        },
        {
            'name': 'adaptive_epoch_5',
            'trainer': 'AdaptiveBiDirMaPLe',
            'config': 'configs/trainers/AdaptiveBiDirMaPLe/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple.yaml',
            'dir': 'output/base2new/train_base/eurosat/shots_16/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple/seed1_5epoch_LR50',
            'epoch': 5
        },
        {
            'name': 'adaptive_epoch_1',
            'trainer': 'AdaptiveBiDirMaPLe',
            'config': 'configs/trainers/AdaptiveBiDirMaPLe/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple.yaml',
            'dir': 'output/base2new/train_base/eurosat/shots_16/AdaptiveBiDirMaPLe/vit_b16_adaptive_bidir_maple/seed1_epoch1_LR50',
            'epoch': 1
        },
        {
            'name': 'maple_epoch_5',
            'trainer': 'MaPLe',
            'config': 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml',
            'dir': 'output/base2new/train_base/eurosat/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed1',
            'epoch': 5
        },
        {
            'name': 'bidir_epoch_5',
            'trainer': 'BiDirMaPLe',
            'config': 'configs/trainers/BiDirMaPLe/vit_b16_bidir_maple.yaml',
            'dir': 'output/base2new/train_base/eurosat/shots_16/BiDirMaPLe/vit_b16_bidir_maple/seed1/logs_20251016-201602',
            'epoch': 5
        },
    ]

    print("Loading models...")
    models_dict = {}

    for model_info in models_to_load:
        try:
            if not os.path.exists(model_info['dir']):
                print(f"  Skip {model_info['name']} (missing directory: {model_info['dir']})")
                continue

            print(f"  Loading {model_info['name']}...")
            model = load_model_from_checkpoint(
                trainer_name=model_info['trainer'],
                config_path=model_info['config'],
                model_dir=model_info['dir'],
                dataset_config=dataset_config,
                epoch=model_info['epoch']
            )
            models_dict[model_info['name']] = model
            print(f"  OK: loaded {model_info['name']}")

        except Exception as e:
            print(f"  Warning: failed to load {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    if not models_dict:
        print("\nError: no models loaded. Check paths.")
        return

    print(f"\nLoaded {len(models_dict)} model(s).")

    print("\nBuilding data loader...")
    from dassl.data import DataManager

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(dataset_config)
    cfg.DATASET.ROOT = '/your/path'

    dm = DataManager(cfg)
    val_loader = dm.val_loader

    print("\n" + "=" * 80)
    print("Running cross-model comparison...")
    print("=" * 80 + "\n")

    comparison_results = CrossModelComparison.compare_models_across_epochs(
        models_dict,
        val_loader,
        device,
        'output/cross_model_comparison'
    )

    print("\n" + "=" * 80)
    print("Comparison done.")
    print("=" * 80)
    print("\nOutputs:")
    print("  - output/cross_model_comparison/drift_comparison.png")
    print("  - output/cross_model_comparison/quality_comparison.png")


if __name__ == '__main__':
    import datasets.imagenet
    import datasets.caltech101
    import datasets.eurosat

    main()