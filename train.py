import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt
import trainers.adaptive_maple
import trainers.bidir_maple
import trainers.adaptive_bidir_maple
import trainers.param_matched_trainer
import trainers.implicit_reg_trainers
import trainers.improved_adaptive_bidir


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # Map new CLI args to cfg for CoOp/CoCoOp
    # We keep defaults in extend_cfg and only override if user specified
    if hasattr(args, "use_gate") and args.use_gate is not None:
        cfg.TRAINER.COOP.USE_GATE = args.use_gate
        cfg.TRAINER.COCOOP.USE_GATE = args.use_gate
    if getattr(args, "gate_type", None):
        cfg.TRAINER.COOP.GATE_TYPE = args.gate_type
        cfg.TRAINER.COCOOP.GATE_TYPE = args.gate_type
    if getattr(args, "gate_init_std", None) is not None:
        cfg.TRAINER.COOP.GATE_INIT_STD = args.gate_init_std
        cfg.TRAINER.COCOOP.GATE_INIT_STD = args.gate_init_std
    if getattr(args, "gate_lr", None) is not None and args.gate_lr > 0:
        # store as lr multiplier relative to base LR
        base_lr = cfg.OPTIM.LR
        cfg.TRAINER.COOP.GATE_LR_MUL = float(args.gate_lr) / float(base_lr)
        cfg.TRAINER.COCOOP.GATE_LR_MUL = float(args.gate_lr) / float(base_lr)
    if getattr(args, "gate_reg_lambda", None) is not None:
        cfg.TRAINER.COOP.GATE_REG_LAMBDA = args.gate_reg_lambda
        cfg.TRAINER.COCOOP.GATE_REG_LAMBDA = args.gate_reg_lambda
    if getattr(args, "gate_mode", None):
        cfg.TRAINER.COCOOP.GATE_MODE = args.gate_mode
    if hasattr(args, "param_matched") and args.param_matched is not None:
        cfg.TRAINER.COOP.PARAM_MATCHED = args.param_matched
        cfg.TRAINER.COCOOP.PARAM_MATCHED = args.param_matched
    if hasattr(args, "equalize_grad") and args.equalize_grad is not None:
        cfg.TRAINER.COOP.EQUALIZE_GRAD = args.equalize_grad
        cfg.TRAINER.COCOOP.EQUALIZE_GRAD = args.equalize_grad
    if getattr(args, "alpha_max", None) is not None:
        cfg.TRAINER.COOP.ALPHA_MAX = args.alpha_max
        cfg.TRAINER.COCOOP.ALPHA_MAX = args.alpha_max


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    # gating defaults (off by default)
    cfg.TRAINER.COOP.USE_GATE = False
    cfg.TRAINER.COOP.GATE_TYPE = "scalar"  # scalar|vector
    cfg.TRAINER.COOP.GATE_INIT_STD = 0.02
    cfg.TRAINER.COOP.GATE_LR_MUL = 1.0
    cfg.TRAINER.COOP.GATE_REG_LAMBDA = 0.0
    cfg.TRAINER.COOP.PARAM_MATCHED = False
    cfg.TRAINER.COOP.EQUALIZE_GRAD = False
    cfg.TRAINER.COOP.ALPHA_MAX = 10.0

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp
    # gating defaults
    cfg.TRAINER.COCOOP.USE_GATE = False
    cfg.TRAINER.COCOOP.GATE_TYPE = "scalar"  # scalar|vector
    cfg.TRAINER.COCOOP.GATE_INIT_STD = 0.02
    cfg.TRAINER.COCOOP.GATE_LR_MUL = 1.0
    cfg.TRAINER.COCOOP.GATE_REG_LAMBDA = 0.0
    cfg.TRAINER.COCOOP.GATE_MODE = "static"  # static|conditional
    cfg.TRAINER.COCOOP.PARAM_MATCHED = False
    cfg.TRAINER.COCOOP.EQUALIZE_GRAD = False
    cfg.TRAINER.COCOOP.ALPHA_MAX = 10.0
    # memory optimization defaults
    cfg.TRAINER.COCOOP.USE_CHECKPOINT = True  # use gradient checkpointing for text encoder
    cfg.TRAINER.COCOOP.SEQUENTIAL_PROCESSING = True  # process samples one at a time

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for AdaptiveMaPLe
    cfg.TRAINER.ADAPTIVE_MAPLE = CN()
    cfg.TRAINER.ADAPTIVE_MAPLE.N_CTX_MAX = 8  # max prompt length
    cfg.TRAINER.ADAPTIVE_MAPLE.CTX_INIT = "a photo of a"  # init text
    cfg.TRAINER.ADAPTIVE_MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.ADAPTIVE_MAPLE.PROMPT_DEPTH_MAX = 12  # max insert depth
    cfg.TRAINER.ADAPTIVE_MAPLE.LAMBDA_SPARSITY = 0.01  # sparsity regular weight
    cfg.TRAINER.ADAPTIVE_MAPLE.LAMBDA_DEPTH_SMOOTH = 0.001  # depth smoothness weight
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    # Ablation: TYPE full | length_only | depth_only
    cfg.TRAINER.ADAPTIVE_MAPLE.TYPE = "full"
    cfg.TRAINER.ADAPTIVE_MAPLE.FIXED_N_CTX = 4  # fixed length when depth_only (<= N_CTX_MAX)
    cfg.TRAINER.ADAPTIVE_MAPLE.FIXED_PROMPT_DEPTH = 6  # fixed depth when length_only (<= PROMPT_DEPTH_MAX)

    # Config for BiDirMaPLe
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

    # Config for AdaptiveBiDirMaPLe
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

    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_GRADIENT_SCALING = False
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.GRADIENT_SCALE_MAX = 10.0
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_ENTROPY_REG = False
    cfg.TRAINER.ADAPTIVE_BIDIR_MAPLE.LAMBDA_ENTROPY = 0.01

    cfg.TRAINER.PARAM_MATCHED = CN()
    cfg.TRAINER.PARAM_MATCHED.N_CTX_MAX = 8
    cfg.TRAINER.PARAM_MATCHED.CTX_INIT = "a photo of a"
    cfg.TRAINER.PARAM_MATCHED.PREC = "fp16"
    cfg.TRAINER.PARAM_MATCHED.PROMPT_DEPTH_MAX = 12
    cfg.TRAINER.PARAM_MATCHED.LAMBDA_SPARSITY = 0.01
    cfg.TRAINER.PARAM_MATCHED.LAMBDA_DEPTH_SMOOTH = 0.001
    cfg.TRAINER.PARAM_MATCHED.TYPE = "full"  # full / length_only / depth_only
    cfg.TRAINER.PARAM_MATCHED.FIXED_N_CTX = 4
    cfg.TRAINER.PARAM_MATCHED.FIXED_PROMPT_DEPTH = 6
    cfg.TRAINER.PARAM_MATCHED.BIDIRECTIONAL = True
    cfg.TRAINER.PARAM_MATCHED.CYCLE_LOSS_WEIGHT = 0.1
    cfg.TRAINER.PARAM_MATCHED.MAPPING_HIDDEN_DIM_RATIO = 0.5
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY = CN()
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.N_CTX_MAX = 8
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.CTX_INIT = "a photo of a"
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.PREC = "fp16"
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.PROMPT_DEPTH_MAX = 12
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.LAMBDA_SPARSITY = 0.01
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.LAMBDA_DEPTH_SMOOTH = 0.001
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.TYPE = "full"  # full / length_only / depth_only
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.FIXED_N_CTX = 4
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.FIXED_PROMPT_DEPTH = 6
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.BIDIRECTIONAL = True
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.CYCLE_LOSS_WEIGHT = 0.1
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.MAPPING_HIDDEN_DIM_RATIO = 0.5
    cfg.TRAINER.ADAPTIVE_FREEZE_EARLY.FREEZE_EPOCH = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN = CN()
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.N_CTX_MAX = 8
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.CTX_INIT = "a photo of a"
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.PREC = "fp16"
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.PROMPT_DEPTH_MAX = 12
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.LAMBDA_SPARSITY = 0.01
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.LAMBDA_DEPTH_SMOOTH = 0.001
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.TYPE = "full"  # full / length_only / depth_only
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.FIXED_N_CTX = 4
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.FIXED_PROMPT_DEPTH = 6
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.BIDIRECTIONAL = True
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.CYCLE_LOSS_WEIGHT = 0.1
    cfg.TRAINER.ADAPTIVE_ALWAYS_FROZEN.MAPPING_HIDDEN_DIM_RATIO = 0.5
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG = CN()
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.N_CTX_MAX = 8
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.CTX_INIT = "a photo of a"
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.PREC = "fp16"
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.PROMPT_DEPTH_MAX = 12
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.LAMBDA_SPARSITY = 0.01
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.LAMBDA_DEPTH_SMOOTH = 0.001
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.TYPE = "full"  # full / length_only / depth_only
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.FIXED_N_CTX = 4
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.FIXED_PROMPT_DEPTH = 6
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.BIDIRECTIONAL = True
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.CYCLE_LOSS_WEIGHT = 0.1
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.MAPPING_HIDDEN_DIM_RATIO = 0.5
    cfg.TRAINER.ADAPTIVE_EXPLICIT_REG.DROPOUT_RATE = 0.1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    # Gating experiment CLI
    parser.add_argument("--use_gate", action="store_true", help="enable adaptive gating")
    parser.add_argument("--gate_type", type=str, default=None, choices=["scalar", "vector"], help="gate parameterization")
    parser.add_argument("--gate_init_std", type=float, default=None, help="std for gate logits init (normal)")
    parser.add_argument("--gate_lr", type=float, default=None, help="learning rate for gate params (absolute)")
    parser.add_argument("--gate_reg_lambda", type=float, default=None, help="regularization weight for gate (entropy)")
    parser.add_argument("--gate_mode", type=str, default=None, choices=["static", "conditional"], help="CoCoOp gate mode")
    parser.add_argument("--param_matched", action="store_true", help="enable parameter-matched baseline instead of gating")
    parser.add_argument("--equalize_grad", action="store_true", help="enable gradient equalization for gate logits")
    parser.add_argument("--alpha_max", type=float, default=None, help="max alpha for gradient equalization clip")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)