#!/bin/bash

# custom config
DATA="/your/path"
TRAINER=ImprovedAdaptiveBiDirMaPLe

DATASET=$1
SEED=$2

CFG=improved_vit_b16_adaptive_bidir_maple
SHOTS=16
LOADEP=5
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}_A/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/AdaptiveBiDirMaPLe/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_GRADIENT_SCALING True \
    TRAINER.ADAPTIVE_BIDIR_MAPLE.GRADIENT_SCALE_MAX 10.0 \
    TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_ENTROPY_REG False \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/AdaptiveBiDirMaPLe/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_GRADIENT_SCALING True \
    TRAINER.ADAPTIVE_BIDIR_MAPLE.GRADIENT_SCALE_MAX 10.0 \
    TRAINER.ADAPTIVE_BIDIR_MAPLE.USE_ENTROPY_REG False \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi