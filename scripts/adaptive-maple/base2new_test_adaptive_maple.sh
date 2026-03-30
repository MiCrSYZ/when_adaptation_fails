#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# custom config
DATA="/your/path"
TRAINER=AdaptiveMaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_adaptive_maple
SHOTS=16
#LOADEP=5
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/AdaptiveMaPLe/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/AdaptiveMaPLe/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi