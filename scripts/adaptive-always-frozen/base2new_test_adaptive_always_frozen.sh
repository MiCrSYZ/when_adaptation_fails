#!/bin/bash

# custom config
DATA="/your/path"
TRAINER=AdaptiveAlwaysFrozen

DATASET=$1
SEED=$2

CFG=vit_b16_always_frozen
SHOTS=16
LOADEP=5
SUB=new

MODEL_DIR=output/ExperimentB/AdaptiveAlwaysFrozen/${DATASET}/${SEED}
DIR=output/ExperimentB/test/AdaptiveAlwaysFrozen/${DATASET}/${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi