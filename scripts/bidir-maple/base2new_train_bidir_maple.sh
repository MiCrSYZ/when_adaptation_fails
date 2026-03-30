#!/bin/bash

# Custom config
DATA="/your/path"
TRAINER=BiDirMaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_bidir_maple
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.BIDIR_MAPLE.BIDIRECTIONAL True
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.BIDIR_MAPLE.BIDIRECTIONAL True 
fi
