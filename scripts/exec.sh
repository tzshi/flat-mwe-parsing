#!/bin/bash

conda activate some-python-env

EVAL_STEPS=800
DECAY_EVALS=4
DECAY_TIMES=2
DECAY_RATIO=0.1

BATCH_SIZE=8

BETA1=0.9
BETA2=0.999
EPSILON=1e-8
WEIGHT_DECAY=0

CLIP=1.0

CUTOFF=2
WORD_SMOOTH=0.3

WDIMS=100
EDIMS=0
CDIMS=64
PDIMS=0
WORD_DROPOUT=0.33

BILSTM_DIMS=800
BILSTM_LAYERS=3
BILSTM_DROPOUT=0.33

CHAR_HIDDEN=256
CHAR_DROPOUT=0.33

UTAGGER_DIMS=500
UTAGGER_LAYERS=1
UTAGGER_DROPOUT=0.33

HSEL_DIMS=500
HSEL_DROPOUT=0.33

REL_DIMS=100
REL_DROPOUT=0.33

UPOS_WEIGHT=0.1
XPOS_WEIGHT=0.1
HSEL_WEIGHT=1.0
REL_WEIGHT=1.0

MODE=$3
BERT=$4
SEED=$1
LANGUAGE=$2
PARSING_WEIGHT=$5

RUN=${LANGUAGE}-${MODE}-${BERT}-${PARSING_WEIGHT}-${SEED}

LOG_FOLDER=/path/to/models/${RUN}/

TRAIN_FILE=/path/to/data/${LANGUAGE}-train.flat.conllu
DEV_FILE=/path/to/data/${LANGUAGE}-dev.flat.conllu

SAVE_PREFIX=${LOG_FOLDER}

mkdir -p $SAVE_PREFIX
LOG_FILE=$SAVE_PREFIX/log.log

hostname > $LOG_FILE

python -m cdpt.parser \
    - build-vocab $TRAIN_FILE --cutoff ${CUTOFF} \
    - create-parser --batch-size $BATCH_SIZE --word-smooth $WORD_SMOOTH \
        --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --clip $CLIP \
        --wdims $WDIMS --cdims $CDIMS --edims $EDIMS --pdims $PDIMS \
        --word-dropout $WORD_DROPOUT \
        --bilstm-dims $BILSTM_DIMS --bilstm-layers $BILSTM_LAYERS --bilstm-dropout $BILSTM_DROPOUT \
        --char-hidden $CHAR_HIDDEN --char-dropout $CHAR_DROPOUT \
        --utagger-dims $UTAGGER_DIMS --utagger-dropout $UTAGGER_DROPOUT \
        --utagger-layers $UTAGGER_LAYERS \
        --hsel-dims $HSEL_DIMS --hsel-dropout $HSEL_DROPOUT \
        --rel-dims $REL_DIMS --rel-dropout $REL_DROPOUT \
        --upos-weight $UPOS_WEIGHT --xpos-weight $XPOS_WEIGHT \
        --weight-decay $WEIGHT_DECAY \
        --parsing-weight $PARSING_WEIGHT \
        --mode $MODE \
        --bert $BERT \
    - train $TRAIN_FILE --dev $DEV_FILE \
        --eval-steps $EVAL_STEPS --decay-evals $DECAY_EVALS --decay-times $DECAY_TIMES --decay-ratio $DECAY_RATIO \
        --save-prefix $SAVE_PREFIX/ \
    - finish \
>> $LOG_FILE
