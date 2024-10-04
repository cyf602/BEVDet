#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORKDIR="./work_dirs/$3"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=$((RANDOM % 60 + 2000))
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# CUDA_VISIBLE_DEVICES="2,3" \
if [ -n "$3" ]; then
    echo "work-dir:$WORKDIR"
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --work-dir $WORKDIR \
        --seed 0 \
        --launcher pytorch ${@:4}
else
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --seed 0 \
        --launcher pytorch ${@:3}
fi
