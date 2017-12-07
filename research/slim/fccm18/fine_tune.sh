#!/bin/sh

DATASET_DIR=/mnt/data2/rz3515/flowers
DATASET=flowers
TRAIN_DIR=/mnt/data2/rz3515/train/flowers/vgg_16
CHECKPOINT_PATH=/mnt/data2/rz3515/checkpoints/vgg_16.ckpt

if [ ! -d ${TRAIN_DIR} ] || [ "$1" = "train" ]; then

echo "Training ..."

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET} \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --trainable_scopes=vgg_16/fc8 \
    --learning_rate=1e-3 \
    --max_number_of_steps=10000

fi

python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET} \
    --dataset_split_name=validation \
    --model_name=vgg_16
