#!/bin/sh

MODEL_NAME=$1
SCOPES=$2
CKPT=$3

python train_image_classifier.py \
  --train_dir=${MODEL_NAME}_log \
  --dataset_dir=/mnt/data2/tensorflow_imagenet_data \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --model_name=$MODEL_NAME \
  --labels_offset=1 \
  --checkpoint_path=$CKPT \
  --checkpoint_exclude_scopes=$SCOPES \
  --trainable_scopes=$SCOPES \
  --save_summaries_secs=20 \
  --max_number_of_steps=10000 
