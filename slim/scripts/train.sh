#!/bin/sh

MODEL_NAME=$1
LEARNING_RATE=$2
MAX_NUMBER_OF_STEPS=$3
SCOPES=$4

python train_image_classifier.py \
  --train_dir=${MODEL_NAME}_log \
  --dataset_dir=/mnt/data2/tensorflow_imagenet_data \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --model_name=$MODEL_NAME \
  --labels_offset=1 \
  --trainable_scopes=$SCOPES \
  --save_summaries_secs=30 \
  --learning_rate=$LEARNING_RATE \
  --max_number_of_steps=$MAX_NUMBER_OF_STEPS \
  --ignore-missing-vars
