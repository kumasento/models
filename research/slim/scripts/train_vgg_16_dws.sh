#!/bin/bash
# This script trains layer replaced VGG-16 models.

DATASET_DIR=/mnt/data2/tensorflow_imagenet_data

TRAIN_DIR=/mnt/data2/rz3515/train/vgg_16_dws_log

MODEL_NAME=vgg_16_dws

ORIGIN_CHECKPOINT_PATH=/mnt/data2/rz3515/checkpoints/vgg_16.ckpt

EXCLUDE_SCOPES=vgg_16/dws_conv5
SCOPES=vgg_16/dws_conv5,vgg_16/fc6,vgg_16/fc7,vgg_16/fc8

MAX_NUMBER_OF_STEPS=$1
if [ ! -z $2 ]; then
  LEARNING_RATE="--learning_rate=$2"
fi

if [ ! -d $TRAIN_DIR ]; then
# use fine-tuning to get the training directory
python train_image_classifier.py \
  --train_dir=$TRAIN_DIR \
  --dataset_dir=$DATASET_DIR \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --model_name=$MODEL_NAME \
  --checkpoint_path=$ORIGIN_CHECKPOINT_PATH \
  --checkpoint_exclude_scopes=$EXCLUDE_SCOPES \
  --trainable_scopes=$SCOPES \
  --labels_offset=1 \
  --log_every_n_steps=10 \
  --weight_decay=0.00001 \
  --optimizer=momentum \
  --num_clones=2 \
  --max_number_of_steps=1000
fi

python train_image_classifier.py \
  --train_dir=$TRAIN_DIR \
  --dataset_dir=$DATASET_DIR \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --model_name=$MODEL_NAME \
  --labels_offset=1 \
  --trainable_scopes=$SCOPES \
  --batch_size=32 \
  --num_preprocessing_threads=8 \
  --num_readers=4 \
  --weight_decay=0.00001 \
  --optimizer=momentum \
  --num_clones=2 \
  $LEARNING_RATE \
  --max_number_of_steps=$MAX_NUMBER_OF_STEPS
