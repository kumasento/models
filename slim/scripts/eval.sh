#!/bin/sh

MODEL_NAME=$1
CKPT=$2

python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=${MODEL_NAME}_log/model.ckpt-$CKPT \
  --dataset_dir=/mnt/data2/tensorflow_imagenet_data \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=$MODEL_NAME \
  --labels_offset=1
