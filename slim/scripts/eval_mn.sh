#!/bin/sh

MODEL_NAME=$1

python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=${MODEL_NAME}/mobilenet_v1_1.0_224.ckpt \
  --dataset_dir=/mnt/data2/tensorflow_imagenet_data \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=$MODEL_NAME
