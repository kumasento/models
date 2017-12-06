#!/bin/sh

MODEL_NAME=vgg_16_dws
CKPT=$1
MODEL_CKPT_DIR=/mnt/data2/rz3515/train/vgg_16_dws_log

python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=${MODEL_CKPT_DIR}/model.ckpt-$CKPT \
  --dataset_dir=/mnt/data2/tensorflow_imagenet_data \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=$MODEL_NAME \
  --labels_offset=1
