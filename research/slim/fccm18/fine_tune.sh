#!/bin/sh

DATASET_DIR=/mnt/data2/rz3515/flowers
DATASET=flowers
MODEL_NAME=vgg_16
TRAIN_DIR=/mnt/data2/rz3515/train/flowers/${MODEL_NAME}
CHECKPOINT_PATH=/mnt/data2/rz3515/checkpoints/${MODEL_NAME}.ckpt
NUM_SEPARABLE_LAYERS=$1
SCOPES=$2
TRAINING=$3

# Update parameters if there are separable layers in the model
if [ $NUM_SEPARABLE_LAYERS -ge 1 ]; then
  TRAIN_DIR=${TRAIN_DIR}_dws_${NUM_SEPARABLE_LAYERS}
fi

# The training command
if [ ! -d ${TRAIN_DIR} ] || [ "${TRAINING}" != "" ]; then

  python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_dir=${DATASET_DIR} \
      --dataset_name=${DATASET} \
      --dataset_split_name=train \
      --model_name=${MODEL_NAME} \
      --checkpoint_path=${CHECKPOINT_PATH} \
      --checkpoint_exclude_scopes=${SCOPES} \
      --trainable_scopes=${SCOPES} \
      --learning_rate=1e-3 \
      --max_number_of_steps=10000 \
      --num_separable_layers=${NUM_SEPARABLE_LAYERS}

fi

python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET} \
    --dataset_split_name=validation \
    --model_name=${MODEL_NAME} \
    --num_separable_layers=${NUM_SEPARABLE_LAYERS}
