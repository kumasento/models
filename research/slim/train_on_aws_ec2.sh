#!/usr/bin/env bash

# Train several networks on an AWS EC2 instance
# May only work for my case


train_eval() {
  local model_name=$1
  local dataset_name=$2
  local checkpoint_exclude_scopes=$3

  python train_image_classifier.py \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_split_name train \
    --dataset_dir $NAS_HOME/datasets/$dataset_name \
    --checkpoint_path $NAS_HOME/checkpoints/$model_name.ckpt \
    --train_dir $NAS_HOME/train/cvpr19/$dataset_name/$model_name/fine_tune/train \
    --learning_rate 1e-3 \
    --max_number_of_steps 30000 \
    --checkpoint_exclude_scopes $checkpoint_exclude_scopes

  python eval_image_classifier.py \
    --alsologtostderr \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_split_name "validation" \
    --dataset_dir $NAS_HOME/datasets/$dataset_name \
    --checkpoint_path $NAS_HOME/train/cvpr19/$dataset_name/$model_name/fine_tune/train  \
    --train_dir "$NAS_HOME/train/cvpr19/$dataset_name/$model_name/fine_tune/eval" 
}


# train_eval "resnet_v1_50" "indoor67" "resnet_v1_50/logits"
# train_eval "resnet_v1_50" "actions40" "resnet_v1_50/logits"
train_eval "resnet_v1_50" "dogs120" "resnet_v1_50/logits"

# train_eval "vgg_16" "indoor67" "vgg_16/fc8"
# train_eval "vgg_16" "actions40" "vgg_16/fc8"
# train_eval "vgg_16" "dogs120" "vgg_16/fc8"

# train_eval "vgg_19" "indoor67" "vgg_19/fc8"
# train_eval "vgg_19" "actions40" "vgg_19/fc8"
