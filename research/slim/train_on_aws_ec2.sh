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
    --dataset_dir $HOME/datasets/$dataset_name \
    --checkpoint_path $HOME/checkpoints/$model_name.ckpt \
    --train_dir $HOME/train/$dataset_name/$model_name/fine_tune/train \
    --learning_rate 1e-3 \
    --max_number_of_steps 10000 \
    --checkpoint_exclude_scopes 

  python eval_image_classifier.py \
    --alsologtostderr \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_split_name "test" \
    --dataset_dir $HOME/datasets/$dataset_name \
    --checkpoint_path $HOME/train/$dataset_name/$model_name/fine_tune/train  \
    --train_dir "$HOME/train/$dataset_name/$model_name/fine_tune/eval" 
}


train_eval_model "vgg_16" "indoor67" "vgg_16/fc8"
train_eval_model "vgg_16" "actions40" "vgg_16/fc8"

train_eval_model "vgg_19" "indoor67" "vgg_19/fc8"
train_eval_model "vgg_19" "actions40" "vgg_19/fc8"

train_eval_model "resnet_v1_50" "indoor67" "resnet_v1_50/logits"
train_eval_model "resnet_v1_50" "actions40" "resnet_v1_50/logits"
