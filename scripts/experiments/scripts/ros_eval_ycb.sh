#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi
#  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  #--model trained_checkpoints/ycb/pose_model_13_0.01985655868300905.pth \
python ./tools/ros_eval_ycb.py --dataset_root ./datasets/ycb/YCB_Video_Dataset\
  --model trained_checkpoints/ycb/pose_model_13_0.01985655868300905.pth \
  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth \
  --checkpoint_path trained_checkpoints/ycb/best_dice_loss.pth \
  --num_classes 21 \
  --context_path resnet101
