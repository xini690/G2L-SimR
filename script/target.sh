#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

config_extractor=../cfgs/featExtraCfg.yaml
config_train=../cfgs/synNetCfg.yaml

python ../pretrain_feature_extractor.py \
    --config $config_extractor

python ../train.py \
    --config $config_train