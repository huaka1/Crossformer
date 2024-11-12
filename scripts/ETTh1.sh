#!/bin/bash

# 创建log文件夹
mkdir -p log

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 定义日志文件路径
log_file="log/log_$timestamp.log"

# 开始记录日志
{
    echo "############################predict length 24####################################"
    python -u main_crossformer.py --data ETTh1 \
    --in_len 168 --out_len 24 --seg_len 6 \
    --learning_rate 1e-4 --itr 5 

    echo "############################predict length 48####################################"
    python -u main_crossformer.py --data ETTh1 \
    --in_len 168 --out_len 48 --seg_len 6 \
    --learning_rate 1e-4 --itr 5 

    echo "############################predict length 168###################################"
    python -u main_crossformer.py --data ETTh1  \
    --in_len 720 --out_len 168 --seg_len 24 \
    --learning_rate 1e-5 --itr 5 

    echo "############################predict length 336###################################"
    python -u main_crossformer.py --data ETTh1 \
    --in_len 720 --out_len 336 --seg_len 24 \
    --learning_rate 1e-5 --itr 5 

    echo "############################predict length 720###################################"
    python -u main_crossformer.py --data ETTh1 \
    --in_len 720 --out_len 720 --seg_len 24 \
    --learning_rate 1e-5 --itr 5 

} 2>&1 | tee "$log_file"
