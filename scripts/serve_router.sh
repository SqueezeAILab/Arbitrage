#!/bin/bash

hostname --ip-address
MODEL=trained_router_model

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=7
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --tensor-parallel-size 1 \
        --port 12343 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.95