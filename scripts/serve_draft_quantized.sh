#!/bin/bash

hostname --ip-address
MODEL="/home/monishwaran/ReasonX/hf_models/Qwen2.5-Math-7B-Instruct-GGUF/Qwen2.5-Math-7B-Instruct-Q3_K_L.gguf"
MODEL_NAME="Qwen2.5-Math-7B-Instruct-Q3_K_L"

python -m sglang.launch_server \
    --model-path $MODEL \
    --tp 1 \
    --dp 1 \
    --port 12340 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.80 \
    --trust-remote-code \
    --context-length 8192