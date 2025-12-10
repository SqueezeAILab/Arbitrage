#!/bin/bash

hostname --ip-address

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_NAME="Qwen2.5-Math-7B-Instruct"

python -m sglang.launch_server \
    --model-path $MODEL \
    --tp 1 \
    --dp 1 \
    --port 12341 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.80 \
    --trust-remote-code \
    --context-length 8192