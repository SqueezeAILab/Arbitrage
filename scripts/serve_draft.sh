#!/bin/bash

hostname --ip-address

MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME="Llama-3.2-1B-Instruct"

python -m sglang.launch_server \
    --model-path $MODEL \
    --tp 1 \
    --dp 1 \
    --port 12340 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.80 \
    --trust-remote-code \
    --context-length 8192