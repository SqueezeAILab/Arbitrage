#!/bin/bash

PROMPT_TYPE="qwen25-math-cot" # or "llama3-custom"

# change to llama3 models if you want to use llama3-custom
DRAFT_MODEL="Qwen2.5-Math-7B-Instruct-Q3_K_L"
DRAFT_TOKENIZER="Qwen/Qwen2.5-Math-7B-Instruct"

# change to llama3 models if you want to use llama3-custom
TARGET_MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

ROUTER=<placeholder for router model>
ROUTER_TOKENIZER="HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"

PRM="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

DRAFT_IP_ADDRESS="http://localhost:12340/v1"
TARGET_IP_ADDRESS="http://localhost:12341/v1"
PRM_IP_ADDRESS="http://localhost:12342/v1"
ROUTER_IP_ADDRESS="http://localhost:12343/v1"


RUN_TYPE="router" # or "generate", "oracle", "rsd"
SPLIT="test"
NUM_TEST_SAMPLE=-1
DATA_NAME="math500" # or "olympiadbench" or "NuminaMath-CoT" (for data generation)
START_SAMPLE=0 
END_SAMPLE=-1
GENERATE_TAG="data_generation"
OUTPUT_DIR="outputs/example_run"

for PRM_THRESHOLD in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
python3 -u arbitrage.py \
    --run_type ${RUN_TYPE} \
    --data_names ${DATA_NAME} \
    --data_dir "./utils/qwen25_math_evaluation/data" \
    --draft_model_name_or_path ${DRAFT_MODEL} \
    --draft_model_ip_address ${DRAFT_IP_ADDRESS} \
    --draft_model_tokenizer ${DRAFT_TOKENIZER} \
    --target_model_name_or_path ${TARGET_MODEL} \
    --target_model_ip_address ${TARGET_IP_ADDRESS} \
    --prm_name_or_path ${PRM} \
    --prm_ip_address ${PRM_IP_ADDRESS} \
    --router_name_or_path ${ROUTER} \
    --router_ip_address ${ROUTER_IP_ADDRESS} \
    --router_tokenizer ${ROUTER_TOKENIZER} \
    --prm_threshold ${PRM_THRESHOLD} \
    --max_steps 100 \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start ${START_SAMPLE} \
    --end ${END_SAMPLE} \
    --save_outputs \
    --overwrite \
    --generate_tag ${GENERATE_TAG} \
    --max_tokens_per_call 4096 \
    --annotate
done