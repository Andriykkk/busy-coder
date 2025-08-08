#!/bin/bash

# This script starts the QLoRA fine-tuning process using train.py.
# It is configured to use the xingyaoww/code-act dataset and applies
# loss masking to train only on the assistant's responses, mimicking the
# original code-act project's methodology.

# --- Configuration ---

# Specify the base model to fine-tune from Hugging Face.
# Using a smaller model like Qwen2-0.5B is good for initial testing.
# For better performance, consider models like 'deepseek-ai/deepseek-coder-6.7b-instruct'.
# MODEL_NAME="Qwen/Qwen2-0.5B"
MODEL_NAME="sbintuitions/tiny-lm-chat"
# Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8

# Directory to save the final LoRA adapter and any intermediate checkpoints.
# A timestamp is added to prevent overwriting previous runs.
OUTPUT_DIR="./output/sbintuitions/tiny-lm-chat-qlora-$(date +%s)"

# A dedicated path to save the processed dataset. This avoids
# re-downloading and re-processing the data on every run.
DATASET_CACHE_PATH="./processed_data/code-act"

# --- Execution ---

echo "Starting QLoRA fine-tuning..."
echo "Model: $MODEL_NAME"
echo "Output Directory: $OUTPUT_DIR"

python3.10 train/main.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "xingyaoww/code-act" \
    --dataset_path "$DATASET_CACHE_PATH" \
    \
    `# Quantization settings for 8-bit` \
    --bits 8 \
    \
    `# LoRA settings` \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    \
    `# Training hyperparameters` \
    --bf16 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "constant" \
    --max_steps 20 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 200 \
    --seed 42 \
    --save_merged_model


echo "--- Training Finished ---"
echo "Final LoRA adapter saved in $OUTPUT_DIR/final_adapter"
