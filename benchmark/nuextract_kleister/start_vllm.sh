#!/bin/bash

echo "Starting vLLM container for NuExtract-2.0 on NVIDIA L4..."

# Run the container in detached mode so the script can continue
docker run -d --rm --gpus all \
  --name nuextract_vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 23333:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model numind/NuExtract-2.0-4B \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 32768 \
  --chat-template-content-format openai

echo "Container nuextract_vllm started."