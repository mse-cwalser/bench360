#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch one of several LLM inference engines via Docker
#
# Usage:
#   ./launch_engine --engine=<engine> --model=<model> --name<name>
#
#   <engine> ∈ { tgi | vllm | mii | sglang | lmdeploy }
#   <model>  is the Hugging Face model ID (e.g. “mistralai/Mistral-7B-Instruct-v0.3”)
#   <name>   is the name for the container
#
# Example:
#   ./launch_engine --engine=tgi --model=mistralai/Mistral-7B-Instruct-v0.3 -name="bench360_inference_engine"
#
# Notes:
#   • Expects HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment.
#   • Always listens on 127.0.0.1:23333 inside the container→host.
#   • Uses $HOME/.cache/huggingface as cache Dir.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Parse arguments “--engine=…” and “--model=…” ────────────────────────────
ENGINE=""
MODEL=""

for ARG in "$@"; do
  case "$ARG" in
    --engine=*)
      ENGINE="${ARG#--engine=}"
      ;;
    --model=*)
      MODEL="${ARG#--model=}"
      ;;
    --name=*)
      NAME="${ARG#--name=}"
      ;;
    *)
      echo "Unknown argument: $ARG"
      echo "Usage: $0 --engine=<tgi|vllm|mii|sglang|lmdeploy> --model=<your-org/your-model-name> --name=<container-name>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" || -z "$NAME" ]]; then
  echo "Error: All --engine, --model and --name must be provided."
  echo "Usage: $0 --engine=<tgi|vllm|mii|sglang|lmdeploy> --model=<your-org/your-model-name> --name=<container-name>"
  exit 1
fi

# ─── Common variables ───────────────────────────────────────────────────────
PORT=23333
CACHE_DIR="$HOME/.cache/huggingface"

# Ensure at least one token is set
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Error: You must export HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
  exit 1
fi

# ─── Select and run the requested engine ────────────────────────────────────
case "$ENGINE" in

  tgi)
    # ────────────────────────────────────────────────────────────────────────
    # TGI (Text Generation Inference) container:
    #
    # docker run --rm \
    #   --gpus all \
    #   --name $NAME \
    #   -v "$HOME/.cache/huggingface:/data" \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HF_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   ghcr.io/huggingface/text-generation-inference:3.3.1 \
    #     --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    #     --trust-remote-code \
    #     --port 23333 \
    #     --max-client-batch-size 128
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/data" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HF_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      ghcr.io/huggingface/text-generation-inference:3.3.1 \
        --model-id "$MODEL" \
        --trust-remote-code \
        --port "$PORT" \
        --max-client-batch-size 512
    ;;

  vllm)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM (OpenAI-compatible) container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   --name $NAME \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   vllm/vllm-openai:latest \
    #     --model mistralai/Mistral-7B-Instruct-v0.3 \
    #     --port 23333
    # max tokens qwen: 18300
    # max tokens mistral:
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --name $NAME \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 16384 \
        --port "$PORT"
    ;;

  vllm-quant)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM Quantized (OpenAI-compatible) container:
    #
    # Supports: fp8, awq, gptq, marlin, bitsandbytes, squeezellm, etc.
    # Override via environment: QUANT_METHOD=awq ./launch_engine ...
    # ────────────────────────────────────────────────────────────────────────
    QUANT_METHOD="${QUANT_METHOD:-fp8}"

    echo "Launching vLLM with quantization method: $QUANT_METHOD"

    docker run --rm \
      --name "$NAME" \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 31872 \
        --gpu-memory-utilization 0.95 \
        --port "$PORT" \
        --quantization fp8
    ;;

  vllm-vl)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM (OpenAI-compatible) container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   vllm/vllm-openai:latest \
    #     --model mistralai/Mistral-7B-Instruct-v0.3 \
    #     --port 23333
    #
    # ───────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 14000 \
        --port "$PORT" \
        --gpu-memory-utilization 0.95 \
        --no-enable-prefix-caching \
        --limit-mm-per-prompt '{"image": 15}'
    ;;

  vllm-vl-quant)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM (OpenAI-compatible) container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   vllm/vllm-openai:latest \
    #     --model mistralai/Mistral-7B-Instruct-v0.3 \
    #     --port 23333
    #
    # ───────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 28000 \
        --port "$PORT" \
        --gpu-memory-utilization 0.95 \
        --no-enable-prefix-caching \
        --quantization fp8 \
        --limit-mm-per-prompt '{"image": 15}'
    ;;

    vllm-quant-cache)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM Quantized (OpenAI-compatible) container:
    #
    # Supports: fp8, awq, gptq, marlin, bitsandbytes, squeezellm, etc.
    # Override via environment: QUANT_METHOD=awq ./launch_engine ...
    # ────────────────────────────────────────────────────────────────────────
    QUANT_METHOD="${QUANT_METHOD:-fp8}"

    echo "Launching vLLM with quantization method: $QUANT_METHOD"

    docker run --rm \
      --name "$NAME" \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 31872 \
        --gpu-memory-utilization 0.9 \
        --port "$PORT" \
        --quantization fp8 \
        --kv_cache_dtype fp8
    ;;

  lmdeploy)
    # ────────────────────────────────────────────────────────────────────────
    # LMDeploy container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   --name $NAME \
    #   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    #   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   openmmlab/lmdeploy:latest \
    #     lmdeploy serve api_server mistralai/Mistral-7B-Instruct-v0.3 \
    #     --server-port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      openmmlab/lmdeploy:latest \
        lmdeploy serve api_server "$MODEL" \
        --server-port "$PORT" \
        --cache-max-entry-count 0.5
    ;;

  sglang)
    # ────────────────────────────────────────────────────────────────────────
    # SGLang (Slim Graph Language) container:
    #
    # docker run --gpus all \
    #   --name $NAME \
    #   -p 127.0.0.1:23333:23333 \
    #   -v ~/.cache/huggingface:/root/.cache/huggingface \
    #   --ipc=host \
    #   lmsysorg/sglang:latest \
    #   bash -c "\
    #     pip install --no-cache-dir protobuf sentencepiece --break-system-packages && \
    #     python3 -m sglang.launch_server \
    #       --model-path mistralai/Mistral-7B-Instruct-v0.3 \
    #       --host 0.0.0.0 \
    #       --port 23333 \
    #       --context-length 4096
    #   "
    # ────────────────────────────────────────────────────────────────────────

    docker run --rm \
      --gpus all \
      --name $NAME \
      --network host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --ipc=host \
      -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
      lmsysorg/sglang:latest \
      bash -c "\
        pip install --no-cache-dir protobuf sentencepiece --break-system-packages && \
        python3 -m sglang.launch_server \
          --model-path $MODEL \
          --host 0.0.0.0 \
          --port $PORT \
          --context-length 18000
        "
    ;;


  mii)
    # ────────────────────────────────────────────────────────────────────────
    # DeepSpeed-MII container:
    #
    # docker run --runtime=nvidia --gpus all \
    #   --name $NAME \
    #   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    #   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   slinusc/deepspeed-mii:latest \
    #   --model mistralai/Mistral-7B-Instruct-v0.3 \
    #   --port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      slinusc/deepspeed-mii:latest \
      --model "$MODEL" \
      --port "$PORT"
    ;;

  vllm-ocr2)
    docker run --rm \
      --runtime=nvidia --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 8192 \
        --port "$PORT" \
        --gpu-memory-utilization 0.9 \
        --no-enable-prefix-caching \
        --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
    ;;

  vllm-vl-test)
    docker run --rm \
      --name $NAME \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 32768 \
        --quantization fp8 \
        --port "$PORT" \
        --gpu-memory-utilization 0.96 \
        #--chat-template-content-format string─────────────────────────────────────────────────────

    ;;

    vllm-vl-test-noquant)
    docker run --rm \
      --name $NAME \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 17000 \
        --port "$PORT" \
        --gpu-memory-utilization 0.96 \
        #--chat-template-content-format string─────────────────────────────────────────────────────

    ;;


  vllm-vl)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM (OpenAI-compatible) container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   vllm/vllm-openai:latest \
    #     --model mistralai/Mistral-7B-Instruct-v0.3 \
    #     --port 23333

    #
    # ───────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      --name $NAME \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 8192 \
        --port "$PORT" \
        --gpu-memory-utilization 0.95 \
        --no-enable-prefix-caching \
        --limit-mm-per-prompt '{"image": 20}'
    ;;



  *)
    echo "Error: unsupported engine '$ENGINE'."
    echo "Please choose one of: tgi, vllm, vllm-vl, lmdeploy, sglang, mii."
    exit 1
    ;;
esac
