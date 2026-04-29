#!/bin/bash
WEIGHTS=/mnt/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16

# The image does not include audio packages so we need to install them with "pip install vllm[audio]" as done in the command below
docker run -d \
  --gpus all \
  --ipc=host -p 8000:8000 \
  --shm-size=16g \
  --name vllm-nemotron-omni \
  -v "${WEIGHTS}:/model:ro" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.0 -c  \
  "pip install vllm[audio] && vllm serve /model \
  --served-model-name=nemotron_3_nano_omni \
  --max-num-seqs 8 \
  --max-model-len 131072 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization .9 \
  --limit-mm-per-prompt '{\"video\": 1, \"image\": 1, \"audio\": 1}' \
  --media-io-kwargs '{\"video\": {\"fps\": 2,  \"num_frames\": 256}}' \
  --allowed-local-media-path=/ \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder"
