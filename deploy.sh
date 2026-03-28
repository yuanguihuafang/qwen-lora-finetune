#!/bin/bash
source ~/vllm-env/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/d/AI/models/Qwen2.5-1.5B-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.75 \
    --dtype bfloat16 \
    --enforce-eager
```