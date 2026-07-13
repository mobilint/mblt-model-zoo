#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/test.py" \
    --images \
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640" \
    --prompt "Compare these images. What are the differences?" \
    --text-mxq-path "/path/to/Qwen_Qwen3-VL-8B-Instruct_decoder_all.mxq"
