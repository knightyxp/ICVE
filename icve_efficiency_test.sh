#!/bin/bash
#
# ICVE Efficiency Test Script
# This script tests inference time, GPU memory, and RAM usage on VideoCOF benchmark
# Only uses the first item from the JSON file for benchmarking
#


# Configuration - matches the original test.sh settings
TEST_JSON="/scratch3/yan204/yxp/VideoX_Fun/data/test_json/20_test.json"
OUTPUT_DIR="./icve_efficiency_results"
SEED=42

# Model settings
DIT_WEIGHT="checkpoint/diffusion_pytorch_model.safetensors"
VIDEO_SIZE_H=400
VIDEO_SIZE_W=704
VIDEO_LENGTH=33
INFER_STEPS=50
EMBEDDED_CFG_SCALE=1.0
CFG_SCALE=6.0
FLOW_SHIFT=7.0

# Benchmark settings
WARMUP_RUNS=1
BENCHMARK_RUNS=3

echo "=============================================="
echo "ICVE Efficiency Test"
echo "=============================================="
echo "Test JSON: ${TEST_JSON}"
echo "Output dir: ${OUTPUT_DIR}"
echo "DIT weight: ${DIT_WEIGHT}"
echo "Video size: ${VIDEO_SIZE_H}x${VIDEO_SIZE_W}"
echo "Video length: ${VIDEO_LENGTH}"
echo "Infer steps: ${INFER_STEPS}"
echo "Warmup runs: ${WARMUP_RUNS}"
echo "Benchmark runs: ${BENCHMARK_RUNS}"
echo "=============================================="

export CUDA_VISIBLE_DEVICES=0
# Pass efficiency test settings via environment variables
export WARMUP_RUNS=${WARMUP_RUNS}
export BENCHMARK_RUNS=${BENCHMARK_RUNS}

python icve_efficiency_test.py \
    --dit-weight "${DIT_WEIGHT}" \
    --video-size ${VIDEO_SIZE_H} ${VIDEO_SIZE_W} \
    --video-length ${VIDEO_LENGTH} \
    --infer-steps ${INFER_STEPS} \
    --seed ${SEED} \
    --embedded-cfg-scale ${EMBEDDED_CFG_SCALE} \
    --cfg-scale ${CFG_SCALE} \
    --flow-shift ${FLOW_SHIFT} \
    --flow-reverse \
    --use-cpu-offload \
    --save-path "${OUTPUT_DIR}" \
    --video "${TEST_JSON}"

echo ""
echo "=============================================="
echo "ICVE Efficiency test complete!"
echo "Results saved to: ${OUTPUT_DIR}/efficiency_results.json"
echo "=============================================="

