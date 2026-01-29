#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICVE Efficiency Test Script

This script measures inference time, GPU memory, and RAM usage for ICVE video editing.
It only processes the first item from the JSON file for benchmarking purposes.

Metrics measured:
- Inference time (seconds)
- Peak GPU memory usage (GB)
- RAM usage before/after (GB)
"""
import os
import sys
import json
import argparse
import time
import gc
import psutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import imageio
from loguru import logger

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def get_gpu_memory_peak():
    """Get peak GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def get_gpu_memory_reserved():
    """Get reserved GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 3)
    return 0.0


def get_ram_usage():
    """Get current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def reset_gpu_memory_stats():
    """Reset GPU memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def _sanitize_filename(name: str) -> str:
    s = (name or "").strip().replace("/", "_").replace("\\", "_")
    s = s.replace(" ", "_")
    return s[:120]


def _standardize_items(raw):
    items = []
    if isinstance(raw, list):
        iterable = raw
    elif isinstance(raw, dict):
        iterable = raw.get("items", list(raw.values()))
    else:
        raise ValueError("JSON must be a list or a dict with items")

    for it in iterable:
        if not isinstance(it, dict):
            continue
        video = (
            it.get("video")
            or it.get("source_video_path")
            or it.get("video_path")
            or it.get("path")
        )
        prompt = (
            it.get("qwen_vl_72b_refined_instruction")
            or it.get("edit_instruction")
            or it.get("text")
            or it.get("prompt")
        )
        if not video or not prompt:
            continue
        items.append({
            "video": video,
            "prompt": prompt,
            "task_type": it.get("task_type"),
            "sample_id": it.get("sample_id"),
            "height": it.get("height"),
            "width": it.get("width"),
            "video_length": it.get("video_length"),
            "seed": it.get("seed"),
            "neg_prompt": it.get("neg_prompt"),
            "infer_steps": it.get("infer_steps"),
            "cfg_scale": it.get("cfg_scale"),
            "num_videos": it.get("num_videos"),
            "flow_shift": it.get("flow_shift"),
            "batch_size": it.get("batch_size"),
            "embedded_cfg_scale": it.get("embedded_cfg_scale"),
            "id": it.get("id") or it.get("sample_id") or it.get("name"),
        })
    return items


def _make_base_from_item(item: dict) -> str:
    task_type = item.get("task_type")
    sample_id = item.get("sample_id")
    if task_type and sample_id:
        return f"{task_type}_{sample_id}"
    if item.get("id"):
        return _sanitize_filename(str(item["id"]))
    base_name = os.path.splitext(os.path.basename(item.get("video", "video")))[0]
    return _sanitize_filename(base_name)


def _normalize_to_01(video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vmin = float(video.min())
        vmax = float(video.max())
        if vmin < 0.0 or vmax > 1.0:
            video = (video + 1.0) / 2.0
        return video.clamp(0.0, 1.0)


def _load_input_video_frames(video_path: str, target_frames: int) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = sum(1 for _ in reader)
        reader = imageio.get_reader(video_path)
    
    stride = max(1, total_frames // target_frames)
    # Use fixed start frame for reproducibility
    start_frame = 0
    
    frames = []
    original_height, original_width = None, None
    
    for i in range(target_frames):
        idx = start_frame + i * stride
        if idx >= total_frames:
            break
        try:
            frame = reader.get_data(idx)
            if original_height is None:
                original_height, original_width = frame.shape[0], frame.shape[1]
            frames.append(frame)
        except IndexError:
            break
    
    reader.close()
    
    arr = np.array(frames)  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float()  # [1,C,T,H,W]
    tensor = tensor * (2.0 / 255.0) - 1.0  # [-1,1]
    return tensor


def _save_input_video(tensor: torch.Tensor, file_path: str, fps: int = 8):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tensor01 = _normalize_to_01(tensor.detach().cpu())
    save_videos_grid(tensor01, file_path, fps=fps)


def _save_side_by_side(input_tensor: torch.Tensor, sample_tensor: torch.Tensor, file_path: str, fps: int = 8):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    a = _normalize_to_01(input_tensor.detach().cpu())
    b = _normalize_to_01(sample_tensor.detach().cpu())
    T = min(a.shape[2], b.shape[2])
    H = min(a.shape[3], b.shape[3])
    W = min(a.shape[4], b.shape[4])
    a = a[:, :, :T, :H, :W]
    b = b[:, :, :T, :H, :W]
    combined = torch.cat([a, b], dim=4)
    save_videos_grid(combined, file_path, fps=fps)


def _get_video_hw(video_path: str):
    reader = imageio.get_reader(video_path)
    try:
        frame = reader.get_data(0)
        h, w = frame.shape[0], frame.shape[1]
        return h, w
    finally:
        reader.close()


def _coalesce_none(value, fallback):
    return fallback if value is None else value


def run_inference(sampler, base_args, item, auto_hw=True):
    """Run a single inference and return the result"""
    prompt = item["prompt"]
    video_path = item["video"]

    if auto_hw:
        try:
            vh, vw = _get_video_hw(video_path)
            height = item.get("height") or vh
            width = item.get("width") or vw
        except Exception:
            height = item.get("height") or base_args.video_size[0]
            width = item.get("width") or base_args.video_size[1]
    else:
        height = item.get("height") or base_args.video_size[0]
        width = item.get("width") or base_args.video_size[1]
    
    video_length = item.get("video_length") or base_args.video_length
    seed = _coalesce_none(item.get("seed"), base_args.seed)
    negative_prompt = _coalesce_none(item.get("neg_prompt"), base_args.neg_prompt)
    infer_steps = _coalesce_none(item.get("infer_steps"), base_args.infer_steps)
    guidance_scale = _coalesce_none(item.get("cfg_scale"), base_args.cfg_scale)
    num_videos = _coalesce_none(item.get("num_videos"), base_args.num_videos)
    flow_shift = _coalesce_none(item.get("flow_shift"), base_args.flow_shift)
    batch_size = _coalesce_none(item.get("batch_size"), base_args.batch_size)
    embedded_cfg_scale = _coalesce_none(item.get("embedded_cfg_scale"), base_args.embedded_cfg_scale)

    outputs = sampler.predict(
        prompt=prompt,
        video=video_path,
        height=height,
        width=width,
        video_length=video_length,
        seed=seed,
        negative_prompt=negative_prompt,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=num_videos,
        flow_shift=flow_shift,
        batch_size=batch_size,
        embedded_guidance_scale=embedded_cfg_scale,
    )

    return outputs, height, width, video_length


def main():
    # Parse base args from command line
    args = parse_args()
    
    # Additional efficiency test args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of benchmark runs")
    eff_args, _ = parser.parse_known_args()
    
    warmup_runs = eff_args.warmup_runs
    benchmark_runs = eff_args.benchmark_runs
    
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)

    # Check if video argument is a JSON file
    video_arg = getattr(args, "video", None)
    if not (isinstance(video_arg, str) and video_arg.lower().endswith(".json") and os.path.isfile(video_arg)):
        raise ValueError("--video must be a JSON file for efficiency testing")

    print("=" * 60)
    print("ICVE Efficiency Test")
    print("=" * 60)
    print(f"Model base: {models_root_path}")
    print(f"DIT weight: {args.dit_weight}")
    print(f"Video size: {args.video_size}")
    print(f"Video length: {args.video_length}")
    print(f"Infer steps: {args.infer_steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Embedded CFG scale: {args.embedded_cfg_scale}")
    print(f"Flow shift: {args.flow_shift}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")
    print("=" * 60)

    # Load JSON and get only the first item
    logger.info(f"Loading tasks from JSON: {video_arg}")
    with open(video_arg, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = _standardize_items(raw)
    
    if len(items) == 0:
        raise ValueError("No valid items found in JSON. Each item requires 'video' and 'prompt'.")
    
    # Take only the first item
    item = items[0]
    
    print(f"\nTest sample:")
    print(f"  Task type: {item.get('task_type', 'N/A')}")
    print(f"  Sample ID: {item.get('sample_id', 'N/A')}")
    print(f"  Video path: {item.get('video', 'N/A')}")
    print(f"  Prompt: {item.get('prompt', 'N/A')[:100]}...")

    # Record initial RAM usage
    ram_before_model_load = get_ram_usage()
    print(f"\nRAM before model load: {ram_before_model_load:.2f} GB")

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
    
    logger.info(f"Loading models on {device}...")
    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args, device=device)
    args = sampler.args
    
    ram_after_model_load = get_ram_usage()
    print(f"\nRAM after model load: {ram_after_model_load:.2f} GB")
    print(f"Model RAM usage: {ram_after_model_load - ram_before_model_load:.2f} GB")

    # Get actual video dimensions for reporting
    try:
        vh, vw = _get_video_hw(item["video"])
        actual_height = item.get("height") or vh
        actual_width = item.get("width") or vw
    except Exception:
        actual_height = args.video_size[0]
        actual_width = args.video_size[1]
    actual_video_length = item.get("video_length") or args.video_length

    print(f"\nActual inference dimensions: {actual_width}x{actual_height}, {actual_video_length} frames")

    # Warmup runs
    print(f"\n{'=' * 60}")
    print(f"Running {warmup_runs} warmup run(s)...")
    print(f"{'=' * 60}")
    
    for i in range(warmup_runs):
        reset_gpu_memory_stats()
        _ = run_inference(sampler, args, item, auto_hw=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print(f"Warmup {i+1}/{warmup_runs} complete")

    # Benchmark runs
    print(f"\n{'=' * 60}")
    print(f"Running {benchmark_runs} benchmark run(s)...")
    print(f"{'=' * 60}")

    inference_times = []
    peak_gpu_memories = []
    ram_usages = []

    for i in range(benchmark_runs):
        # Reset stats
        reset_gpu_memory_stats()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        ram_before = get_ram_usage()
        
        # Time the inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        outputs, height, width, video_length = run_inference(sampler, args, item, auto_hw=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        inference_time = end_time - start_time
        peak_gpu_memory = get_gpu_memory_peak()
        ram_after = get_ram_usage()
        
        inference_times.append(inference_time)
        peak_gpu_memories.append(peak_gpu_memory)
        ram_usages.append(ram_after)
        
        print(f"\nRun {i+1}/{benchmark_runs}:")
        print(f"  Inference time: {inference_time:.2f} seconds")
        print(f"  Peak GPU memory: {peak_gpu_memory:.2f} GB")
        print(f"  RAM usage: {ram_after:.2f} GB")

    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    avg_peak_gpu_memory = np.mean(peak_gpu_memories)
    avg_ram_usage = np.mean(ram_usages)

    # Print summary
    print(f"\n{'=' * 60}")
    print("ICVE EFFICIENCY TEST RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Test Configuration:")
    print(f"  - Model base: {models_root_path}")
    print(f"  - DIT weight: {args.dit_weight}")
    print(f"  - Video resolution: {actual_width}x{actual_height}")
    print(f"  - Video length: {actual_video_length} frames")
    print(f"  - Inference steps: {args.infer_steps}")
    print(f"  - CFG scale: {args.cfg_scale}")
    print(f"  - Embedded CFG scale: {args.embedded_cfg_scale}")
    print(f"  - Flow shift: {args.flow_shift}")
    print(f"  - Use CPU offload: {args.use_cpu_offload}")
    print(f"\nResults (averaged over {benchmark_runs} runs):")
    print(f"  - Inference Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} seconds")
    print(f"  - Peak GPU Memory: {avg_peak_gpu_memory:.2f} GB")
    print(f"  - RAM Usage: {avg_ram_usage:.2f} GB")
    print(f"  - Model RAM Usage: {ram_after_model_load - ram_before_model_load:.2f} GB")
    print(f"{'=' * 60}")

    # Save the last generated video for verification
    base = _make_base_from_item(item)
    samples = outputs["samples"]
    
    if len(samples) > 0:
        output_video_path = os.path.join(save_path, f"gen_{base}.mp4")
        input_video_path = os.path.join(save_path, f"gen_{base}_input.mp4")
        compare_video_path = os.path.join(save_path, f"gen_{base}_compare.mp4")
        
        sample_tensor = samples[0].unsqueeze(0)
        save_videos_grid(sample_tensor, output_video_path, fps=8)
        print(f"\nSaved test video: {output_video_path}")
        
        # Save input and comparison
        try:
            input_tensor = _load_input_video_frames(item["video"], actual_video_length)
            _save_input_video(input_tensor, input_video_path, fps=8)
            _save_side_by_side(input_tensor, sample_tensor, compare_video_path, fps=8)
            print(f"Saved comparison video: {compare_video_path}")
        except Exception as e:
            logger.warning(f"Failed to save input/comparison video: {e}")

    # Save results to JSON
    results = {
        "config": {
            "model_base": str(models_root_path),
            "dit_weight": str(args.dit_weight),
            "video_resolution": f"{actual_width}x{actual_height}",
            "video_length": actual_video_length,
            "infer_steps": args.infer_steps,
            "cfg_scale": args.cfg_scale,
            "embedded_cfg_scale": args.embedded_cfg_scale,
            "flow_shift": args.flow_shift,
            "use_cpu_offload": args.use_cpu_offload,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
        },
        "results": {
            "inference_time_avg_sec": float(avg_inference_time),
            "inference_time_std_sec": float(std_inference_time),
            "inference_times_sec": [float(t) for t in inference_times],
            "peak_gpu_memory_avg_gb": float(avg_peak_gpu_memory),
            "peak_gpu_memories_gb": [float(m) for m in peak_gpu_memories],
            "ram_usage_avg_gb": float(avg_ram_usage),
            "ram_usages_gb": [float(r) for r in ram_usages],
            "model_ram_usage_gb": float(ram_after_model_load - ram_before_model_load),
        },
        "test_sample": {
            "task_type": item.get("task_type", "N/A"),
            "sample_id": item.get("sample_id", "N/A"),
            "video_path": item.get("video", "N/A"),
            "prompt": item.get("prompt", "N/A"),
        }
    }

    results_path = os.path.join(save_path, "efficiency_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {results_path}")

    print("\nICVE Efficiency test complete!")


if __name__ == "__main__":
    main()

