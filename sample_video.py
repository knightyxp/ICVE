import os
import time
import json
import multiprocessing as mp
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import torch
import imageio
import numpy as np


def _sanitize_filename(name: str) -> str:
    s = (name or "").strip().replace("/", "_").replace("\\", "_")
    s = s.replace(" ", "_")
    return s[:120]


def _standardize_items(raw):
    items = []
    # Accept list, dict with 'items', or dict mapping ids->item
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
        # Prefer refined instruction first, then edit_instruction, then text/prompt
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
    # Fallbacks
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
    frames = []
    try:
        for i in range(target_frames):
            try:
                frame = reader.get_data(i)
            except IndexError:
                frame = frames[-1] if frames else np.zeros((480, 832, 3), dtype=np.uint8)
            frames.append(frame)
    finally:
        reader.close()
    arr = np.array(frames)  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float()  # [1,C,T,H,W]
    tensor = tensor * (2.0 / 255.0) - 1.0  # [-1,1]
    return tensor


def _save_input_video(tensor: torch.Tensor, file_path: str, fps: int = 24):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tensor01 = _normalize_to_01(tensor.detach().cpu())
    save_videos_grid(tensor01, file_path, fps=fps)


def _save_side_by_side(input_tensor: torch.Tensor, sample_tensor: torch.Tensor, file_path: str, fps: int = 24):
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


def _process_one(sampler: HunyuanVideoSampler, base_args, item: dict, save_path: str, auto_hw: bool = False):
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

    base = _make_base_from_item(item)
    output_video_path = os.path.join(save_path, f"gen_{base}.mp4")
    input_video_path = os.path.join(save_path, f"gen_{base}_input.mp4")
    compare_video_path = os.path.join(save_path, f"gen_{base}_compare.mp4")
    info_path = os.path.join(save_path, f"gen_{base}_info.txt")
    if os.path.exists(output_video_path):
        logger.info(f"Skip existing: {output_video_path}")
        return

    # Prepare input preview tensor
    try:
        input_tensor = _load_input_video_frames(video_path, video_length)
    except Exception as e:
        logger.warning(f"Failed to load input preview for {video_path}: {e}")
        input_tensor = None

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

    samples = outputs["samples"]
    # Use only the first sample for naming consistency
    if len(samples) == 0:
        logger.warning(f"No sample generated for {base}")
        return
    sample_tensor = samples[0].unsqueeze(0)
    save_videos_grid(sample_tensor, output_video_path, fps=24)
    logger.info(f"Saved video → {output_video_path}")
    # Save input and side-by-side if available
    if input_tensor is not None:
        _save_input_video(input_tensor, input_video_path, fps=24)
        _save_side_by_side(input_tensor, sample_tensor, compare_video_path, fps=24)
    # Save prompt info
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(str(prompt))
    except Exception as e:
        logger.warning(f"Failed writing info file {info_path}: {e}")


def _worker(worker_id: int, device: str, models_root_path_str: str, args_dict: dict, items_chunk: list, save_path: str, auto_hw: bool):
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            device_index = int(device.split(":")[1]) if ":" in device else 0
            torch.cuda.set_device(device_index)
        models_root_path = Path(models_root_path_str)
        # Rebuild args Namespace
        from argparse import Namespace
        worker_args = Namespace(**args_dict)
        logger.info(f"[Worker {worker_id}] Loading models on {device} with {len(items_chunk)} items...")
        sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=worker_args, device=device)
        # Use potential updated args from sampler
        worker_args = sampler.args
        for idx, item in enumerate(items_chunk):
            try:
                logger.info(f"[Worker {worker_id}] ({idx+1}/{len(items_chunk)}) {item.get('video')}")
                _process_one(sampler, worker_args, item, save_path, auto_hw=auto_hw)
            except Exception as e:
                logger.exception(f"[Worker {worker_id}] Failed on item {item}: {e}")
        logger.info(f"[Worker {worker_id}] Done.")
    except Exception as e:
        logger.exception(f"[Worker {worker_id}] Fatal error: {e}")


def _chunk_list(seq, n):
    if n <= 0:
        return [seq]
    k = min(n, max(1, len(seq)))
    avg = (len(seq) + k - 1) // k
    return [seq[i:i + avg] for i in range(0, len(seq), avg)]


def _split_items(items, split_index: int, split_total: int, round_robin: bool = False):
    if split_total <= 1:
        return items
    split_index = max(0, min(split_index, split_total - 1))
    if round_robin:
        return [it for i, it in enumerate(items) if i % split_total == split_index]
    # contiguous chunk split
    items_per = len(items) // split_total
    start = split_index * items_per
    end = (split_index + 1) * items_per if split_index != split_total - 1 else len(items)
    return items[start:end]


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # If args.video is a JSON file, run parallel multi-item inference
    video_arg = getattr(args, "video", None)
    if isinstance(video_arg, str) and video_arg.lower().endswith(".json") and os.path.isfile(video_arg):
        logger.info(f"Loading tasks from JSON: {video_arg}")
        with open(video_arg, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = _standardize_items(raw)
        if len(items) == 0:
            raise ValueError("No valid items found in JSON. Each item requires 'video' and 'prompt'.")

        # Filter already generated outputs
        pending = []
        for it in items:
            base = _make_base_from_item(it)
            gen_path = os.path.join(save_path, f"gen_{base}.mp4")
            if not os.path.exists(gen_path):
                pending.append(it)
        items = pending
        logger.info(f"Pending items: {len(items)}")

        # Optional manual sharding via environment variables (single process or multi-process均可用)
        split_total = int(os.environ.get("SPLIT_TOTAL", "1") or "1")
        split_index = int(os.environ.get("SPLIT_INDEX", "0") or "0")
        rr = os.environ.get("ROUND_ROBIN_SPLIT", "0") in ("1", "true", "True")
        if split_total > 1:
            before = len(items)
            items = _split_items(items, split_index, split_total, round_robin=rr)
            logger.info(f"Shard applied: index {split_index}/{split_total}, took {len(items)}/{before} items")

        # Single-process mode: avoid multiprocessing entirely
        single_process = os.environ.get("SINGLE_PROCESS", "0") in ("1", "true", "True")
        if single_process:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Running SINGLE_PROCESS mode on {device} with {len(items)} items...")
            from argparse import Namespace
            worker_args = Namespace(**vars(args))
            try:
                if device.startswith("cuda"):
                    torch.cuda.set_device(int(device.split(":")[1]))
            except Exception:
                pass
            sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=worker_args, device=device)
            worker_args = sampler.args
            for idx, item in enumerate(items):
                try:
                    logger.info(f"[Single] ({idx+1}/{len(items)}) {item.get('video')}")
                    _process_one(sampler, worker_args, item, save_path, auto_hw=True)
                except Exception as e:
                    logger.exception(f"[Single] Failed on item {item}: {e}")
            logger.info("Single-process run finished.")
            return

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(device_count)]
        else:
            devices = ["cpu"]

        # Limit workers to number of items
        num_workers = min(len(devices), len(items))
        devices = devices[:num_workers]
        chunks = _chunk_list(items, num_workers)

        logger.info(f"Starting parallel inference with {num_workers} workers over {len(items)} items")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        base_args_dict = vars(args)
        procs = []
        for wid, device in enumerate(devices):
            chunk = chunks[wid] if wid < len(chunks) else []
            if not chunk:
                continue
            p = mp.Process(
                target=_worker,
                args=(wid, device, str(models_root_path), base_args_dict, chunk, save_path, True),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        logger.info("All workers finished.")
        return

    # Single-item fallback (original behavior)
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        video=args.video,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )
    samples = outputs["samples"]
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, _ in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f"Sample save to: {cur_save_path}")

if __name__ == "__main__":
    main()
