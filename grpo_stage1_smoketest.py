import argparse
import json
import os
import random
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except Exception as exc:
    print(f"Warning: torchvision compatibility fix was not applied: {exc}")

from src.utils.train_util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 GRPO smoke test for original MaterialMVP. "
            "This script performs rollout + reward + group advantage only; it never updates model weights."
        )
    )
    parser.add_argument("--base", default="cfgs/v1_lora_test.yaml", help="Config used only for dataset loading.")
    parser.add_argument(
        "--dataset-split",
        default="validation",
        choices=["train", "validation"],
        help="Dataset split from the config.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default="tencent/Hunyuan3D-2.1",
        help=(
            "Local PaintPBR model folder or Hugging Face repo id. "
            "If this is a repo id, --pretrained-subdir is downloaded/loaded."
        ),
    )
    parser.add_argument("--pretrained-subdir", default="hunyuan3d-paintpbr-v2-1")
    parser.add_argument("--dino-ckpt-path", default="facebook/dinov2-giant")
    parser.add_argument("--out-dir", default="outputs/grpo_stage1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--use-dataset-augmentation",
        action="store_true",
        help=(
            "Use the dataset __getitem__ path exactly. By default this script loads raw condition/PBR/normal/position "
            "files without training augmentation, which is closer to normal inference."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def tensor_to_pil(image_chw):
    array = image_chw.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_rgb_pil(path, size, bg_color=(127, 127, 127)):
    image = Image.open(path)
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, bg_color)
        bg.paste(image, mask=image.getchannel("A"))
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((size, size))


def resize_view_tensor(views_nchw, size):
    return F.interpolate(views_nchw, size=(size, size), mode="bilinear", align_corners=False).clamp(0, 1)


def load_dataset_from_config(base_config, split):
    config = OmegaConf.load(base_config)
    split_cfgs = config.data.params.get(split)
    if not split_cfgs:
        raise ValueError(f"No data split '{split}' found in {base_config}")
    datasets = [instantiate_from_config(loader_cfg) for loader_cfg in split_cfgs]
    return ConcatDataset(datasets)


def get_sample_dir(dataset, index):
    if not isinstance(dataset, ConcatDataset):
        if hasattr(dataset, "data"):
            return dataset.data[index]
        raise ValueError("Dataset does not expose a .data list; use --use-dataset-augmentation.")

    offset = 0
    for sub_dataset, cumulative_size in zip(dataset.datasets, dataset.cumulative_sizes):
        if index < cumulative_size:
            local_index = index - offset
            if hasattr(sub_dataset, "data"):
                return sub_dataset.data[local_index]
            raise ValueError("Sub-dataset does not expose a .data list; use --use-dataset-augmentation.")
        offset = cumulative_size
    raise IndexError(index)


def get_num_view(dataset, default=6):
    if isinstance(dataset, ConcatDataset) and dataset.datasets:
        return int(getattr(dataset.datasets[0], "num_view", default))
    return int(getattr(dataset, "num_view", default))


def texture_albedo_paths(sample_dir, num_view):
    render_tex = Path(sample_dir) / "render_tex"
    transforms_path = render_tex / "transforms.json"
    paths = []

    if transforms_path.exists():
        with open(transforms_path, "r", encoding="utf-8") as f:
            transforms = json.load(f)
        frames = sorted(
            transforms.get("frames", []),
            key=lambda frame: (
                int(frame.get("elevation_index", 0)),
                int(frame.get("azimuth_index", len(paths))),
                frame.get("file_path", ""),
            ),
        )
        for frame in frames:
            stem = Path(frame["file_path"]).stem
            path = render_tex / f"{stem}_albedo.png"
            if path.exists():
                paths.append(str(path))

    if not paths:
        for ext in ["*_albedo.png", "*_albedo.jpg", "*_albedo.jpeg"]:
            paths.extend(glob.glob(str(render_tex / ext)))
        paths = sorted(paths)

    if len(paths) < num_view:
        raise ValueError(f"Only {len(paths)} texture views found in {render_tex}, need {num_view}.")
    return paths[:num_view]


def condition_image_path(sample_dir):
    render_cond = Path(sample_dir) / "render_cond"
    transforms_path = render_cond / "transforms.json"
    candidates = []

    if transforms_path.exists():
        with open(transforms_path, "r", encoding="utf-8") as f:
            transforms = json.load(f)
        for frame in transforms.get("frames", []):
            path = render_cond / frame["file_path"]
            if path.exists():
                candidates.append(
                    {
                        "path": str(path),
                        "lighting_type": frame.get("lighting_type", ""),
                        "azimuth_index": int(frame.get("azimuth_index", 0)),
                    }
                )

    if candidates:
        point_lights = [item for item in candidates if item["lighting_type"] == "PL" or "_light_PL" in item["path"]]
        selected = sorted(point_lights or candidates, key=lambda item: item["azimuth_index"])[0]
        return selected["path"]

    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob.glob(str(render_cond / ext)))
    if not files:
        raise ValueError(f"No condition image found in {render_cond}.")
    point_lights = [path for path in files if "_light_PL" in Path(path).name]
    return sorted(point_lights or files)[0]


def build_clean_sample(dataset, index, resolution, num_view):
    sample_dir = get_sample_dir(dataset, index)
    albedo_paths = texture_albedo_paths(sample_dir, num_view)
    cond_path = condition_image_path(sample_dir)

    cond_pil = load_rgb_pil(cond_path, resolution)
    normal_pils = [load_rgb_pil(path.replace("_albedo", "_normal"), resolution) for path in albedo_paths]
    position_pils = [load_rgb_pil(path.replace("_albedo", "_pos"), resolution) for path in albedo_paths]

    target_albedo = torch.stack([pil_to_tensor(load_rgb_pil(path, resolution)) for path in albedo_paths], dim=0)
    target_mr = torch.stack(
        [pil_to_tensor(load_rgb_pil(path.replace("_albedo", "_mr"), resolution)) for path in albedo_paths],
        dim=0,
    )

    return {
        "name": sample_dir,
        "condition_path": cond_path,
        "albedo_paths": albedo_paths,
        "cond_pil": cond_pil,
        "normal_pils": normal_pils,
        "position_pils": position_pils,
        "target_albedo": target_albedo,
        "target_mr": target_mr,
    }


def resolve_model_path(model_path, subdir):
    local_path = Path(model_path)
    if local_path.exists():
        return str(local_path / subdir) if subdir and (local_path / subdir).exists() else str(local_path)

    import huggingface_hub

    snapshot = huggingface_hub.snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return str(Path(snapshot) / subdir) if subdir else snapshot


def load_original_mvp_pipeline(args):
    dtype = resolve_dtype(args.dtype)
    model_path = resolve_model_path(args.pretrained_model_path, args.pretrained_subdir)
    pipe = DiffusionPipeline.from_pretrained(model_path, custom_pipeline="materialmvp", torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()
    pipe.to(args.device)

    dino = None
    if getattr(pipe.unet, "use_dino", False):
        from materialmvp.modules import Dino_v2

        dino = Dino_v2(args.dino_ckpt_path).to(device=args.device, dtype=dtype)
        dino.eval().requires_grad_(False)

    return pipe, dino


def build_condition(batch, resolution):
    cond_tensor = batch["images_cond"][0, 0]
    cond_pil = tensor_to_pil(cond_tensor).resize((resolution, resolution))

    normal_pils = [tensor_to_pil(view).resize((resolution, resolution)) for view in batch["images_normal"][0]]
    position_pils = [tensor_to_pil(view).resize((resolution, resolution)) for view in batch["images_position"][0]]

    return cond_pil, normal_pils, position_pils


@torch.no_grad()
def rollout_one(pipe, dino, cond_pil, normal_pils, position_pils, args, seed):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    kwargs = {
        "generator": generator,
        "width": args.resolution,
        "height": args.resolution,
        "num_in_batch": len(normal_pils),
        "images_normal": [normal_pils],
        "images_position": [position_pils],
    }

    if dino is not None:
        kwargs["dino_hidden_states"] = dino(cond_pil)

    output = pipe(
        [cond_pil],
        prompt="high quality",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        **kwargs,
    ).images

    num_view = len(normal_pils)
    albedo = output[:num_view]
    mr = output[num_view : 2 * num_view]
    return {"albedo": albedo, "mr": mr}


def stack_generated_views(pils, device, resolution):
    tensors = torch.stack([pil_to_tensor(pil.resize((resolution, resolution))) for pil in pils], dim=0)
    return tensors.to(device)


def compute_reward(generated, target, device, resolution):
    gen_albedo = stack_generated_views(generated["albedo"], device, resolution)
    gen_mr = stack_generated_views(generated["mr"], device, resolution)

    target_albedo = resize_view_tensor(target["target_albedo"].to(device), resolution)
    target_mr = resize_view_tensor(target["target_mr"].to(device), resolution)

    albedo_l1 = F.l1_loss(gen_albedo, target_albedo).detach()
    mr_l1 = F.l1_loss(gen_mr, target_mr).detach()

    albedo_saturation = ((gen_albedo < 0.02) | (gen_albedo > 0.98)).float().mean()
    mr_saturation = ((gen_mr < 0.02) | (gen_mr > 0.98)).float().mean()
    range_penalty = 0.5 * (albedo_saturation + mr_saturation)

    reward = -(0.6 * albedo_l1 + 0.4 * mr_l1) - 0.05 * range_penalty

    return {
        "reward": float(reward.cpu()),
        "albedo_l1": float(albedo_l1.cpu()),
        "mr_l1": float(mr_l1.cpu()),
        "range_penalty": float(range_penalty.cpu()),
    }


def save_contact_sheet(generated, path):
    rows = [generated["albedo"], generated["mr"]]
    cell_w, cell_h = rows[0][0].size
    sheet = Image.new("RGB", (cell_w * len(rows[0]), cell_h * len(rows)), (127, 127, 127))
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            sheet.paste(image.convert("RGB"), (col_idx * cell_w, row_idx * cell_h))
    sheet.save(path)


def save_target_sheet(target, path):
    rows = [
        [tensor_to_pil(view) for view in target["target_albedo"]],
        [tensor_to_pil(view) for view in target["target_mr"]],
    ]
    cell_w, cell_h = rows[0][0].size
    sheet = Image.new("RGB", (cell_w * len(rows[0]), cell_h * len(rows)), (127, 127, 127))
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            sheet.paste(image.convert("RGB"), (col_idx * cell_w, row_idx * cell_h))
    sheet.save(path)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.group_size < 1:
        raise ValueError("--group-size must be >= 1")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_from_config(args.base, args.dataset_split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) if args.use_dataset_augmentation else None
    pipe, dino = load_original_mvp_pipeline(args)

    all_records = []
    for sample_idx in range(min(args.max_samples, len(dataset))):
        if sample_idx >= args.max_samples:
            break

        sample_dir = out_dir / f"sample_{sample_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.use_dataset_augmentation:
            batch = next(iter(loader)) if sample_idx == 0 else dataset[sample_idx]
            if not isinstance(batch, dict):
                raise ValueError("Unexpected dataset batch format.")
            if batch["images_cond"].ndim == 4:
                batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in batch.items()}
            cond_pil, normal_pils, position_pils = build_condition(batch, args.resolution)
            target = {
                "target_albedo": batch["images_albedo"][0],
                "target_mr": batch["images_mr"][0],
            }
            name = batch.get("name", [""])[0]
        else:
            clean_sample = build_clean_sample(dataset, sample_idx, args.resolution, get_num_view(dataset))
            cond_pil = clean_sample["cond_pil"]
            normal_pils = clean_sample["normal_pils"]
            position_pils = clean_sample["position_pils"]
            target = clean_sample
            name = clean_sample["name"]

        cond_pil.save(sample_dir / "condition.png")
        save_target_sheet(target, sample_dir / "target_albedo_mr.png")

        group_records = []
        for group_idx in range(args.group_size):
            seed = args.seed + sample_idx * 1000 + group_idx
            generated = rollout_one(pipe, dino, cond_pil, normal_pils, position_pils, args, seed)
            metrics = compute_reward(generated, target, pipe.device, args.resolution)
            metrics.update({"group_idx": group_idx, "seed": seed})

            save_contact_sheet(generated, sample_dir / f"group_{group_idx:02d}_albedo_mr.png")
            group_records.append(metrics)

            print(
                f"sample={sample_idx:04d} group={group_idx:02d} "
                f"reward={metrics['reward']:.6f} "
                f"albedo_l1={metrics['albedo_l1']:.6f} mr_l1={metrics['mr_l1']:.6f}"
            )

        rewards = torch.tensor([record["reward"] for record in group_records], dtype=torch.float32)
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        for record, advantage in zip(group_records, advantages):
            record["advantage"] = float(advantage)

        sample_record = {
            "sample_idx": sample_idx,
            "name": name,
            "uses_dataset_augmentation": args.use_dataset_augmentation,
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "groups": group_records,
        }
        all_records.append(sample_record)
        (sample_dir / "stage1_metrics.json").write_text(
            json.dumps(sample_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"sample={sample_idx:04d} group_reward_mean={reward_mean:.6f} "
            f"group_reward_std={reward_std:.6f}"
        )

    summary = {
        "stage": "grpo_stage1_rollout_reward_smoketest",
        "updates_model": False,
        "base_config": args.base,
        "pretrained_model_path": args.pretrained_model_path,
        "pretrained_subdir": args.pretrained_subdir,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "group_size": args.group_size,
        "records": all_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stage-1 summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
