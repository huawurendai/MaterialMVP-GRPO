import argparse
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from textureGenPipeline import MaterialMVPConfig, MaterialMVPPipeline
from materialmvp.lora_utils import inject_lora_into_attention, normalize_lora_config

try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as exc:
    print(f"Warning: Failed to apply torchvision fix: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original MaterialMVP inference with a LoRA adapter.")
    parser.add_argument("--mesh", default="test_examples/mesh.glb", help="Input mesh path.")
    parser.add_argument("--image", default="test_examples/image.png", help="Input reference image path.")
    parser.add_argument("--cases", default=None, help="JSON list of fixed test cases.")
    parser.add_argument(
        "--lora",
        required=True,
        help="LoRA adapter checkpoint, for example logs/.../checkpoints/lora_last.pt.",
    )
    parser.add_argument("--lora-name", default=None, help="Name for LoRA output folder.")
    parser.add_argument("--out-dir", default="outputs/lora_compare", help="Directory for comparison outputs.")
    parser.add_argument("--resolution", type=int, default=512, help="Diffusion view resolution.")
    parser.add_argument("--max-num-view", type=int, default=6, help="Number of selected views for baking.")
    parser.add_argument("--seed", type=int, default=1234, help="Fixed inference seed.")
    parser.add_argument("--skip-base", action="store_true", help="Only run the LoRA model.")
    parser.add_argument("--skip-lora", action="store_true", help="Only run the original model.")
    parser.add_argument("--no-remesh", action="store_true", help="Disable remeshing before texture generation.")
    parser.add_argument("--no-glb", action="store_true", help="Do not export a GLB beside the OBJ.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cases(args):
    if args.cases is None:
        return [{"name": "single", "mesh": args.mesh, "image": args.image}]

    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for i, case in enumerate(cases):
        case.setdefault("name", f"case_{i + 1:03d}")
        if "mesh" not in case or "image" not in case:
            raise ValueError(f"Case {case['name']} must contain both 'mesh' and 'image'.")
    return cases


def load_adapter_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        lora_config = checkpoint.get("lora_config", {})
        state_dict = checkpoint["state_dict"]
    else:
        lora_config = {}
        state_dict = checkpoint
    lora_config = normalize_lora_config({**lora_config, "enabled": True, "print_trainable": False})
    return lora_config, state_dict


def apply_lora_adapter(pipe, lora_path):
    lora_config, state_dict = load_adapter_checkpoint(lora_path)
    unet = pipe.models["multiview_model"].pipeline.unet

    report = inject_lora_into_attention(
        unet,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
        target_suffixes=lora_config["target_suffixes"],
        include_keywords=lora_config["include_keywords"],
        exclude_keywords=lora_config["exclude_keywords"],
        extra_trainable_keywords=(),
        freeze_first=False,
    )
    incompatible = unet.load_state_dict(state_dict, strict=False)

    unexpected_adapter_keys = [
        key
        for key in incompatible.unexpected_keys
        if ".lora_down." in key or ".lora_up." in key or "learned_text_clip" in key
    ]
    if unexpected_adapter_keys:
        preview = ", ".join(unexpected_adapter_keys[:8])
        raise RuntimeError(f"LoRA adapter has keys that did not match this UNet: {preview}")

    print(
        "Loaded LoRA adapter: "
        f"path={lora_path}, modules={report.module_count}, "
        f"adapter_tensors={len(state_dict)}"
    )


def build_pipeline(max_num_view, resolution):
    config = MaterialMVPConfig(max_num_view=max_num_view, resolution=resolution)
    return MaterialMVPPipeline(config)


def run_one(name, args, case, use_lora):
    set_seed(args.seed)

    output_dir = Path(args.out_dir) / case["name"] / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_path = output_dir / "textured_mesh.obj"

    pipe = build_pipeline(args.max_num_view, args.resolution)
    if use_lora:
        apply_lora_adapter(pipe, args.lora)

    set_seed(args.seed)

    result = pipe(
        mesh_path=case["mesh"],
        image_path=case["image"],
        output_mesh_path=str(output_mesh_path),
        use_remesh=not args.no_remesh,
        save_glb=not args.no_glb,
    )

    if Path(case["image"]).exists():
        shutil.copy2(case["image"], output_dir / Path(case["image"]).name)
    manifest = {
        "name": name,
        "case": case["name"],
        "mesh": case["mesh"],
        "image": case["image"],
        "lora": args.lora if use_lora else None,
        "seed": args.seed,
        "resolution": args.resolution,
        "max_num_view": args.max_num_view,
        "output_mesh": result,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    del pipe
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cases = load_cases(args)
    lora_name = args.lora_name or Path(args.lora).stem
    summary = {}

    for case in cases:
        print(f"\nRunning {case['name']}")
        summary[case["name"]] = {}

        if not args.skip_base:
            summary[case["name"]]["base"] = run_one("base", args, case, use_lora=False)
        if not args.skip_lora:
            summary[case["name"]][lora_name] = run_one(lora_name, args, case, use_lora=True)

    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nComparison outputs:")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
