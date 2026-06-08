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

from materialmvp.lora_utils import inject_lora_into_attention, normalize_lora_config
from textureGenPipeline import MaterialMVPConfig, MaterialMVPPipeline

try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as exc:
    print(f"Warning: Failed to apply torchvision fix: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate textured meshes for MaterialMVP evaluation. "
            "Supports original MVP plus any number of LoRA/RL-LoRA checkpoints."
        )
    )
    parser.add_argument("--cases", required=True, help="JSON list of {name, mesh, reference/image}.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        help="LoRA method in the form name=/path/to/lora_last.pt. Can be provided multiple times.",
    )
    parser.add_argument("--no-remesh", action="store_true", help="Disable remeshing before texture generation.")
    parser.add_argument("--no-glb", action="store_true", help="Do not export GLB beside OBJ.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if manifest.json already exists.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cases(path):
    cases_path = Path(path)
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("--cases must be a JSON list.")

    normalized = []
    for idx, case in enumerate(cases):
        if "mesh" not in case:
            raise ValueError(f"Case {idx} is missing 'mesh'.")
        image = case.get("reference", case.get("image"))
        if image is None:
            raise ValueError(f"Case {idx} must contain 'reference' or 'image'.")

        root = cases_path.parent
        mesh_path = Path(case["mesh"])
        image_path = Path(image)
        if not mesh_path.is_absolute():
            mesh_path = root / mesh_path
        if not image_path.is_absolute():
            image_path = root / image_path
        normalized.append(
            {
                "name": case.get("name", f"case_{idx + 1:04d}"),
                "mesh": str(mesh_path),
                "image": str(image_path),
                "raw": case,
            }
        )
    return normalized


def parse_lora_methods(items):
    methods = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--lora must use name=/path/to/lora.pt format, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Empty LoRA method name in: {item}")
        if not path:
            raise ValueError(f"Empty LoRA path in: {item}")
        methods.append({"name": name, "lora": path})
    return methods


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
        f"path={lora_path}, modules={report.module_count}, adapter_tensors={len(state_dict)}"
    )


def build_pipeline(args):
    config = MaterialMVPConfig(max_num_view=args.max_num_view, resolution=args.resolution)
    return MaterialMVPPipeline(config)


def copy_reference(case, case_out_dir):
    src = Path(case["image"])
    if src.exists():
        shutil.copy2(src, case_out_dir / "reference.png")


def method_manifest_path(case, method, args):
    return Path(args.out_dir) / case["name"] / method["name"] / "manifest.json"


def needs_generation(case, method, args):
    return args.overwrite or not method_manifest_path(case, method, args).exists()


def generate_one(case, method, pipe, args):
    case_out_dir = Path(args.out_dir) / case["name"]
    method_out_dir = case_out_dir / method["name"]
    manifest_path = method_out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"Skip existing: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    method_out_dir.mkdir(parents=True, exist_ok=True)
    copy_reference(case, case_out_dir)
    set_seed(args.seed)

    output_obj = method_out_dir / "textured_mesh.obj"
    result = pipe(
        mesh_path=case["mesh"],
        image_path=case["image"],
        output_mesh_path=str(output_obj),
        use_remesh=not args.no_remesh,
        save_glb=not args.no_glb,
    )

    output_glb = str(output_obj).replace(".obj", ".glb")
    manifest = {
        "case": case["name"],
        "method": method["name"],
        "mesh": case["mesh"],
        "reference": case["image"],
        "lora": method.get("lora"),
        "seed": args.seed,
        "resolution": args.resolution,
        "max_num_view": args.max_num_view,
        "output_obj": str(result),
        "output_glb": output_glb if Path(output_glb).exists() else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    args = parse_args()
    cases = load_cases(args.cases)
    methods = []
    if not args.skip_base:
        methods.append({"name": "base", "lora": None})
    methods.extend(parse_lora_methods(args.lora))
    if not methods:
        raise ValueError("No methods to run. Remove --skip-base or provide at least one --lora.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_records = {
        case["name"]: {"name": case["name"], "mesh": case["mesh"], "reference": case["image"], "methods": {}}
        for case in cases
    }
    summary = {"cases": list(case_records.values()), "methods": [method["name"] for method in methods], "outputs": []}

    for method in methods:
        pending_count = sum(needs_generation(case, method, args) for case in cases)
        print(f"\nMethod: {method['name']} ({pending_count}/{len(cases)} cases pending)")
        pipe = None
        if pending_count > 0:
            print(f"Loading pipeline once for method: {method['name']}")
            set_seed(args.seed)
            pipe = build_pipeline(args)
            if method.get("lora"):
                apply_lora_adapter(pipe, method["lora"])

        try:
            for case_idx, case in enumerate(cases, start=1):
                print(f"\n[{case_idx}/{len(cases)}] Case: {case['name']} / Method: {method['name']}")
                manifest = generate_one(case, method, pipe, args)
                case_records[case["name"]]["methods"][method["name"]] = manifest
                summary["outputs"].append(manifest)
                (out_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        finally:
            if pipe is not None:
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nDone. Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
