import argparse
import os
from collections import OrderedDict


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def dtype_from_name(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def module_stats(module):
    total = 0
    trainable = 0
    bytes_ = 0
    seen = set()
    for param in module.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        n = param.numel()
        total += n
        bytes_ += n * param.element_size()
        if param.requires_grad:
            trainable += n
    return total, trainable, bytes_


def add_row(rows, name: str, module):
    total, trainable, bytes_ = module_stats(module)
    rows[name] = {
        "params": total,
        "trainable": trainable,
        "bytes": bytes_,
    }


def print_rows(rows):
    name_width = max([len("component")] + [len(name) for name in rows])
    print(f"{'component':<{name_width}}  {'params':>12}  {'trainable':>12}  {'param_mem':>10}")
    print(f"{'-' * name_width}  {'-' * 12}  {'-' * 12}  {'-' * 10}")
    for name, stats in rows.items():
        gib = stats["bytes"] / (1024**3)
        print(
            f"{name:<{name_width}}  "
            f"{fmt_params(stats['params']):>12}  "
            f"{fmt_params(stats['trainable']):>12}  "
            f"{gib:>9.2f}G"
        )


def resolve_materialmvp_path(args):
    if args.model_path:
        return args.model_path

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=args.repo_id,
        allow_patterns=[f"{args.subfolder}/*"],
        local_files_only=args.local_files_only,
    )
    return os.path.join(root, args.subfolder)


def main():
    parser = argparse.ArgumentParser(description="Count MaterialMVP inference parameters by loaded component.")
    parser.add_argument("--repo-id", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", default="hunyuan3d-paintpbr-v2-1")
    parser.add_argument("--model-path", default=None, help="Local path to hunyuan3d-paintpbr-v2-1.")
    parser.add_argument("--dino-path", default="facebook/dinov2-giant")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download from Hugging Face.")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-realesrgan", action="store_true")
    args = parser.parse_args()

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    rows = OrderedDict()
    dtype = dtype_from_name(args.torch_dtype)

    import torch
    from diffusers import DiffusionPipeline

    model_path = resolve_materialmvp_path(args)
    print(f"Loading MaterialMVP diffusion pipeline from: {model_path}")
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="materialmvp",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipeline.eval()

    for name in ("unet", "vae", "text_encoder"):
        module = getattr(pipeline, name, None)
        if isinstance(module, torch.nn.Module):
            add_row(rows, f"diffusion.{name}", module)

    if not args.skip_dino and getattr(pipeline.unet, "use_dino", False):
        print(f"Loading DINO from: {args.dino_path}")
        from materialmvp.modules import Dino_v2

        dino = Dino_v2(args.dino_path).to(dtype)
        add_row(rows, "dino_v2", dino)

    if not args.skip_realesrgan:
        print("Instantiating RealESRGAN RRDBNet architecture.")
        try:
            from utils.torchvision_fix import apply_fix

            apply_fix()
        except Exception as exc:
            print(f"Warning: torchvision compatibility fix was not applied: {exc}")

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet

            realesrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            realesrgan = realesrgan.to(dtype)
            add_row(rows, "realesrgan.rrdbnet", realesrgan)
        except ModuleNotFoundError as exc:
            print(f"Warning: skipped RealESRGAN parameter count because a dependency is missing: {exc}")
            print("Run with --skip-realesrgan to count only the diffusion/DINO models.")

    total = {
        "params": sum(stats["params"] for stats in rows.values()),
        "trainable": sum(stats["trainable"] for stats in rows.values()),
        "bytes": sum(stats["bytes"] for stats in rows.values()),
    }
    rows["TOTAL_LOADED_FOR_INFERENCE"] = total
    print()
    print_rows(rows)


if __name__ == "__main__":
    main()
