import argparse
from collections import OrderedDict


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def fmt_mem(bytes_: int) -> str:
    return f"{bytes_ / (1024 ** 3):.2f}GiB"


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
        print(
            f"{name:<{name_width}}  "
            f"{fmt_params(stats['params']):>12}  "
            f"{fmt_params(stats['trainable']):>12}  "
            f"{fmt_mem(stats['bytes']):>10}"
        )


def main():
    parser = argparse.ArgumentParser(description="Count parameters in a local Stable Diffusion diffusers model.")
    parser.add_argument(
        "--model-path",
        default="/home/zengxiangzhao/models/stable-diffusion-2-1-base",
        help="Local diffusers model directory.",
    )
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    dtype = dtype_from_name(args.torch_dtype)

    print(f"Loading pipeline from: {args.model_path}")
    print(f"Counting with dtype: {args.torch_dtype}")

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )

    rows = OrderedDict()
    component_names = (
        "unet",
        "vae",
        "text_encoder",
        "image_encoder",
        "safety_checker",
        "controlnet",
    )

    for name in component_names:
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            module.eval()
            add_row(rows, name, module)

    total = {
        "params": sum(stats["params"] for stats in rows.values()),
        "trainable": sum(stats["trainable"] for stats in rows.values()),
        "bytes": sum(stats["bytes"] for stats in rows.values()),
    }
    rows["TOTAL"] = total

    print()
    print_rows(rows)


if __name__ == "__main__":
    main()
