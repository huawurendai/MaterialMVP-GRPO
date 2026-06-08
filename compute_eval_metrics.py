import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Compute MaterialMVP evaluation metrics from fixed renders.")
    parser.add_argument("--eval-cases", required=True, help="Benchmark eval_cases.json.")
    parser.add_argument("--generated-root", required=True, help="Root containing case/method/renders.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to evaluate, e.g. base ft.")
    parser.add_argument("--renders-subdir", default="renders")
    parser.add_argument("--clip-model-path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default=None, help="Device for CLIP-FID/CMMD math. Default: --device.")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument(
        "--fid-weights-path",
        default=None,
        help="Optional local torchvision InceptionV3 weights path. Default: torchvision cached/downloaded weights.",
    )
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--aesthetic-encoder-path", default=None)
    parser.add_argument("--aesthetic-predictor-path", default=None)
    parser.add_argument("--aesthetic-device", default=None)
    parser.add_argument("--aesthetic-dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--skip-aesthetic", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def resolve_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("--eval-cases must be a JSON list.")
    return cases


def detect_methods(generated_root, cases, renders_subdir):
    if not cases:
        return []
    first_case_dir = Path(generated_root) / cases[0]["name"]
    methods = []
    for path in sorted(first_case_dir.iterdir()):
        if path.is_dir() and (path / renders_subdir).exists():
            methods.append(path.name)
    if not methods:
        raise RuntimeError(f"No rendered method folders found under {first_case_dir}")
    return methods


def list_pngs(path):
    return sorted(Path(path).glob("*.png"))


def build_pairs(cases, generated_root, method, renders_subdir):
    pairs = []
    ref_pairs = []
    missing = []
    for case in cases:
        case_name = case["name"]
        gt_dir = Path(case["gt_renders"])
        gen_dir = Path(generated_root) / case_name / method / renders_subdir
        if not gt_dir.exists():
            missing.append(str(gt_dir))
            continue
        if not gen_dir.exists():
            missing.append(str(gen_dir))
            continue
        for gt_path in list_pngs(gt_dir):
            gen_path = gen_dir / gt_path.name
            if gen_path.exists():
                pairs.append({"case": case_name, "key": gt_path.name, "gt": str(gt_path), "gen": str(gen_path)})
                ref_pairs.append({"case": case_name, "reference": case["reference"], "gen": str(gen_path)})
            else:
                missing.append(str(gen_path))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing expected files, first entries:\n{preview}")
    return pairs, ref_pairs


def load_rgb(path):
    image = Image.open(path)
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (127, 127, 127, 255))
        image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


class CLIPFeatureExtractor:
    def __init__(self, model_path, device, dtype_name, batch_size):
        from transformers import CLIPImageProcessor, CLIPModel

        self.device = device
        self.dtype = resolve_dtype(dtype_name)
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_path).to(device=device, dtype=self.dtype).eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_path)
        self.feature_cache = {}

    @torch.no_grad()
    def encode(self, paths):
        paths = [str(path) for path in paths]
        if not paths:
            raise ValueError("No image paths were provided to CLIPFeatureExtractor.encode().")

        missing_paths = list(dict.fromkeys(path for path in paths if path not in self.feature_cache))
        for start in range(0, len(missing_paths), self.batch_size):
            batch_paths = missing_paths[start : start + self.batch_size]
            images = [load_rgb(path) for path in batch_paths]
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            batch = self.model.get_image_features(pixel_values=pixel_values).float()
            batch = torch.nn.functional.normalize(batch, dim=-1)
            for path, feature in zip(batch_paths, batch.cpu()):
                self.feature_cache[path] = feature
        return torch.stack([self.feature_cache[path] for path in paths], dim=0)


class InceptionFeatureExtractor:
    def __init__(self, device, batch_size, weights_path=None):
        from torch import nn
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device
        self.batch_size = batch_size
        self.feature_cache = {}

        if weights_path:
            self.model = inception_v3(weights=None, aux_logits=True, transform_input=False)
            checkpoint = torch.load(weights_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            checkpoint = {
                key.removeprefix("module.").removeprefix("model."): value
                for key, value in checkpoint.items()
            }
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            self.model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )

        self.model.fc = nn.Identity()
        self.model = self.model.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(self, image):
        image = image.resize((299, 299), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    @torch.no_grad()
    def encode(self, paths):
        paths = [str(path) for path in paths]
        if not paths:
            raise ValueError("No image paths were provided to InceptionFeatureExtractor.encode().")

        missing_paths = list(dict.fromkeys(path for path in paths if path not in self.feature_cache))
        for start in range(0, len(missing_paths), self.batch_size):
            batch_paths = missing_paths[start : start + self.batch_size]
            images = [self.preprocess(load_rgb(path)) for path in batch_paths]
            pixel_values = torch.stack(images, dim=0).to(self.device)
            pixel_values = (pixel_values - self.mean) / self.std
            batch = self.model(pixel_values).float()
            if isinstance(batch, tuple):
                batch = batch[0]
            batch = batch.reshape(batch.shape[0], -1)
            for path, feature in zip(batch_paths, batch.cpu()):
                self.feature_cache[path] = feature
        return torch.stack([self.feature_cache[path] for path in paths], dim=0)


class AestheticV25Scorer:
    def __init__(self, device, dtype_name, batch_size, encoder_path, predictor_path):
        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        except ImportError as exc:
            raise ImportError(
                "aesthetic_predictor_v2_5 is not installed. "
                "Install it or pass --skip-aesthetic."
            ) from exc

        self.device = device
        self.dtype = resolve_dtype(dtype_name)
        self.batch_size = batch_size
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=predictor_path,
            encoder_model_name=encoder_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = self.model.to(device=device, dtype=self.dtype).eval()

    @torch.no_grad()
    def score(self, paths):
        scores = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = [load_rgb(path) for path in batch_paths]
            pixel_values = self.preprocessor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            batch_scores = self.model(pixel_values).logits.squeeze(-1).float().detach().cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(float(item) for item in batch_scores)
        values = np.asarray(scores, dtype=np.float64)
        return {
            "aesthetic_mean": float(values.mean()),
            "aesthetic_std": float(values.std()),
            "aesthetic_min": float(values.min()),
            "aesthetic_max": float(values.max()),
        }


def covariance(features):
    x = features.double()
    mean = x.mean(dim=0)
    xc = x - mean
    cov = xc.T @ xc / max(x.shape[0] - 1, 1)
    return mean, cov


def matrix_sqrt_psd(matrix):
    matrix = (matrix + matrix.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=0.0)
    return (eigvecs * torch.sqrt(eigvals).unsqueeze(0)) @ eigvecs.T


def frechet_distance(features_a, features_b):
    mu_a, cov_a = covariance(features_a)
    mu_b, cov_b = covariance(features_b)
    diff = mu_a - mu_b

    sqrt_cov_a = matrix_sqrt_psd(cov_a)
    middle = sqrt_cov_a @ cov_b @ sqrt_cov_a
    covmean = matrix_sqrt_psd(middle)
    fid = diff.dot(diff) + torch.trace(cov_a + cov_b - 2.0 * covmean)
    return float(torch.clamp(fid, min=0.0).cpu())


def polynomial_kernel_self_mean(features):
    x = features.double()
    n = x.shape[0]
    if n <= 1:
        return torch.tensor(0.0, dtype=torch.float64, device=x.device)
    dim = x.shape[1]
    kernel = ((x @ x.T) / dim + 1.0) ** 3
    return (kernel.sum() - torch.diagonal(kernel).sum()) / (n * (n - 1))


def prepare_reference_distribution(features):
    mean, cov = covariance(features)
    return {
        "features": features.double(),
        "mean": mean,
        "cov": cov,
        "sqrt_cov": matrix_sqrt_psd(cov),
        "polynomial_self_mean": polynomial_kernel_self_mean(features),
    }


def frechet_distance_to_reference(features, reference_stats):
    features = features.to(reference_stats["mean"].device)
    mean, cov = covariance(features)
    diff = mean - reference_stats["mean"]
    sqrt_ref_cov = reference_stats["sqrt_cov"]
    middle = sqrt_ref_cov @ cov @ sqrt_ref_cov
    covmean = matrix_sqrt_psd(middle)
    fid = diff.dot(diff) + torch.trace(cov + reference_stats["cov"] - 2.0 * covmean)
    return float(torch.clamp(fid, min=0.0).cpu())


def polynomial_mmd_to_reference(features, reference_stats):
    x = features.to(reference_stats["features"].device).double()
    y = reference_stats["features"]
    dim = x.shape[1]
    xx = polynomial_kernel_self_mean(x)
    xy = (((x @ y.T) / dim + 1.0) ** 3).mean()
    value = xx + reference_stats["polynomial_self_mean"] - 2.0 * xy
    return float(torch.clamp(value, min=0.0).cpu())


def polynomial_mmd(features_a, features_b):
    # CLIP-MMD/CMMD-style distribution distance with a polynomial kernel.
    x = features_a.double()
    y = features_b.double()
    dim = x.shape[1]

    def kernel(a, b):
        return ((a @ b.T) / dim + 1.0) ** 3

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    n = x.shape[0]
    m = y.shape[0]
    if n > 1:
        xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    else:
        xx = torch.tensor(0.0, dtype=torch.float64)
    if m > 1:
        yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    else:
        yy = torch.tensor(0.0, dtype=torch.float64)
    xy = k_xy.mean()
    return float(torch.clamp(xx + yy - 2.0 * xy, min=0.0).cpu())


def clip_i_score(clipper, ref_pairs, gen_features=None):
    refs = []
    gens = []
    for item in ref_pairs:
        refs.append(item["reference"])
        gens.append(item["gen"])
    ref_features = clipper.encode(refs)
    if gen_features is None:
        gen_features = clipper.encode(gens)
    sims = (ref_features * gen_features).sum(dim=-1)
    return {
        "clip_i_mean": float(sims.mean()),
        "clip_i_std": float(sims.std(unbiased=False)),
        "clip_i_min": float(sims.min()),
        "clip_i_max": float(sims.max()),
    }


def load_lpips_model(device, net):
    try:
        import lpips
    except ImportError:
        return None
    model = lpips.LPIPS(net=net).to(device).eval()
    return model


def image_to_lpips_tensor(path_or_image, size=None):
    image = path_or_image if isinstance(path_or_image, Image.Image) else load_rgb(path_or_image)
    image = image.convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


@torch.no_grad()
def lpips_score(lpips_model, pairs, device, batch_size=4):
    if lpips_model is None:
        return None
    scores = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        gt_batch = []
        gen_batch = []
        for item in batch_pairs:
            gt_img = load_rgb(item["gt"])
            gen_img = load_rgb(item["gen"])
            size = gt_img.size
            gt_batch.append(image_to_lpips_tensor(gt_img, size=size))
            gen_batch.append(image_to_lpips_tensor(gen_img, size=size))
        gt = torch.cat(gt_batch, dim=0).to(device)
        gen = torch.cat(gen_batch, dim=0).to(device)
        batch_scores = lpips_model(gen, gt).reshape(-1).detach().cpu().tolist()
        scores.extend(float(score) for score in batch_scores)
    values = np.asarray(scores, dtype=np.float64)
    return {
        "lpips_mean": float(values.mean()),
        "lpips_std": float(values.std()),
        "lpips_min": float(values.min()),
        "lpips_max": float(values.max()),
    }


def evaluate_method(
    method,
    pairs,
    ref_pairs,
    clipper,
    lpips_model,
    aesthetic_scorer,
    inception,
    device,
    lpips_batch_size=4,
    gt_features=None,
    reference_distribution=None,
    gt_inception_features=None,
    inception_reference_distribution=None,
):
    gt_paths = [item["gt"] for item in pairs]
    gen_paths = [item["gen"] for item in pairs]
    if gt_features is None:
        gt_features = clipper.encode(gt_paths)
    print(f"  [{method}] extracting/reusing generated CLIP features")
    gen_features = clipper.encode(gen_paths)
    paired_cos = (gt_features * gen_features).sum(dim=-1)
    paired_clip_distance = 1.0 - paired_cos

    if reference_distribution is None:
        clip_fid = frechet_distance(gen_features, gt_features)
        cmmd = polynomial_mmd(gen_features, gt_features)
    else:
        clip_fid = frechet_distance_to_reference(gen_features, reference_distribution)
        cmmd = polynomial_mmd_to_reference(gen_features, reference_distribution)

    result = {
        "method": method,
        "num_pairs": len(pairs),
        "clip_fid": clip_fid,
        "cmmd": cmmd,
        "clip_paired_distance_mean": float(paired_clip_distance.mean()),
        "clip_paired_distance_std": float(paired_clip_distance.std(unbiased=False)),
        "clip_paired_similarity_mean": float(paired_cos.mean()),
    }
    if inception is not None:
        if gt_inception_features is None:
            gt_inception_features = inception.encode(gt_paths)
        print(f"  [{method}] extracting/reusing Inception features for FID")
        gen_inception_features = inception.encode(gen_paths)
        if inception_reference_distribution is None:
            result["fid"] = frechet_distance(gen_inception_features, gt_inception_features)
        else:
            result["fid"] = frechet_distance_to_reference(
                gen_inception_features,
                inception_reference_distribution,
            )
    result.update(clip_i_score(clipper, ref_pairs, gen_features=gen_features))
    if lpips_model is not None:
        print(f"  [{method}] computing LPIPS with batch_size={lpips_batch_size}")
    lpips_result = lpips_score(lpips_model, pairs, device, batch_size=lpips_batch_size)
    if lpips_result is not None:
        result.update(lpips_result)
    if aesthetic_scorer is not None:
        print(f"  [{method}] computing aesthetic scores")
        result.update(aesthetic_scorer.score(gen_paths))
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = [
        "method",
        "num_pairs",
        "clip_fid",
        "fid",
        "cmmd",
        "clip_i_mean",
        "clip_paired_similarity_mean",
        "clip_paired_distance_mean",
        "lpips_mean",
        "aesthetic_mean",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(keys))
    fieldnames = keys + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    metric_device = args.metric_device or args.device
    if metric_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA metric device requested but not available.")

    cases = load_cases(args.eval_cases)
    generated_root = Path(args.generated_root)
    methods = args.methods or detect_methods(generated_root, cases, args.renders_subdir)
    out_json = Path(args.out_json) if args.out_json else generated_root / "metrics_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else generated_root / "metrics_summary.csv"

    clipper = CLIPFeatureExtractor(args.clip_model_path, args.device, args.dtype, args.batch_size)
    inception = None
    if not args.skip_fid:
        try:
            inception = InceptionFeatureExtractor(args.device, args.batch_size, args.fid_weights_path)
        except Exception as exc:
            print(f"Warning: FID InceptionV3 model could not be loaded; FID will be skipped. {exc!r}")
    lpips_model = None if args.skip_lpips else load_lpips_model(args.device, args.lpips_net)
    if lpips_model is None and not args.skip_lpips:
        print("Warning: lpips is not installed; LPIPS will be skipped.")
    aesthetic_scorer = None
    if not args.skip_aesthetic:
        if args.aesthetic_encoder_path and args.aesthetic_predictor_path:
            aesthetic_device = args.aesthetic_device or args.device
            aesthetic_dtype = args.aesthetic_dtype or args.dtype
            aesthetic_scorer = AestheticV25Scorer(
                device=aesthetic_device,
                dtype_name=aesthetic_dtype,
                batch_size=args.batch_size,
                encoder_path=args.aesthetic_encoder_path,
                predictor_path=args.aesthetic_predictor_path,
            )
        else:
            print(
                "Warning: aesthetic paths were not provided; aesthetic score will be skipped. "
                "Use --aesthetic-encoder-path and --aesthetic-predictor-path to enable it."
            )

    if not methods:
        raise ValueError("No methods were selected for evaluation.")

    first_pairs, _first_ref_pairs = build_pairs(cases, generated_root, methods[0], args.renders_subdir)
    shared_gt_paths = [item["gt"] for item in first_pairs]
    print(
        f"Preparing shared GT CLIP features and distribution statistics: "
        f"{len(shared_gt_paths)} renders, metric_device={metric_device}"
    )
    shared_gt_features = clipper.encode(shared_gt_paths)
    reference_distribution = prepare_reference_distribution(shared_gt_features.to(metric_device))
    inception_reference_distribution = None
    shared_gt_inception_features = None
    if inception is not None:
        print(f"Preparing shared GT Inception features for FID: {len(shared_gt_paths)} renders")
        shared_gt_inception_features = inception.encode(shared_gt_paths)
        inception_reference_distribution = prepare_reference_distribution(
            shared_gt_inception_features.to(metric_device)
        )

    rows = []
    details = {}
    for method in methods:
        pairs, ref_pairs = build_pairs(cases, generated_root, method, args.renders_subdir)
        print(f"Evaluating {method}: {len(pairs)} paired renders")
        gt_paths = [item["gt"] for item in pairs]
        if gt_paths != shared_gt_paths:
            raise ValueError(f"GT render ordering differs for method {method}; cannot reuse shared GT statistics.")
        result = evaluate_method(
            method,
            pairs,
            ref_pairs,
            clipper,
            lpips_model,
            aesthetic_scorer,
            inception,
            args.device,
            lpips_batch_size=args.lpips_batch_size,
            gt_features=shared_gt_features,
            reference_distribution=reference_distribution,
            gt_inception_features=shared_gt_inception_features,
            inception_reference_distribution=inception_reference_distribution,
        )
        rows.append(result)
        details[method] = {
            "metrics": result,
            "pairs_preview": pairs[:5],
        }
        print(
            f"{method}: CLIP-FID={result['clip_fid']:.6f}, "
            + (f"FID={result['fid']:.6f}, " if "fid" in result else "")
            + f"CMMD={result['cmmd']:.6f}, CLIP-I={result['clip_i_mean']:.6f}"
            + (f", LPIPS={result['lpips_mean']:.6f}" if "lpips_mean" in result else "")
            + (f", Aesthetic={result['aesthetic_mean']:.6f}" if "aesthetic_mean" in result else "")
        )

    summary = {
        "eval_cases": str(Path(args.eval_cases).resolve()),
        "generated_root": str(generated_root.resolve()),
        "clip_model_path": args.clip_model_path,
        "fid_weights_path": args.fid_weights_path,
        "fid_implementation": "torchvision_inception_v3_imagenet1k_pool3",
        "metric_device": metric_device,
        "aesthetic_encoder_path": args.aesthetic_encoder_path,
        "aesthetic_predictor_path": args.aesthetic_predictor_path,
        "methods": methods,
        "metrics": rows,
        "details": details,
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
