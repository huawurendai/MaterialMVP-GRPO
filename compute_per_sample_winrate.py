import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from compute_eval_metrics import (
    AestheticV25Scorer,
    CLIPFeatureExtractor,
    frechet_distance_to_reference,
    image_to_lpips_tensor,
    list_pngs,
    load_cases,
    load_lpips_model,
    load_rgb,
    polynomial_mmd_to_reference,
    prepare_reference_distribution,
)


LOWER_IS_BETTER = {
    "clip_fid",
    "cmmd",
    "clip_paired_distance_mean",
    "lpips_mean",
}
HIGHER_IS_BETTER = {
    "clip_paired_similarity_mean",
    "clip_i_mean",
    "aesthetic_mean",
}
METRIC_DIRECTIONS = {metric: "lower" for metric in LOWER_IS_BETTER}
METRIC_DIRECTIONS.update({metric: "higher" for metric in HIGHER_IS_BETTER})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-case win rates for generated MaterialMVP renders against a baseline."
    )
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--renders-subdir", default="renders")
    parser.add_argument("--clip-model-path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default=None, help="Device for CLIP-FID/CMMD math. Default: --device.")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--aesthetic-encoder-path", default=None)
    parser.add_argument("--aesthetic-predictor-path", default=None)
    parser.add_argument("--aesthetic-device", default=None)
    parser.add_argument("--aesthetic-dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--skip-aesthetic", action="store_true")
    parser.add_argument("--primary-metric", default="clip_paired_distance_mean", choices=sorted(METRIC_DIRECTIONS))
    parser.add_argument("--tie-eps", type=float, default=1e-8)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-case-csv", default=None)
    parser.add_argument("--out-winrate-csv", default=None)
    parser.add_argument("--out-comparison-csv", default=None)
    return parser.parse_args()


def resolve_case_paths(case, generated_root, method, renders_subdir):
    case_name = case["name"]
    gt_dir = Path(case["gt_renders"])
    gen_dir = Path(generated_root) / case_name / method / renders_subdir
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not gen_dir.exists():
        raise FileNotFoundError(gen_dir)

    pairs = []
    for gt_path in list_pngs(gt_dir):
        gen_path = gen_dir / gt_path.name
        if not gen_path.exists():
            raise FileNotFoundError(gen_path)
        pairs.append({"key": gt_path.name, "gt": str(gt_path), "gen": str(gen_path)})
    if not pairs:
        raise RuntimeError(f"No render pairs found for {case_name}/{method}")
    return pairs


@torch.no_grad()
def lpips_case_score(lpips_model, pairs, device, batch_size=4):
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
            gt_batch.append(image_to_lpips_tensor(gt_img, size=gt_img.size))
            gen_batch.append(image_to_lpips_tensor(gen_img, size=gt_img.size))
        gt = torch.cat(gt_batch, dim=0).to(device)
        gen = torch.cat(gen_batch, dim=0).to(device)
        scores.extend(float(score) for score in lpips_model(gen, gt).reshape(-1).detach().cpu().tolist())
    return float(np.asarray(scores, dtype=np.float64).mean())


def evaluate_case_method(
    case,
    generated_root,
    method,
    renders_subdir,
    clipper,
    lpips_model,
    aesthetic_scorer,
    device,
    lpips_batch_size=4,
    gt_features=None,
    reference_distribution=None,
):
    pairs = resolve_case_paths(case, generated_root, method, renders_subdir)
    gt_paths = [item["gt"] for item in pairs]
    gen_paths = [item["gen"] for item in pairs]

    if gt_features is None:
        gt_features = clipper.encode(gt_paths)
    if reference_distribution is None:
        reference_distribution = prepare_reference_distribution(gt_features)
    gen_features = clipper.encode(gen_paths)
    paired_cos = (gt_features * gen_features).sum(dim=-1)
    paired_dist = 1.0 - paired_cos

    reference_paths = [case["reference"] for _ in gen_paths]
    ref_features = clipper.encode(reference_paths)
    clip_i = (ref_features * gen_features).sum(dim=-1)

    result = {
        "case": case["name"],
        "method": method,
        "num_pairs": len(pairs),
        "clip_fid": frechet_distance_to_reference(gen_features, reference_distribution),
        "cmmd": polynomial_mmd_to_reference(gen_features, reference_distribution),
        "clip_paired_distance_mean": float(paired_dist.mean()),
        "clip_paired_similarity_mean": float(paired_cos.mean()),
        "clip_i_mean": float(clip_i.mean()),
    }

    lpips_mean = lpips_case_score(lpips_model, pairs, device, batch_size=lpips_batch_size)
    if lpips_mean is not None:
        result["lpips_mean"] = lpips_mean
    if aesthetic_scorer is not None:
        result["aesthetic_mean"] = aesthetic_scorer.score(gen_paths)["aesthetic_mean"]
    return result


def winner_for_metric(base_score, cand_score, metric, tie_eps):
    direction = METRIC_DIRECTIONS[metric]
    delta = cand_score - base_score
    if abs(delta) <= tie_eps:
        return "tie", delta
    if direction == "lower":
        return ("candidate" if cand_score < base_score else "base"), delta
    return ("candidate" if cand_score > base_score else "base"), delta


def summarize_comparisons(case_metrics, baseline, candidates, primary_metric, tie_eps):
    comparisons = []
    winrate_rows = []
    metrics = [
        metric
        for metric in METRIC_DIRECTIONS
        if all(metric in case_metrics[case][baseline] for case in case_metrics)
    ]

    for candidate in candidates:
        for metric in metrics:
            wins = losses = ties = 0
            deltas = []
            for case_name, methods in case_metrics.items():
                base_score = methods[baseline][metric]
                cand_score = methods[candidate][metric]
                winner, delta = winner_for_metric(base_score, cand_score, metric, tie_eps)
                wins += int(winner == "candidate")
                losses += int(winner == "base")
                ties += int(winner == "tie")
                deltas.append(delta)
                comparisons.append(
                    {
                        "case": case_name,
                        "candidate": candidate,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "base_score": base_score,
                        "candidate_score": cand_score,
                        "delta_candidate_minus_base": delta,
                        "winner": winner,
                        "is_primary_metric": metric == primary_metric,
                    }
                )
            total = wins + losses + ties
            winrate_rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS[metric],
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "total": total,
                    "win_rate": wins / max(total, 1),
                    "non_tie_win_rate": wins / max(wins + losses, 1),
                    "mean_delta_candidate_minus_base": float(np.asarray(deltas, dtype=np.float64).mean()),
                    "is_primary_metric": metric == primary_metric,
                }
            )
    return comparisons, winrate_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    metric_device = args.metric_device or args.device
    if metric_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA metric device requested but not available.")

    generated_root = Path(args.generated_root)
    out_json = Path(args.out_json) if args.out_json else generated_root / "per_sample_winrate.json"
    out_case_csv = Path(args.out_case_csv) if args.out_case_csv else generated_root / "per_sample_metrics.csv"
    out_winrate_csv = Path(args.out_winrate_csv) if args.out_winrate_csv else generated_root / "per_sample_winrate.csv"
    out_comparison_csv = (
        Path(args.out_comparison_csv) if args.out_comparison_csv else generated_root / "per_sample_comparisons.csv"
    )

    cases = load_cases(args.eval_cases)
    methods = [args.baseline] + args.candidates
    clipper = CLIPFeatureExtractor(args.clip_model_path, args.device, args.dtype, args.batch_size)
    lpips_model = None if args.skip_lpips else load_lpips_model(args.device, args.lpips_net)

    aesthetic_scorer = None
    if not args.skip_aesthetic:
        if args.aesthetic_encoder_path and args.aesthetic_predictor_path:
            aesthetic_scorer = AestheticV25Scorer(
                device=args.aesthetic_device or args.device,
                dtype_name=args.aesthetic_dtype or args.dtype,
                batch_size=args.batch_size,
                encoder_path=args.aesthetic_encoder_path,
                predictor_path=args.aesthetic_predictor_path,
            )
        else:
            print("Warning: aesthetic paths not provided; skipping aesthetic per-case score.")

    case_metrics = {}
    flat_case_rows = []
    for case in cases:
        print(f"Evaluating case: {case['name']}")
        case_metrics[case["name"]] = {}
        baseline_pairs = resolve_case_paths(case, generated_root, args.baseline, args.renders_subdir)
        gt_features = clipper.encode([item["gt"] for item in baseline_pairs])
        reference_distribution = prepare_reference_distribution(gt_features.to(metric_device))
        for method in methods:
            row = evaluate_case_method(
                case,
                generated_root,
                method,
                args.renders_subdir,
                clipper,
                lpips_model,
                aesthetic_scorer,
                args.device,
                lpips_batch_size=args.lpips_batch_size,
                gt_features=gt_features,
                reference_distribution=reference_distribution,
            )
            case_metrics[case["name"]][method] = row
            flat_case_rows.append(row)

    comparisons, winrate_rows = summarize_comparisons(
        case_metrics,
        args.baseline,
        args.candidates,
        args.primary_metric,
        args.tie_eps,
    )

    summary = {
        "eval_cases": str(Path(args.eval_cases).resolve()),
        "generated_root": str(generated_root.resolve()),
        "baseline": args.baseline,
        "candidates": args.candidates,
        "primary_metric": args.primary_metric,
        "metric_device": metric_device,
        "metric_directions": METRIC_DIRECTIONS,
        "case_metrics": case_metrics,
        "winrates": winrate_rows,
        "comparisons": comparisons,
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_case_csv, flat_case_rows)
    write_csv(out_winrate_csv, winrate_rows)
    write_csv(out_comparison_csv, comparisons)

    print(f"Saved JSON: {out_json}")
    print(f"Saved per-case metrics CSV: {out_case_csv}")
    print(f"Saved winrate CSV: {out_winrate_csv}")
    print(f"Saved comparisons CSV: {out_comparison_csv}")
    print("Primary metric win rates:")
    for row in winrate_rows:
        if row["metric"] == args.primary_metric:
            print(
                f"  {row['candidate']}: wins={row['wins']}, losses={row['losses']}, "
                f"ties={row['ties']}, win_rate={row['win_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
