import argparse
import csv
import json
from pathlib import Path

import torch

from compute_eval_metrics import (
    InceptionFeatureExtractor,
    build_pairs,
    frechet_distance_to_reference,
    load_cases,
    prepare_reference_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute InceptionV3 FID only from fixed MaterialMVP renders.")
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--renders-subdir", default="renders")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fid-weights-path", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:    
        writer = csv.DictWriter(f, fieldnames=["method", "num_pairs", "fid"])
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
    out_json = Path(args.out_json) if args.out_json else generated_root / "fid_only_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else generated_root / "fid_only_summary.csv"

    first_pairs, _ = build_pairs(cases, generated_root, args.methods[0], args.renders_subdir)
    shared_gt_paths = [item["gt"] for item in first_pairs]

    inception = InceptionFeatureExtractor(args.device, args.batch_size, args.fid_weights_path)
    print(f"Preparing GT Inception features: {len(shared_gt_paths)} renders, metric_device={metric_device}")
    gt_features = inception.encode(shared_gt_paths)
    reference_distribution = prepare_reference_distribution(gt_features.to(metric_device))

    rows = []
    details = {}
    for method in args.methods:
        pairs, _ = build_pairs(cases, generated_root, method, args.renders_subdir)
        gt_paths = [item["gt"] for item in pairs]
        if gt_paths != shared_gt_paths:
            raise ValueError(f"GT render ordering differs for method {method}; cannot reuse shared GT statistics.")

        gen_paths = [item["gen"] for item in pairs]
        print(f"Evaluating {method}: {len(gen_paths)} paired renders")
        gen_features = inception.encode(gen_paths)
        fid = frechet_distance_to_reference(gen_features, reference_distribution)
        row = {"method": method, "num_pairs": len(pairs), "fid": fid}
        rows.append(row)
        details[method] = {"metrics": row, "pairs_preview": pairs[:5]}
        print(f"{method}: FID={fid:.6f}")

    summary = {
        "eval_cases": str(Path(args.eval_cases).resolve()),
        "generated_root": str(generated_root.resolve()),
        "fid_implementation": "torchvision_inception_v3_imagenet1k_pool3",
        "fid_weights_path": args.fid_weights_path,
        "metric_device": metric_device,
        "methods": args.methods,
        "metrics": rows,
        "details": details,
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
