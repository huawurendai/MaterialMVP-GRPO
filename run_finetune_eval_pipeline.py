import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MaterialMVP LoRA fine-tuning followed by generation, fixed rendering, "
            "overall metrics, and per-sample win-rate evaluation."
        )
    )
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--train-name", required=True)
    parser.add_argument("--train-logdir", default="logs")
    parser.add_argument("--train-visible-gpus", default="0,1,2,3")
    parser.add_argument("--train-gpus", default="4", help="Value passed to train.py --gpus.")
    parser.add_argument("--method-name", required=True, help="Evaluation method folder/name for the trained LoRA.")
    parser.add_argument("--lora-checkpoint", default=None, help="Override auto-detected lora_last.pt.")
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--test-cases", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--train-sample-cases", required=True)
    parser.add_argument("--train-sample-output", required=True)

    parser.add_argument("--eval-visible-gpu", default="4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--no-remesh", action="store_true")
    parser.add_argument("--skip-base-generation", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")

    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--env-names", default="map1,map2,map3,map4,map5")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-samples", type=int, default=64)
    parser.add_argument("--render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--render-background-color", default="127,127,127")
    parser.add_argument("--render-max-workers", type=int, default=8)
    parser.add_argument("--render-timeout", type=int, default=0, help="Per Blender job timeout in seconds.")
    parser.add_argument("--skip-render", action="store_true")

    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--aesthetic-encoder-path", required=True)
    parser.add_argument("--aesthetic-predictor-path", required=True)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--fid-weights-path", default=None)
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-winrate", action="store_true")

    parser.add_argument("--hf-home", default="/data1/zxz/.cache/huggingface")
    parser.add_argument("--project-dir", default=None, help="Default: directory containing this script.")
    parser.add_argument("--train-script", default=None, help="Default: auto-detect train.py.")
    parser.add_argument("--generation-script", default=None, help="Default: auto-detect batch_generate_eval_meshes.py.")
    parser.add_argument("--render-script", default=None, help="Default: auto-detect batch_render_eval_outputs.py.")
    parser.add_argument("--metrics-script", default=None, help="Default: auto-detect compute_eval_metrics.py.")
    parser.add_argument("--winrate-script", default=None, help="Default: auto-detect compute_per_sample_winrate.py.")
    parser.add_argument("--state-path", default=None, help="Default: <train experiment dir>/pipeline_state.json.")
    return parser.parse_args()


def resolve_path(path, project_dir):
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = project_dir / value
    return value.resolve()


def resolve_script(path, filename, project_dir):
    if path:
        return resolve_path(path, project_dir)
    candidates = [project_dir / filename, project_dir / "scripts" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def validate_eval_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Benchmark cases must be a non-empty JSON list: {path}")

    required_keys = {"name", "mesh", "reference", "gt_renders"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case {index} is not an object: {path}")
        missing = sorted(required_keys - set(case))
        if missing:
            raise ValueError(f"Benchmark case {index} in {path} is missing keys: {', '.join(missing)}")
    return cases


def validate_existing_base(cases, output):
    missing = []
    for case in cases:
        mesh_path = output / case["name"] / "base" / "textured_mesh.glb"
        if not mesh_path.exists():
            missing.append(str(mesh_path))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            "--skip-base-generation was requested, but base meshes are missing. "
            f"First missing paths:\n{preview}"
        )


def experiment_dir(args, project_dir):
    config_stem = Path(args.train_config).stem
    train_logdir = resolve_path(args.train_logdir, project_dir)
    suffix = f"-{args.train_name}" if args.train_name else ""
    return train_logdir / f"{config_stem}{suffix}"


def auto_lora_checkpoint(args, project_dir):
    if args.lora_checkpoint:
        return resolve_path(args.lora_checkpoint, project_dir)
    return experiment_dir(args, project_dir) / "checkpoints" / "lora_last.pt"


def display_command(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def run_stage(name, command, cwd, env, state, state_path):
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
    print(display_command(command), flush=True)
    started = time.time()
    state["stages"][name] = {"status": "running", "started_at": started, "command": command}
    write_state(state_path, state)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except Exception as exc:
        state["stages"][name].update(
            {"status": "failed", "finished_at": time.time(), "error": repr(exc)}
        )
        write_state(state_path, state)
        raise
    state["stages"][name].update({"status": "completed", "finished_at": time.time()})
    write_state(state_path, state)


def base_environment(args):
    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home
    env["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def generation_command(args, cases, output, lora_checkpoint, generation_script):
    command = [
        sys.executable,
        "-u",
        str(generation_script),
        "--cases",
        str(cases),
        "--out-dir",
        str(output),
        "--resolution",
        str(args.resolution),
        "--max-num-view",
        str(args.max_num_view),
        "--seed",
        str(args.seed),
        "--lora",
        f"{args.method_name}={lora_checkpoint}",
    ]
    if args.skip_base_generation:
        command.append("--skip-base")
    if args.no_remesh:
        command.append("--no-remesh")
    return command


def render_command(args, cases, output, render_script, project_dir):
    command = [
        sys.executable,
        "-u",
        str(render_script),
        "--eval-cases",
        str(cases),
        "--generated-root",
        str(output),
        "--env-dir",
        str(resolve_path(args.env_dir, project_dir)),
        "--blender-bin",
        str(resolve_path(args.blender_bin, project_dir)),
        "--methods",
        "base",
        args.method_name,
        "--env-names",
        args.env_names,
        "--resolution",
        str(args.render_resolution),
        "--samples",
        str(args.render_samples),
        "--engine",
        args.render_engine,
        "--background-color",
        args.render_background_color,
        "--max-workers",
        str(args.render_max_workers),
    ]
    if args.render_timeout > 0:
        command.extend(["--timeout", str(args.render_timeout)])
    return command


def metrics_command(args, cases, output, metrics_script, project_dir):
    command = [
        sys.executable,
        "-u",
        str(metrics_script),
        "--eval-cases",
        str(cases),
        "--generated-root",
        str(output),
        "--methods",
        "base",
        args.method_name,
        "--renders-subdir",
        "renders",
        "--clip-model-path",
        str(resolve_path(args.clip_model_path, project_dir)),
        "--aesthetic-encoder-path",
        str(resolve_path(args.aesthetic_encoder_path, project_dir)),
        "--aesthetic-predictor-path",
        str(resolve_path(args.aesthetic_predictor_path, project_dir)),
        "--device",
        "cuda",
        "--metric-device",
        "cuda",
        "--dtype",
        args.dtype,
        "--aesthetic-device",
        "cuda",
        "--aesthetic-dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--lpips-batch-size",
        str(args.lpips_batch_size),
        "--out-json",
        str(output / f"metrics_base_{args.method_name}.json"),
        "--out-csv",
        str(output / f"metrics_base_{args.method_name}.csv"),
    ]
    if args.skip_fid:
        command.append("--skip-fid")
    if args.fid_weights_path:
        command.extend(["--fid-weights-path", str(resolve_path(args.fid_weights_path, project_dir))])
    return command


def winrate_command(args, cases, output, winrate_script, project_dir):
    return [
        sys.executable,
        "-u",
        str(winrate_script),
        "--eval-cases",
        str(cases),
        "--generated-root",
        str(output),
        "--baseline",
        "base",
        "--candidates",
        args.method_name,
        "--renders-subdir",
        "renders",
        "--clip-model-path",
        str(resolve_path(args.clip_model_path, project_dir)),
        "--aesthetic-encoder-path",
        str(resolve_path(args.aesthetic_encoder_path, project_dir)),
        "--aesthetic-predictor-path",
        str(resolve_path(args.aesthetic_predictor_path, project_dir)),
        "--device",
        "cuda",
        "--metric-device",
        "cuda",
        "--dtype",
        args.dtype,
        "--aesthetic-device",
        "cuda",
        "--aesthetic-dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--lpips-batch-size",
        str(args.lpips_batch_size),
        "--primary-metric",
        "clip_paired_distance_mean",
        "--out-json",
        str(output / f"per_sample_winrate_base_{args.method_name}.json"),
        "--out-case-csv",
        str(output / f"per_sample_metrics_base_{args.method_name}.csv"),
        "--out-winrate-csv",
        str(output / f"per_sample_winrate_base_{args.method_name}.csv"),
        "--out-comparison-csv",
        str(output / f"per_sample_comparisons_base_{args.method_name}.csv"),
    ]


def main():
    args = parse_args()
    project_dir = (
        resolve_path(args.project_dir, Path.cwd())
        if args.project_dir
        else Path(__file__).resolve().parent
    )
    train_config = resolve_path(args.train_config, project_dir)
    test_cases = resolve_path(args.test_cases, project_dir)
    train_sample_cases = resolve_path(args.train_sample_cases, project_dir)
    test_output = resolve_path(args.test_output, project_dir)
    train_sample_output = resolve_path(args.train_sample_output, project_dir)
    lora_checkpoint = auto_lora_checkpoint(args, project_dir)
    exp_dir = experiment_dir(args, project_dir)
    state_path = resolve_path(args.state_path, project_dir) if args.state_path else exp_dir / "pipeline_state.json"
    train_script = resolve_script(args.train_script, "train.py", project_dir)
    generation_script = resolve_script(args.generation_script, "batch_generate_eval_meshes.py", project_dir)
    render_script = resolve_script(args.render_script, "batch_render_eval_outputs.py", project_dir)
    metrics_script = resolve_script(args.metrics_script, "compute_eval_metrics.py", project_dir)
    winrate_script = resolve_script(args.winrate_script, "compute_per_sample_winrate.py", project_dir)

    required_paths = {
        "project_dir": project_dir,
        "test_cases": test_cases,
        "train_sample_cases": train_sample_cases,
    }
    if not args.skip_train:
        required_paths["train_config"] = train_config
        required_paths["train_script"] = train_script
    if not args.skip_generation:
        required_paths["generation_script"] = generation_script
    if not args.skip_render:
        required_paths["blender_bin"] = resolve_path(args.blender_bin, project_dir)
        required_paths["env_dir"] = resolve_path(args.env_dir, project_dir)
        required_paths["render_script"] = render_script
    if not args.skip_metrics or not args.skip_winrate:
        required_paths["clip_model_path"] = resolve_path(args.clip_model_path, project_dir)
        required_paths["aesthetic_encoder_path"] = resolve_path(args.aesthetic_encoder_path, project_dir)
        required_paths["aesthetic_predictor_path"] = resolve_path(args.aesthetic_predictor_path, project_dir)
    if not args.skip_metrics:
        required_paths["metrics_script"] = metrics_script
    if not args.skip_winrate:
        required_paths["winrate_script"] = winrate_script
    missing_paths = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Required pipeline paths are missing:\n" + "\n".join(missing_paths))

    test_case_entries = validate_eval_cases(test_cases)
    train_sample_case_entries = validate_eval_cases(train_sample_cases)
    if args.skip_train and not lora_checkpoint.exists():
        raise FileNotFoundError(f"Existing LoRA checkpoint not found: {lora_checkpoint}")
    if args.skip_base_generation:
        validate_existing_base(test_case_entries, test_output)
        validate_existing_base(train_sample_case_entries, train_sample_output)

    test_output.mkdir(parents=True, exist_ok=True)
    train_sample_output.mkdir(parents=True, exist_ok=True)

    state = {
        "project_dir": str(project_dir),
        "experiment_dir": str(exp_dir),
        "lora_checkpoint": str(lora_checkpoint),
        "method_name": args.method_name,
        "test_cases": str(test_cases),
        "train_sample_cases": str(train_sample_cases),
        "scripts": {
            "train": str(train_script),
            "generation": str(generation_script),
            "render": str(render_script),
            "metrics": str(metrics_script),
            "winrate": str(winrate_script),
        },
        "stages": {},
    }
    write_state(state_path, state)

    env = base_environment(args)
    train_env = env.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.train_visible_gpus
    eval_env = env.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = args.eval_visible_gpu

    if not args.skip_train:
        train_command = [
            sys.executable,
            "-u",
            str(train_script),
            "--base",
            str(train_config),
            "--name",
            args.train_name,
            "--logdir",
            str(resolve_path(args.train_logdir, project_dir)),
            "--gpus",
            args.train_gpus,
            "--seed",
            str(args.seed),
        ]
        run_stage("train", train_command, project_dir, train_env, state, state_path)

    if not lora_checkpoint.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found after training: {lora_checkpoint}. "
            "Pass --lora-checkpoint when evaluating an existing checkpoint."
        )

    benchmarks = [
        ("test", test_cases, test_output),
        ("train_sample", train_sample_cases, train_sample_output),
    ]

    for benchmark_name, cases, output in benchmarks:
        if not args.skip_generation:
            run_stage(
                f"{benchmark_name}_generate",
                generation_command(args, cases, output, lora_checkpoint, generation_script),
                project_dir,
                eval_env,
                state,
                state_path,
            )
        if not args.skip_render:
            run_stage(
                f"{benchmark_name}_render",
                render_command(args, cases, output, render_script, project_dir),
                project_dir,
                eval_env,
                state,
                state_path,
            )
        if not args.skip_metrics:
            run_stage(
                f"{benchmark_name}_metrics",
                metrics_command(args, cases, output, metrics_script, project_dir),
                project_dir,
                eval_env,
                state,
                state_path,
            )
        if not args.skip_winrate:
            run_stage(
                f"{benchmark_name}_winrate",
                winrate_command(args, cases, output, winrate_script, project_dir),
                project_dir,
                eval_env,
                state,
                state_path,
            )

    print(f"\nPipeline completed. State: {state_path}", flush=True)


if __name__ == "__main__":
    main()
