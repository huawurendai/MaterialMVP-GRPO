import argparse
import copy
import json
from pathlib import Path

import torch
from einops import rearrange

try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except Exception as exc:
    print(f"Warning: torchvision compatibility fix was not applied: {exc}")

from grpo_stage1_smoketest import (
    build_clean_sample,
    compute_reward,
    get_num_view,
    load_dataset_from_config,
    save_contact_sheet,
    save_target_sheet,
    set_seed,
)
from grpo_stage2_logprob_smoketest import (
    apply_materialmvp_guidance,
    load_original_mvp_pipeline_for_ddim,
    prepare_materialmvp_conditions,
    rollout_with_logprob,
    trajectory_summary,
)
from materialmvp.lora_utils import inject_lora_into_attention, save_lora_checkpoint
from materialmvp.rl.ddim_with_logprob import ddim_step_with_logprob


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-3 minimal GRPO training for original MaterialMVP. "
            "Starts from the original model, injects a fresh LoRA adapter, samples trajectories, "
            "and updates only trainable adapter/extra parameters."
        )
    )
    parser.add_argument("--base", default="cfgs/v1_lora_test.yaml", help="Config used only for dataset loading.")
    parser.add_argument("--dataset-split", default="validation", choices=["train", "validation"])
    parser.add_argument("--pretrained-model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--pretrained-subdir", default="hunyuan3d-paintpbr-v2-1")
    parser.add_argument("--dino-ckpt-path", default="facebook/dinov2-giant")
    parser.add_argument("--out-dir", default="outputs/grpo_stage3")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument(
        "--train-timestep-fraction",
        type=float,
        default=0.25,
        help="Fraction of rollout timesteps used for GRPO updates. Lower values reduce memory/time.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def setup_trainable_lora(pipe, args):
    report = inject_lora_into_attention(
        pipe.unet,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_suffixes=(
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "to_q_mr",
            "to_k_mr",
            "to_v_mr",
            "to_out_mr.0",
        ),
        include_keywords=("attn1", "attn2", "attn_refview"),
        exclude_keywords=("unet_dual", "attn_multiview", "attn_dino"),
        extra_trainable_keywords=(
            "learned_text_clip_albedo",
            "learned_text_clip_mr",
            "learned_text_clip_ref",
        ),
        freeze_first=True,
    )
    print(
        "Stage-3 LoRA enabled: "
        f"modules={report.module_count}, trainable={report.trainable_params}, total={report.total_params}, "
        f"ratio={report.trainable_params / max(report.total_params, 1):.6f}"
    )
    return report


def trainable_parameters(pipe):
    params = [param for param in pipe.unet.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found after LoRA setup.")
    return params


def prepare_training_condition(pipe, dino, cond_pil, normal_pils, position_pils, args):
    prompt_embeds, cached_condition = prepare_materialmvp_conditions(
        pipe,
        dino,
        cond_pil,
        normal_pils,
        position_pils,
        args.guidance_scale,
    )

    if getattr(pipe.unet, "use_learned_text_clip", False):
        # Rebuild prompt embeddings with gradients for learned text tokens.
        batch_size = 1
        tokens = []
        for token in pipe.unet.pbr_setting:
            tokens.append(getattr(pipe.unet, f"learned_text_clip_{token}").unsqueeze(0).repeat(batch_size, 1, 1))
        prompt_embeds_grad = torch.stack(tokens, dim=1)
        negative_prompt_embeds_grad = torch.stack(tokens, dim=1)
        if args.guidance_scale > 1:
            prompt_embeds = torch.cat([negative_prompt_embeds_grad, prompt_embeds_grad, prompt_embeds_grad])
        else:
            prompt_embeds = prompt_embeds_grad

    train_condition = copy.copy(cached_condition)
    train_condition["cache"] = {}
    return prompt_embeds, train_condition


def compute_new_log_probs_for_trajectory(pipe, dino, sample, trajectory, args):
    prompt_embeds, cached_condition = prepare_training_condition(
        pipe,
        dino,
        sample["cond_pil"],
        sample["normal_pils"],
        sample["position_pils"],
        args,
    )

    device = pipe._execution_device
    dtype = prompt_embeds.dtype
    n_pbr = len(pipe.unet.pbr_setting)
    num_view = len(sample["normal_pils"])
    num_channels_latents = pipe.unet.config.in_channels

    timesteps = trajectory["timesteps"].to(device)
    latents = trajectory["latents"].to(device=device, dtype=dtype)
    next_latents = trajectory["next_latents"].to(device=device, dtype=dtype)

    total_steps = timesteps.numel()
    train_steps = max(1, int(total_steps * args.train_timestep_fraction))
    step_indices = list(range(train_steps))

    new_log_probs = []
    for step_idx in step_indices:
        timestep = timesteps[step_idx]
        latents_t = latents[step_idx]
        next_latents_t = next_latents[step_idx]

        latent_grid = rearrange(
            latents_t,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            b=1,
            n_pbr=n_pbr,
            n=num_view,
        )
        if args.guidance_scale > 1:
            latent_model_input = latent_grid.repeat(3, 1, 1, 1, 1, 1)
        else:
            latent_model_input = latent_grid
        latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        latent_model_input = rearrange(
            latent_model_input,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            n=num_view,
            n_pbr=n_pbr,
        )

        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
            **cached_condition,
        )[0]
        if args.guidance_scale > 1:
            noise_pred = apply_materialmvp_guidance(noise_pred, num_view, n_pbr, args.guidance_scale)

        _prev_sample, log_prob = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timestep,
            latents_t[:, :num_channels_latents],
            eta=args.eta,
            prev_sample=next_latents_t[:, :num_channels_latents],
        )
        new_log_probs.append(log_prob)

    return torch.stack(new_log_probs, dim=0), step_indices


def backward_grpo_for_trajectory(pipe, dino, sample, trajectory, advantage, args, loss_scale=1.0):
    device = pipe._execution_device
    total_steps = int(trajectory["timesteps"].numel())
    train_steps = max(1, int(total_steps * args.train_timestep_fraction))
    step_indices = list(range(train_steps))

    old_log_probs_all = trajectory["old_log_probs"]
    losses = []
    ratio_means = []
    ratio_stds = []
    approx_kls = []
    clipfracs = []

    for step_idx in step_indices:
        # Rebuild the train-time condition every timestep. This is slower but
        # avoids keeping the huge MaterialMVP attention/position-RoPE graph
        # alive across all timesteps on 16GB GPUs.
        prompt_embeds, cached_condition = prepare_training_condition(
            pipe,
            dino,
            sample["cond_pil"],
            sample["normal_pils"],
            sample["position_pils"],
            args,
        )

        dtype = prompt_embeds.dtype
        n_pbr = len(pipe.unet.pbr_setting)
        num_view = len(sample["normal_pils"])
        num_channels_latents = pipe.unet.config.in_channels

        timestep = trajectory["timesteps"][step_idx].to(device)
        latents_t = trajectory["latents"][step_idx].to(device=device, dtype=dtype)
        next_latents_t = trajectory["next_latents"][step_idx].to(device=device, dtype=dtype)

        latent_grid = rearrange(
            latents_t,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            b=1,
            n_pbr=n_pbr,
            n=num_view,
        )
        latent_model_input = latent_grid.repeat(3, 1, 1, 1, 1, 1) if args.guidance_scale > 1 else latent_grid
        latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        latent_model_input = rearrange(
            latent_model_input,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            n=num_view,
            n_pbr=n_pbr,
        )

        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
            **cached_condition,
        )[0]
        if args.guidance_scale > 1:
            noise_pred = apply_materialmvp_guidance(noise_pred, num_view, n_pbr, args.guidance_scale)

        _prev_sample, new_log_prob = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timestep,
            latents_t[:, :num_channels_latents],
            eta=args.eta,
            prev_sample=next_latents_t[:, :num_channels_latents],
        )

        old_log_prob = old_log_probs_all[step_idx].to(new_log_prob.device, dtype=new_log_prob.dtype)
        loss, info = grpo_loss_from_log_probs(new_log_prob, old_log_prob, advantage, args)
        (loss * loss_scale / len(step_indices)).backward()

        losses.append(float(loss.detach().cpu()))
        ratio_means.append(info["ratio_mean"])
        ratio_stds.append(info["ratio_std"])
        approx_kls.append(info["approx_kl"])
        clipfracs.append(info["clipfrac"])

        del prompt_embeds, cached_condition, latent_model_input, noise_pred, new_log_prob, loss

    def mean(values):
        return float(sum(values) / max(len(values), 1))

    return {
        "trained_timestep_indices": step_indices,
        "loss": mean(losses),
        "ratio_mean": mean(ratio_means),
        "ratio_std": mean(ratio_stds),
        "approx_kl": mean(approx_kls),
        "clipfrac": mean(clipfracs),
    }


def grpo_loss_from_log_probs(new_log_probs, old_log_probs, advantage, args):
    old_log_probs = old_log_probs.to(new_log_probs.device, dtype=new_log_probs.dtype)
    advantage = torch.as_tensor(advantage, device=new_log_probs.device, dtype=new_log_probs.dtype)
    advantage = advantage.clamp(-args.adv_clip_max, args.adv_clip_max)

    ratio = torch.exp((new_log_probs - old_log_probs).clamp(-20, 20))
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
    loss = torch.maximum(unclipped, clipped).mean()
    approx_kl = 0.5 * (new_log_probs - old_log_probs).pow(2).mean()
    clipfrac = (torch.abs(ratio - 1.0) > args.clip_range).float().mean()
    return loss, {
        "ratio_mean": float(ratio.detach().mean().cpu()),
        "ratio_std": float(ratio.detach().std(unbiased=False).cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clipfrac": float(clipfrac.detach().cpu()),
    }


def save_stage3_checkpoint(path, pipe, lora_report, args, update_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    lora_config = {
        **lora_report.config,
        "extra_trainable_keywords": [
            "learned_text_clip_albedo",
            "learned_text_clip_mr",
            "learned_text_clip_ref",
        ],
        "stage": "grpo_stage3",
        "update_idx": update_idx,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
    }
    save_lora_checkpoint(str(path), pipe.unet, lora_config)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.eta <= 0:
        raise ValueError("--eta must be > 0.")
    if not (0 < args.train_timestep_fraction <= 1):
        raise ValueError("--train-timestep-fraction must be in (0, 1].")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_from_config(args.base, args.dataset_split)
    pipe, dino = load_original_mvp_pipeline_for_ddim(args)
    lora_report = setup_trainable_lora(pipe, args)
    optimizer = torch.optim.AdamW(trainable_parameters(pipe), lr=args.learning_rate)

    update_records = []
    global_update = 0
    max_samples = min(args.max_samples, len(dataset))
    while global_update < args.max_updates:
        sample_idx = global_update % max_samples
        sample_out_dir = out_dir / f"update_{global_update:04d}_sample_{sample_idx:04d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        clean_sample = build_clean_sample(dataset, sample_idx, args.resolution, get_num_view(dataset))
        cond_pil = clean_sample["cond_pil"]
        normal_pils = clean_sample["normal_pils"]
        position_pils = clean_sample["position_pils"]
        target = clean_sample

        cond_pil.save(sample_out_dir / "condition.png")
        save_target_sheet(target, sample_out_dir / "target_albedo_mr.png")

        pipe.unet.eval()
        group_payloads = []
        for group_idx in range(args.group_size):
            seed = args.seed + global_update * 1000 + group_idx
            generated, trajectory = rollout_with_logprob(
                pipe,
                dino,
                cond_pil,
                normal_pils,
                position_pils,
                args,
                seed,
            )
            metrics = compute_reward(generated, target, pipe.device, args.resolution)
            save_contact_sheet(generated, sample_out_dir / f"group_{group_idx:02d}_albedo_mr_before.png")
            group_payloads.append(
                {
                    "group_idx": group_idx,
                    "seed": seed,
                    "metrics": metrics,
                    "trajectory": trajectory,
                }
            )

        rewards = torch.tensor([item["metrics"]["reward"] for item in group_payloads], dtype=torch.float32)
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        pipe.unet.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_value = 0.0
        group_train_records = []
        train_sample = {
            "cond_pil": cond_pil,
            "normal_pils": normal_pils,
            "position_pils": position_pils,
        }

        for payload, advantage in zip(group_payloads, advantages):
            info = backward_grpo_for_trajectory(
                pipe,
                dino,
                train_sample,
                payload["trajectory"],
                float(advantage),
                args,
                loss_scale=1.0 / len(group_payloads),
            )
            total_loss_value += info["loss"]
            group_train_records.append(
                {
                    **payload["metrics"],
                    "group_idx": payload["group_idx"],
                    "seed": payload["seed"],
                    "advantage": float(advantage),
                    "loss": info["loss"],
                    "trajectory": trajectory_summary(payload["trajectory"]),
                    **info,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(pipe), args.max_grad_norm)
        optimizer.step()
        total_loss_value /= max(len(group_payloads), 1)

        record = {
            "update": global_update,
            "sample_idx": sample_idx,
            "name": clean_sample["name"],
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "total_loss": total_loss_value,
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "groups": group_train_records,
        }
        update_records.append(record)
        (sample_out_dir / "stage3_metrics.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"update={global_update:04d} sample={sample_idx:04d} "
            f"loss={record['total_loss']:.6f} reward_mean={record['reward_mean']:.6f} "
            f"reward_std={record['reward_std']:.6f} grad_norm={record['grad_norm']:.6f}"
        )

        if args.save_every > 0 and (global_update + 1) % args.save_every == 0:
            ckpt_path = out_dir / "checkpoints" / f"lora_grpo_step_{global_update + 1:06d}.pt"
            save_stage3_checkpoint(ckpt_path, pipe, lora_report, args, global_update + 1)
            print(f"Saved Stage-3 LoRA checkpoint: {ckpt_path}")

        global_update += 1

    summary = {
        "stage": "grpo_stage3_minimal_train",
        "updates_model": True,
        "trainable": "fresh_lora_plus_learned_text_tokens",
        "base_config": args.base,
        "pretrained_model_path": args.pretrained_model_path,
        "pretrained_subdir": args.pretrained_subdir,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "eta": args.eta,
        "guidance_scale": args.guidance_scale,
        "group_size": args.group_size,
        "max_updates": args.max_updates,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "lora": lora_report.config,
        "records": update_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    final_path = out_dir / "checkpoints" / "lora_grpo_last.pt"
    save_stage3_checkpoint(final_path, pipe, lora_report, args, global_update)
    print(f"Saved final Stage-3 LoRA checkpoint: {final_path}")
    print(f"Stage-3 summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
