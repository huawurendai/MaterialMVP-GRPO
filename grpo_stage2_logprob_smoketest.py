import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from einops import rearrange
from PIL import Image

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
    pil_to_tensor,
    resolve_dtype,
    resolve_model_path,
    save_contact_sheet,
    save_target_sheet,
    set_seed,
    tensor_to_pil,
)
from materialmvp.pipeline import to_rgb_image
from materialmvp.rl.ddim_with_logprob import ddim_step_with_logprob


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 GRPO smoke test for original MaterialMVP. "
            "This records DDIM trajectory + old log-probs + reward/advantage, but does not update weights."
        )
    )
    parser.add_argument("--base", default="cfgs/v1_lora_test.yaml", help="Config used only for dataset loading.")
    parser.add_argument("--dataset-split", default="validation", choices=["train", "validation"])
    parser.add_argument("--pretrained-model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--pretrained-subdir", default="hunyuan3d-paintpbr-v2-1")
    parser.add_argument("--dino-ckpt-path", default="facebook/dinov2-giant")
    parser.add_argument("--out-dir", default="outputs/grpo_stage2")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0, help="DDIM stochasticity. Must be > 0 for log-prob.")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--save-latent-trajectory",
        action="store_true",
        help="Save full latent tensors. Without this, only metadata and log_probs are saved.",
    )
    return parser.parse_args()


def load_original_mvp_pipeline_for_ddim(args):
    dtype = resolve_dtype(args.dtype)
    model_path = resolve_model_path(args.pretrained_model_path, args.pretrained_subdir)
    pipe = DiffusionPipeline.from_pretrained(model_path, custom_pipeline="materialmvp", torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()
    pipe.to(args.device)

    dino = None
    if getattr(pipe.unet, "use_dino", False):
        from materialmvp.modules import Dino_v2

        dino = Dino_v2(args.dino_ckpt_path).to(device=args.device, dtype=dtype)
        dino.eval().requires_grad_(False)

    return pipe, dino


def pil_batch_to_tensor(batch_images, device, dtype):
    batches = []
    for images in batch_images:
        views = []
        for pil_img in images:
            image = to_rgb_image(pil_img)
            tensor = pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
            views.append(tensor)
        batches.append(torch.cat(views, dim=0).unsqueeze(0))
    return torch.cat(batches, dim=0)


@torch.no_grad()
def prepare_materialmvp_conditions(pipe, dino, cond_pil, normal_pils, position_pils, guidance_scale):
    pipe.prepare()
    device = pipe.vae.device
    dtype = pipe.unet.dtype

    images = [to_rgb_image(cond_pil)]
    images_vae = [torch.tensor(np.array(image) / 255.0) for image in images]
    images_vae = [image_vae.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0) for image_vae in images_vae]
    images_vae = torch.cat(images_vae, dim=1).to(device=device, dtype=dtype)

    batch_size = images_vae.shape[0]
    cached_condition = {
        "num_in_batch": len(normal_pils),
        "images_normal": pil_batch_to_tensor([normal_pils], device, dtype),
        "images_position": pil_batch_to_tensor([position_pils], device, dtype),
    }

    if getattr(pipe.unet, "use_ra", False):
        cached_condition["ref_latents"] = pipe.encode_images(images_vae)

    cached_condition["embeds_normal"] = pipe.encode_images(cached_condition["images_normal"])
    cached_condition["position_maps"] = cached_condition["images_position"]
    cached_condition["embeds_position"] = pipe.encode_images(cached_condition["images_position"])

    if getattr(pipe.unet, "use_dino", False):
        if dino is None:
            raise ValueError("Pipeline UNet uses DINO but no DINO model was loaded.")
        cached_condition["dino_hidden_states"] = dino(cond_pil)

    if getattr(pipe.unet, "use_learned_text_clip", False):
        tokens = []
        for token in pipe.unet.pbr_setting:
            tokens.append(getattr(pipe.unet, f"learned_text_clip_{token}").unsqueeze(0).repeat(batch_size, 1, 1))
        prompt_embeds = torch.stack(tokens, dim=1)
        negative_prompt_embeds = torch.stack(tokens, dim=1)
    else:
        prompt_embeds, _ = pipe.encode_prompt(["high quality"], device, 1, False)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    if guidance_scale > 1:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds])
        if "ref_latents" in cached_condition:
            cached_condition["ref_latents"] = cached_condition["ref_latents"].repeat(
                3, *([1] * (cached_condition["ref_latents"].dim() - 1))
            )
            cached_condition["ref_scale"] = torch.as_tensor([0.0, 1.0, 1.0]).to(cached_condition["ref_latents"])
        if "dino_hidden_states" in cached_condition:
            zero_states = torch.zeros_like(cached_condition["dino_hidden_states"])
            cached_condition["dino_hidden_states"] = torch.cat(
                [zero_states, zero_states, cached_condition["dino_hidden_states"]]
            )
        for key in ["embeds_normal", "embeds_position", "position_maps"]:
            if key in cached_condition:
                cached_condition[key] = cached_condition[key].repeat(
                    3, *([1] * (cached_condition[key].dim() - 1))
                )

    return prompt_embeds, cached_condition


def apply_materialmvp_guidance(noise_pred, num_view, n_pbr, guidance_scale, camera_azims=None):
    noise_pred_uncond, noise_pred_ref, noise_pred_full = noise_pred.chunk(3)
    camera_azims = camera_azims or [0] * num_view

    def cam_mapping(azim):
        if 0 <= azim < 90:
            return float(azim) / 90.0 + 1.0
        if 90 <= azim < 330:
            return 2.0
        return -float(azim) / 90.0 + 5.0

    view_scale_tensor = (
        torch.from_numpy(np.asarray([cam_mapping(azim) for azim in camera_azims]))
        .unsqueeze(0)
        .repeat(n_pbr, 1)
        .view(-1)
        .to(noise_pred_uncond)[:, None, None, None]
    )
    guided = noise_pred_uncond + guidance_scale * view_scale_tensor * (noise_pred_ref - noise_pred_uncond)
    guided = guided + guidance_scale * view_scale_tensor * (noise_pred_full - noise_pred_ref)
    return guided


@torch.no_grad()
def rollout_with_logprob(pipe, dino, cond_pil, normal_pils, position_pils, args, seed):
    if args.eta <= 0:
        raise ValueError("--eta must be > 0 for Stage-2 log-prob rollouts.")

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    prompt_embeds, cached_condition = prepare_materialmvp_conditions(
        pipe,
        dino,
        cond_pil,
        normal_pils,
        position_pils,
        args.guidance_scale,
    )
    rollout_condition = copy.copy(cached_condition)
    rollout_condition["cache"] = {}

    device = pipe._execution_device
    n_pbr = len(pipe.unet.pbr_setting)
    num_view = len(normal_pils)
    num_channels_latents = pipe.unet.config.in_channels
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = pipe.prepare_latents(
        num_view * n_pbr,
        num_channels_latents,
        args.resolution,
        args.resolution,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )

    all_latents = []
    all_next_latents = []
    all_log_probs = []
    all_noise_pred_norms = []

    for timestep in timesteps:
        latents_before = latents
        latent_grid = rearrange(latents, "(b n_pbr n) c h w -> b n_pbr n c h w", b=1, n_pbr=n_pbr, n=num_view)
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
            **rollout_condition,
        )[0]

        if args.guidance_scale > 1:
            noise_pred = apply_materialmvp_guidance(noise_pred, num_view, n_pbr, args.guidance_scale)

        latents, log_prob = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timestep,
            latents_before[:, :num_channels_latents],
            eta=args.eta,
            generator=generator,
        )

        all_latents.append(latents_before.detach().cpu())
        all_next_latents.append(latents.detach().cpu())
        all_log_probs.append(log_prob.detach().float().cpu())
        all_noise_pred_norms.append(float(noise_pred.detach().float().norm().cpu()))

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image_pils = [tensor_to_pil(image_i) for image_i in image]

    generated = {
        "albedo": image_pils[:num_view],
        "mr": image_pils[num_view : 2 * num_view],
    }
    trajectory = {
        "timesteps": timesteps.detach().cpu(),
        "latents": torch.stack(all_latents, dim=0),
        "next_latents": torch.stack(all_next_latents, dim=0),
        "old_log_probs": torch.stack(all_log_probs, dim=0),
        "noise_pred_norms": all_noise_pred_norms,
    }
    return generated, trajectory


def trajectory_summary(trajectory):
    log_probs = trajectory["old_log_probs"].float()
    timesteps = trajectory["timesteps"].long()
    return {
        "num_steps": int(timesteps.numel()),
        "timesteps": [int(item) for item in timesteps.tolist()],
        "latents_shape": list(trajectory["latents"].shape),
        "next_latents_shape": list(trajectory["next_latents"].shape),
        "old_log_probs_shape": list(log_probs.shape),
        "old_log_probs_mean": float(log_probs.mean()),
        "old_log_probs_std": float(log_probs.std(unbiased=False)),
        "old_log_probs_min": float(log_probs.min()),
        "old_log_probs_max": float(log_probs.max()),
        "noise_pred_norms": trajectory["noise_pred_norms"],
    }


def save_trajectory(trajectory, path, include_latents):
    payload = {
        "timesteps": trajectory["timesteps"],
        "old_log_probs": trajectory["old_log_probs"],
        "noise_pred_norms": trajectory["noise_pred_norms"],
    }
    if include_latents:
        payload["latents"] = trajectory["latents"]
        payload["next_latents"] = trajectory["next_latents"]
    torch.save(payload, path)


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
    pipe, dino = load_original_mvp_pipeline_for_ddim(args)

    all_records = []
    for sample_idx in range(min(args.max_samples, len(dataset))):
        sample_out_dir = out_dir / f"sample_{sample_idx:04d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        clean_sample = build_clean_sample(dataset, sample_idx, args.resolution, get_num_view(dataset))
        cond_pil = clean_sample["cond_pil"]
        normal_pils = clean_sample["normal_pils"]
        position_pils = clean_sample["position_pils"]
        target = clean_sample

        cond_pil.save(sample_out_dir / "condition.png")
        save_target_sheet(target, sample_out_dir / "target_albedo_mr.png")

        group_records = []
        for group_idx in range(args.group_size):
            seed = args.seed + sample_idx * 1000 + group_idx
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
            traj_summary = trajectory_summary(trajectory)
            metrics.update(
                {
                    "group_idx": group_idx,
                    "seed": seed,
                    "trajectory": traj_summary,
                }
            )

            save_contact_sheet(generated, sample_out_dir / f"group_{group_idx:02d}_albedo_mr.png")
            save_trajectory(
                trajectory,
                sample_out_dir / f"group_{group_idx:02d}_trajectory.pt",
                include_latents=args.save_latent_trajectory,
            )
            group_records.append(metrics)

            print(
                f"sample={sample_idx:04d} group={group_idx:02d} "
                f"reward={metrics['reward']:.6f} logp_mean={traj_summary['old_log_probs_mean']:.6f} "
                f"logp_std={traj_summary['old_log_probs_std']:.6f}"
            )

        rewards = torch.tensor([record["reward"] for record in group_records], dtype=torch.float32)
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)
        for record, advantage in zip(group_records, advantages):
            record["advantage"] = float(advantage)

        sample_record = {
            "sample_idx": sample_idx,
            "name": clean_sample["name"],
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "groups": group_records,
        }
        all_records.append(sample_record)
        (sample_out_dir / "stage2_metrics.json").write_text(
            json.dumps(sample_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"sample={sample_idx:04d} group_reward_mean={reward_mean:.6f} "
            f"group_reward_std={reward_std:.6f}"
        )

    summary = {
        "stage": "grpo_stage2_ddim_logprob_smoketest",
        "updates_model": False,
        "base_config": args.base,
        "pretrained_model_path": args.pretrained_model_path,
        "pretrained_subdir": args.pretrained_subdir,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "eta": args.eta,
        "guidance_scale": args.guidance_scale,
        "group_size": args.group_size,
        "records": all_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stage-2 summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
