# 原版 MaterialMVP 直接做 GRPO：阶段 2 说明

## 阶段 2 是什么

阶段 2 是 **DDIM trajectory + old log-prob smoke test**。对应脚本：

```bash
python grpo_stage2_logprob_smoketest.py --base cfgs/v1_lora_test.yaml --group-size 2 --max-samples 1 --resolution 256 --num-inference-steps 8 --eta 1.0
```

它仍然不训练模型，不创建 optimizer，不 backward。它在阶段 1 的 rollout/reward/advantage 基础上，额外记录每一步扩散转移：

- `timesteps`
- `latents`
- `next_latents`
- `old_log_probs`
- `noise_pred_norms`

这些就是下一阶段实现 GRPO/PPO loss 时需要的核心数据。

## 为什么阶段 2 改用 DDIM

DanceGRPO 的 Stable Diffusion 实现使用的是 `DDIMScheduler + ddim_step_with_logprob`。原因是 GRPO 需要知道：

```text
log p_theta(latents_{t-1} | latents_t, condition)
```

普通 UniPC / Euler 推理更适合生成质量，但不方便直接给出这种 transition log-prob。阶段 2 因此使用 DDIM，并要求：

```text
--eta > 0
```

如果 `eta = 0`，DDIM 变成确定性转移，transition 分布退化，log-prob 不适合作为策略概率。

## 输出内容

默认输出目录：

```text
outputs/grpo_stage2/
```

每个 sample 会保存：

- `condition.png`
- `target_albedo_mr.png`
- `group_00_albedo_mr.png`
- `group_00_trajectory.pt`
- `group_01_albedo_mr.png`
- `group_01_trajectory.pt`
- `stage2_metrics.json`

总目录会保存：

- `summary.json`

默认 `group_xx_trajectory.pt` 只保存轻量数据：

- `timesteps`
- `old_log_probs`
- `noise_pred_norms`

如果需要保存完整 latent 轨迹，额外加：

```bash
--save-latent-trajectory
```

这会保存完整 `latents` 和 `next_latents`，文件会明显变大。

## 应该检查什么

阶段 2 跑通后，重点看 `stage2_metrics.json` 里的：

1. `trajectory.num_steps`
   - 应等于 `--num-inference-steps`。

2. `trajectory.old_log_probs_shape`
   - 应类似 `[num_steps, num_pbr * num_view]`。
   - 6 视角、albedo/MR 两类时通常是 `[steps, 12]`。

3. `old_log_probs_mean/std/min/max`
   - 必须是有限数值，不能是 NaN / inf。

4. `latents_shape` / `next_latents_shape`
   - 应类似 `[num_steps, 12, 4, H/8, W/8]`。
   - `resolution=256` 时 latent 空间通常是 `32 x 32`。

5. reward / advantage
   - 和阶段 1 一样，用来确认组内相对信号还存在。

## 阶段 2 不做什么

阶段 2 仍然不做：

1. 不重新计算 new log-prob。
2. 不计算 ratio。
3. 不做 clip objective。
4. 不做 KL penalty。
5. 不更新 UNet / LoRA / learned token。

这些属于阶段 3。

## 和阶段 3 的连接

阶段 3 会读取或在线生成阶段 2 的这些数据，然后在当前模型下重新计算：

```text
new_log_prob = log p_theta_new(next_latents | latents, timestep, condition)
ratio = exp(new_log_prob - old_log_prob)
loss = -min(ratio * advantage, clip(ratio) * advantage)
```

所以阶段 2 的成功标准是：MaterialMVP 能够稳定产生 trajectory、old log-prob、reward 和 advantage。只要这四个东西 shape 正确、数值有限，就可以进入真正 GRPO loss 实现。
