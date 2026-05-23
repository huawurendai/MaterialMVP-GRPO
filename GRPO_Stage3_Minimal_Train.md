# 原版 MaterialMVP 直接做 GRPO：阶段 3 说明

## 阶段 3 是什么

阶段 3 是第一个真正会更新参数的最小 GRPO 训练闭环。对应脚本：

```bash
python grpo_stage3_train.py --base cfgs/v1_lora_test.yaml --group-size 2 --max-samples 1 --max-updates 1 --resolution 256 --num-inference-steps 8 --eta 1.0 --dtype fp16
```

它从原版 MaterialMVP / Hunyuan PaintPBR 权重启动，不要求先做 SFT。脚本会现场注入一个新的 LoRA adapter，并直接用 GRPO 更新：

- LoRA 参数
- `learned_text_clip_albedo`
- `learned_text_clip_mr`
- `learned_text_clip_ref`

默认不全量训练 UNet，主要是为了 5080 这类 16GB 显存机器能先跑通。

## 阶段 3 做了什么

每个 update 执行：

1. 读取 clean sample。
2. 对同一 condition 采样 `group_size` 个候选。
3. 记录 DDIM trajectory 和 `old_log_probs`。
4. 计算 albedo / MR reward。
5. 做 group advantage：

```text
advantage_i = (reward_i - mean(reward_group)) / (std(reward_group) + eps)
```

6. 用当前模型重新计算 `new_log_probs`。
7. 计算 GRPO/PPO clipped loss：

```text
ratio = exp(new_log_prob - old_log_prob)
loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

8. backward 更新 LoRA / learned token。
9. 保存 LoRA checkpoint。

## 推荐首跑命令

先跑最小配置：

```bash
python grpo_stage3_train.py \
  --base cfgs/v1_lora_test.yaml \
  --group-size 2 \
  --max-samples 1 \
  --max-updates 1 \
  --resolution 256 \
  --num-inference-steps 8 \
  --eta 1.0 \
  --learning-rate 1e-5 \
  --clip-range 0.2 \
  --dtype fp16
```

如果显存紧，降低：

```bash
--num-inference-steps 4
```

如果想让训练信号更稳定，跑通后再试：

```bash
--group-size 4
```

## 输出内容

默认输出目录：

```text
outputs/grpo_stage3/
```

会保存：

- `update_xxxx_sample_xxxx/stage3_metrics.json`
- `update_xxxx_sample_xxxx/group_00_albedo_mr_before.png`
- `checkpoints/lora_grpo_step_000001.pt`
- `checkpoints/lora_grpo_last.pt`
- `summary.json`

## 应该检查什么

首跑重点看：

1. `total_loss` 是否是有限值。
2. `grad_norm` 是否是有限值且非 0。
3. `ratio_mean` 是否接近 1。
4. `approx_kl` 初始是否接近 0。
5. `clipfrac` 是否不要一开始就接近 1。
6. checkpoint 是否正常保存。

第一步 update 时，因为 `old_log_probs` 和 `new_log_probs` 来自同一模型，`ratio_mean` 通常会接近 1，这是正常的。梯度仍然存在，因为 loss 对 `new_log_prob` 有梯度。

## 当前阶段的边界

这版阶段 3 是“最小可训练闭环”，还不是完整长训框架：

- 没有 DDP / Lightning。
- 没有周期性 validation。
- reward 仍然是简单 albedo/MR L1 + range penalty。
- 没有 KL reference model。
- 默认不全量训练 UNet。
- 每个 update 现采样、现训练，适合先验证而不是大规模训练。

如果阶段 3 首跑稳定，下一步可以升级为阶段 4：多样本循环、周期性评估、更多 PBR reward、resume checkpoint、以及更完整的日志。
