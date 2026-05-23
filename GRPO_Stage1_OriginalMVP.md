# 原版 MaterialMVP 直接做 GRPO：阶段 1 说明

## 阶段 1 是什么

阶段 1 是一个 **rollout + reward + group advantage 的 smoke test**，对应新增脚本：

```bash
python grpo_stage1_smoketest.py --base cfgs/v1_lora_test.yaml --group-size 2 --max-samples 1 --resolution 256 --num-inference-steps 8
```

它使用原版 MaterialMVP / Hunyuan PaintPBR 模型直接采样，不先做 SFT，也不加载 LoRA 微调权重。这个阶段不会创建 optimizer，不会 backward，不会保存训练 checkpoint，因此不会修改模型参数。

默认情况下，脚本会直接从样本目录读取原始 `render_cond` / `render_tex` 文件，不走训练 dataset 的随机增广。这一点很重要：normal / position 是几何条件，如果使用训练 loader 的随机旋转、缩放或透视增广，输入给 pipeline 的几何条件会和正常 mesh 渲染路径不一致，生成质量可能明显变差。只有需要复现训练 loader 行为时，才使用 `--use-dataset-augmentation`。

## 为什么要先做阶段 1

直接上完整 GRPO 风险比较高，因为 MaterialMVP 不是普通 text-to-image，它一次 rollout 会同时生成多视角 albedo 和 MR，并且依赖 reference 图、normal 图、position 图、DINO 特征、multi-view attention、reference attention 和自定义 CFG。只要其中任何一环 shape 或 dtype 不对，后面的 log-prob / PPO ratio / advantage 训练都会变成难定位的问题。

阶段 1 的作用就是先把 RL 前半段钉牢：

1. 同一个条件能否生成一组候选结果。
2. 每个候选能否拆成 `albedo` 和 `mr` 两组多视角图。
3. 生成图能否和数据集里的 GT albedo / MR 对齐计算 reward。
4. 同组 reward 是否能计算 mean / std / advantage。
5. 输出可视化和 JSON 日志是否足够后续排查。

## 阶段 1 能验证什么

### 1. 原版 MVP rollout 是否跑通

它会从数据集中读取：

- `images_cond`
- `images_normal`
- `images_position`
- `images_albedo`
- `images_mr`

然后用原版 pipeline 按同一 condition 生成 `group_size` 个候选。每个候选保存一张 contact sheet，第一行是 albedo 多视角，第二行是 MR 多视角。

如果这里失败，说明问题在模型加载、条件构造、DINO、scheduler、显存或 pipeline 输入输出，而不是 GRPO loss。

### 2. group sampling 是否真的有差异

DanceGRPO / GRPO 的关键前提是：同一 prompt / condition 下要有一组不同候选，reward 才能做组内归一化。

阶段 1 会为每个 group candidate 使用不同 seed。如果 `reward_std` 长期接近 0，要么采样没有差异，要么 reward 太迟钝，这两种情况都会让 GRPO advantage 失效。

### 3. reward 计算是否有信号

当前阶段 1 使用最简单的可验证 reward：

```text
reward = -(0.6 * albedo_l1 + 0.4 * mr_l1) - 0.05 * range_penalty
```

它不是最终 reward，只是用来验证：

- 生成 albedo 是否能和 GT albedo 比较。
- 生成 MR 是否能和 GT MR 比较。
- PBR 结果是否出现大面积饱和。
- reward 是否是有限数值，不是 NaN / inf。

### 4. advantage 是否可计算

阶段 1 会对同一条件下的 group reward 做：

```text
advantage_i = (reward_i - mean(reward_group)) / (std(reward_group) + eps)
```

这一步对应真正 GRPO 训练里最核心的组内相对排序。阶段 1 不训练，但会提前验证 advantage 的数值稳定性。

### 5. 后续 log-prob 训练需要的数据边界

阶段 1 暂时不记录 diffusion trajectory 和 log-prob。它先确认最终图像级 rollout/reward 闭环。下一阶段才应该参考 DanceGRPO 的 `pipeline_with_logprob.py` / `ddim_with_logprob.py`，把 MaterialMVP 的 denoising loop 改成可返回：

- `latents_t`
- `latents_{t-1}`
- `timesteps`
- `old_log_probs`
- `reward`
- `advantage`

## 阶段 1 不验证什么

阶段 1 不验证以下内容：

1. 不验证 policy gradient 是否正确。
2. 不验证 DDIM / DDPM step log-prob。
3. 不验证 PPO / GRPO ratio clipping。
4. 不验证 KL penalty。
5. 不验证训练是否能提升 reward。
6. 不验证 LoRA 或全量 UNet 参数更新。

这些属于阶段 2 和阶段 3。

## 输出内容

默认输出目录是：

```text
outputs/grpo_stage1/
```

每个样本会生成：

- `condition.png`：输入 reference condition。
- `target_albedo_mr.png`：用于 reward 对齐的 GT albedo / MR。
- `group_00_albedo_mr.png`：第 0 个候选的 albedo / MR 多视角结果。
- `group_01_albedo_mr.png`：第 1 个候选。
- `stage1_metrics.json`：当前样本的 reward、advantage 和每个 group 的指标。

总目录下会生成：

- `summary.json`：所有样本的阶段 1 汇总。

## 成功标准

阶段 1 跑通后，至少应满足：

1. 原版模型能正常加载并生成 albedo / MR。
2. 每个 group 都能保存可视化图。
3. `stage1_metrics.json` 中 reward 是有限值。
4. `reward_std` 不是长期严格为 0。
5. albedo / MR 输出数量等于 `num_view`。
6. 显存不会在小配置下爆掉，例如 `resolution=256`、`num_inference_steps=8`、`group_size=2`。

如果这些都成立，下一步才值得进入阶段 2：给 MaterialMVP pipeline 增加 DDIM/DDPM log-prob trajectory，并开始实现真正的 GRPO loss。
