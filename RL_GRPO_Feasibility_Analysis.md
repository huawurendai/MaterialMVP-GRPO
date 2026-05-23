# MaterialMVP 接入 GRPO / DanceGRPO 的可行性分析

本文基于当前工作区 `D:\Code\MaterialMVP` 的源码阅读结果，重点覆盖 `train.py`、`materialmvp/model.py`、数据加载器、validation/inference pipeline，以及当前 albedo / MR / normal / position 等 PBR 结果的生成方式。本文只做分析，不涉及训练代码修改。

参考背景：DanceGRPO 官方说明将 GRPO 扩展到视觉生成任务，核心思想是对同一条件采样一组结果，用 reward 做组内归一化 advantage，再用扩散轨迹上的 policy log-prob 做近端策略优化；其官方实现已支持 Stable Diffusion / FLUX / HunyuanVideo 等视觉生成模型，并强调 timestep 选择、噪声初始化、噪声尺度和 CFG 下的梯度累积会显著影响稳定性。

## 1. 当前 MaterialMVP 的训练流程梳理

当前训练入口是 `train.py`，整体使用 PyTorch Lightning：

1. 读取 `--base` YAML 配置，例如 `cfgs/v1.yaml`、`cfgs/v1_lora_1000.yaml`。
2. 通过 `src.utils.train_util.instantiate_from_config` 实例化 `materialmvp.model.MaterialMVP`。
3. 从 `stable_diffusion_config` 加载 diffusers `DiffusionPipeline`，自定义 pipeline 为 `./materialmvp`。
4. 如果底层 UNet 是 `UNet2DConditionModel`，包装为 `UNet2p5DConditionModel`，使其支持多视角、多 PBR token、reference attention、multi-view attention、DINO 条件和 position RoPE。
5. 训练 scheduler 使用 `DDPMScheduler`，validation / inference scheduler 通常使用 `EulerAncestralDiscreteScheduler` 或 `UniPCMultistepScheduler`。
6. 数据模块 `src.data.objaverse_hunyuan.DataModuleFromConfig` 构建 train / validation dataset，并用 `ConcatDataset + DistributedSampler` 提供 batch。
7. `MaterialMVP.training_step` 执行监督扩散训练：
   - 从 batch 取 reference 条件图、目标 albedo / MR、normal / position。
   - VAE 编码目标 PBR 图，得到 latent。
   - 对目标 latent 加随机噪声。
   - UNet 预测 `v_prediction` 或 `epsilon`。
   - 对 albedo、MR 分别做 MSE，再加双 reference 条件下的 consistency loss。
8. checkpoint / logger / metrics：
   - Lightning checkpoint 默认保存完整模型。
   - 当前项目已加入 LoRA 支持，可额外保存 `lora_step_xxx.pt` 和 `lora_last.pt`。
   - `MetricsCSVCallback` 记录 `train/total_loss`、`train/albedo_loss`、`train/mr_loss`、`train/cons_loss`。

目前训练是典型监督 fine-tuning，不是 RL：没有 rollout 多样本组、没有 reward、没有 advantage、没有策略比率、没有 KL/reference policy 约束。

## 2. 当前模型输入输出是什么

### 训练 batch 输入

数据加载器主要有两个：

- `src/data/dataloader/objaverse_loader_forTexturePBR.py`
- `src/data/dataloader/objaverse_loader_forTexturePBR_paperpair.py`

它们返回的核心字段基本一致：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `images_cond` | `[B, 2, 3, H, W]` | 两张 reference / condition 渲染图，通常为同物体不同光照或邻近视角 |
| `images_albedo` | `[B, N, 3, H, W]` | N 个视角的 albedo GT |
| `images_mr` | `[B, N, 3, H, W]` | N 个视角的 metallic-roughness GT |
| `images_normal` | `[B, N, 3, H, W]` | N 个视角 normal 条件 |
| `images_position` | `[B, N, 3, H, W]` | N 个视角 position 条件 |
| `name` | list / str | 样本目录 |

其中 `N` 通常是 `num_view=6`。`TextureDatasetPaperPair` 比原始 `TextureDataset` 更适合 RL，因为它会根据 `transforms.json` 采样 reference pair 和目标纹理视角，且对相邻方位与点光源概率有更明确的控制。

### 模型条件输入

`MaterialMVP.prepare_batch_data` 会把输入 resize 到 `view_size`，并组织为：

- `cond_imgs`：第一张 reference 图，用于 reference latent 和 DINO 条件。
- `cond_imgs_another`：第二张 reference 图，用于 consistency loss。
- `target_imgs["albedo"]` / `target_imgs["mr"]`：监督目标。
- `normal_imgs`：normal 条件，会 VAE 编码为 `embeds_normal`。
- `position_imgs`：position 条件，会 VAE 编码为 `embeds_position`，同时原图作为 `position_maps` 供 position RoPE 计算体素索引。

### UNet 输入输出

`UNet2p5DConditionModel.forward` 的主要输入是：

- `sample`：形状 `[B, N_pbr, N_view, C, h, w]` 的 noisy latent。
- `encoder_hidden_states`：每个 PBR 类型对应的 learned text token，例如 `learned_text_clip_albedo`、`learned_text_clip_mr`。
- `ref_latents`：reference 图 VAE latent。
- `dino_hidden_states`：DINOv2 提取的 reference 图语义特征。
- `embeds_normal` / `embeds_position`：normal / position 的 VAE latent。
- `position_maps`：用于 position RoPE 的原始 position 图。
- `mva_scale` / `ref_scale`：训练时随机关闭 multi-view attention 或 reference attention。

UNet 输出为同 shape 的预测噪声或速度项，当前主要使用 `v_prediction`：

- 训练输出：`v_pred`，再拆成 albedo 和 MR 两组。
- 推理输出：scheduler 迭代后的 latent，经 VAE decode 变回图像。

### 推理输出

`utils/multiview_utils.py` 中 `multiviewDiffusionNet.forward_one` 调用自定义 pipeline 后，将输出图像按顺序拆分：

```python
mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
```

也就是说当前模型直接生成的是多视角 albedo 图和多视角 MR 图。normal / position 并不是模型生成结果，而是由 mesh renderer 根据几何渲染出来，作为条件输入。

## 3. 当前 loss 是什么

当前核心 loss 在 `materialmvp/model.py::training_step` 中：

1. 随机 timestep `t`。
2. 对 albedo / MR GT latent 添加噪声。
3. UNet 对第一张 reference 条件预测 `v_pred`。
4. UNet 对第二张 reference 条件再次预测 `v_pred_another`。
5. 计算：
   - `albedo_loss = 0.5 * (MSE(v_pred_albedo, v_target_albedo) + MSE(v_pred_another_albedo, v_target_albedo))`
   - `mr_loss = 0.5 * (MSE(v_pred_mr, v_target_mr) + MSE(v_pred_another_mr, v_target_mr))`
   - `consistency_loss = MSE(v_pred_another, v_pred)`
6. 总 loss：

```python
total_loss = 0.85 * (albedo_loss + mr_loss) + 0.15 * consistency_loss
```

这说明当前 fine-tuning 目标是“给定 reference、normal、position 条件，在扩散噪声预测空间回归 GT PBR latent”，同时要求不同 reference 光照下预测一致。

注意一个源码细节：`training_step` 中做 condition dropout 时判断了 `"normal_imgs"` / `"position_imgs"`，但实际缓存键是 `"embeds_normal"` / `"embeds_position"`。因此 normal / position embedding 的 dropout 分支看起来不会按预期触发，只有 `position_maps` 的 dropout 分支可能触发。这个问题不需要现在修，但后续 RL 训练设计要注意条件 dropout 的真实行为。

## 4. 哪些模块适合接入 GRPO / DanceGRPO

最适合接入的位置不是直接替换当前 `training_step`，而是新增一条 RL fine-tuning 训练路径。建议按优先级分层：

### 最适合接入

1. `materialmvp/pipeline.py::denoise`
   - 这里有完整推理采样循环。
   - GRPO 需要记录每一步 latent、timestep、noise prediction、scheduler transition log-prob。
   - 需要新增“可训练 rollout”版本，不能长期依赖 `@torch.no_grad()` 的推理路径。

2. `materialmvp/model.py`
   - 适合新增 `grpo_training_step` 或单独 `MaterialMVPGRPO` LightningModule。
   - 可复用 `prepare_batch_data`、`encode_images`、`forward_unet`、LoRA 参数管理和 optimizer。

3. 新增 `rl/rewards.py`
   - reward 与模型主体解耦。
   - 支持图像级、多视角级、PBR 物理约束、mesh bake 后渲染级 reward。

4. 新增 `train_grpo.py`
   - 保持当前监督训练入口稳定。
   - GRPO 对 rollout、reference policy、KL、group size、采样步数、reward logging 的控制更多，单独入口更清晰。

### 可以接入但不建议第一阶段动太多

1. `textureGenPipeline.py`
   - 这里包含完整 mesh -> normal/position render -> multiview PBR -> super-res -> bake -> inpaint -> save mesh 的产品级流程。
   - 可用于离线评估或高质量 reward，但训练内直接调用成本很高。

2. `utils/pipeline_utils.py::ViewProcessor`
   - 适合做 bake / UV consistency reward。
   - 最小实现阶段建议先只做图像级和多视角级 reward，避免训练吞吐崩掉。

3. `materialmvp/modules.py`
   - 2.5D attention 结构本身不需要为了 GRPO 改。
   - 更建议只训练 LoRA 或少量 learned token，降低策略漂移风险。

## 5. 可以设计哪些 reward

建议把 reward 分为“便宜、稳定、可训练早期使用”和“昂贵、接近最终目标、后期或离线使用”两类。

### 便宜且适合最小实现

1. GT 图像相似度 reward
   - 有监督数据时可直接比较生成 albedo / MR 与 GT。
   - 可用 `-L1`、`-MSE`、`SSIM`、`LPIPS` 的加权组合。
   - 优点是稳定；缺点是更像 RL 形式包装的监督训练。

2. Latent-space reward
   - 比较生成图 VAE latent 与 GT latent。
   - 成本低，和当前训练目标更接近。
   - 可作为 warmup reward。

3. Reference 保真 reward
   - 生成 albedo 在语义和颜色上应接近输入 reference 的材质外观。
   - 可用 DINO / CLIP image embedding cosine similarity。
   - 对没有完整 GT 的数据更有用。

4. PBR 值域与通道合理性 reward
   - albedo 不应大面积过曝、全灰、全黑。
   - MR 的 metallic / roughness 通道应在合理分布，避免饱和或塌缩。
   - 可加入均值、方差、直方图、边缘能量约束。

5. 多视角一致性 reward
   - 用 position map 或可见 mask 将相近 3D 区域的预测颜色拉近。
   - 最小版本可以先用相邻视角的 DINO / low-frequency consistency 近似。

### 更接近最终目标但更昂贵

1. Bake 后 UV texture reward
   - 将多视角 albedo / MR bake 回 UV 纹理后检查空洞、接缝、噪声、覆盖率。
   - 对真实产品目标很重要，但训练中调用 renderer 成本较高。

2. Re-render reward
   - 用生成的 PBR 贴图在若干光照下重新渲染，与 reference 或 held-out render 对比。
   - 最符合“材质正确性”，但需要稳定、快速的渲染管线。

3. Illumination-invariance reward
   - 同一材质在不同输入光照 reference 下，生成的 albedo / MR 应一致。
   - 当前 supervised consistency loss 的 RL 版本可以设计为组内或双条件 reward。

4. Human / preference reward
   - 用人工 pairwise preference 或训练 reward model。
   - 更贴近 GRPO / DanceGRPO 的优势，但需要额外标注或偏好模型。

5. Aesthetic / quality reward
   - 可参考 DanceGRPO 常见的 HPS、CLIP、PickScore 等图像 reward。
   - 对 MaterialMVP 不能直接照搬，应降低权重，因为 PBR 贴图不是普通自然图像，美观 reward 可能鼓励错误高光、阴影或纹理幻觉。

### 推荐 reward 初始组合

最小可行实现建议：

```text
reward =
  0.40 * albedo_gt_similarity
+ 0.25 * mr_gt_similarity
+ 0.15 * reference_dino_similarity
+ 0.10 * multiview_consistency
+ 0.10 * pbr_range_regularization
```

若没有 GT，则改为：

```text
reward =
  0.35 * reference_dino_similarity
+ 0.25 * illumination_invariance
+ 0.20 * multiview_consistency
+ 0.20 * pbr_range_regularization
```

## 6. 需要新增哪些脚本和配置

建议新增：

1. `train_grpo.py`
   - 独立 RL 训练入口。
   - 加载 base / LoRA checkpoint。
   - 构建 rollout policy、frozen reference policy、reward functions、GRPO optimizer。

2. `cfgs/rl_grpo_lora.yaml`
   - 包含 group size、rollout steps、采样 scheduler、reward 权重、KL 系数、clip range、LoRA 配置。

3. `materialmvp/rl/rollout.py`
   - 从当前 pipeline 抽出可训练采样循环。
   - 返回 generated images / latents、timesteps、old log-probs、model predictions。

4. `materialmvp/rl/rewards.py`
   - 实现 albedo / MR / consistency / DINO / PBR range reward。

5. `materialmvp/rl/grpo_loss.py`
   - 实现组内 reward normalize、advantage、ratio clipping、KL penalty。

6. `materialmvp/rl/eval.py`
   - 固定 cases 评估 base vs RL checkpoint。
   - 可复用 `demo_lora_compare.py` 的比较思路。

7. `scripts/run_grpo_smoketest.ps1` 或简单命令说明
   - 用 `materialmvp_dataset_test1` 跑极小步数，验证显存、shape、日志和 checkpoint。

## 7. 需要改哪些源码文件

为了尽量不破坏现有监督训练，建议改动范围如下。

### 必须新增或扩展

1. `materialmvp/pipeline.py`
   - 新增可返回 denoising trajectory 的函数。
   - 新增 log-prob 计算所需的 scheduler transition 信息。
   - 推理用 `__call__` 保持兼容，不改变默认行为。

2. `materialmvp/model.py`
   - 新增 RL 专用 LightningModule，或在 `MaterialMVP` 中新增独立方法。
   - 复用现有条件准备、VAE encode/decode、UNet forward。
   - 加入 frozen reference model 或 frozen old policy 的管理。

3. `materialmvp/lora_utils.py`
   - 现有 LoRA 已基本可用。
   - 可能需要支持 RL checkpoint 中同时保存 LoRA、reward config、KL reference 路径。

4. `src/data/dataloader/*`
   - 最小阶段可不改。
   - 若做 renderer / view-aware reward，建议额外返回 camera metadata、mask、azimuth/elevation、mesh path。

### 建议新增，不改旧文件

1. `materialmvp/rl/`
2. `train_grpo.py`
3. `cfgs/rl_grpo_lora.yaml`
4. `scripts/` 下的 smoke test 和 eval 脚本

### 暂不建议修改

1. `materialmvp/modules.py`
   - 2.5D attention 主干保持稳定。
2. `textureGenPipeline.py`
   - 第一阶段不要把完整 mesh baking 放进训练内。
3. 现有 `train.py`
   - 保持监督训练和 LoRA SFT 可复现。

## 8. 风险和可行性评估

### 可行性

整体可行，尤其适合“LoRA + 小 group size + 图像级 reward + 冻结 reference policy”的路线。当前项目已经有几个有利条件：

- 使用 diffusers pipeline，具备清晰 denoising loop。
- 已有 LoRA 注入和保存机制，便于低风险 RL fine-tuning。
- 数据集已经提供 albedo / MR / normal / position，多种 reward 可以从现有字段直接计算。
- validation pipeline 已能生成多视角 albedo / MR，可作为 rollout 的基础。
- `demo_lora_compare.py` 已经有 base vs LoRA 的外层比较框架。

### 主要风险

1. 扩散 log-prob 实现复杂
   - 当前 inference scheduler 包括 Euler / UniPC，其中 UniPC 偏确定性，不天然提供简单 policy log-prob。
   - GRPO 训练建议使用带随机转移、可计算 log-prob 的 DDPM / DDIM / SDE 采样版本。

2. 训练成本高
   - 一个样本包含 `N_pbr=2`、`N_view=6`，一次生成就是 12 张 latent 图。
   - GRPO group size 如果设为 4，则单 prompt 等价 48 张图的 denoising trajectory。
   - 需要优先用 LoRA、低分辨率、少步数、梯度累积。

3. Reward hacking
   - 普通图像美学 reward 可能鼓励 albedo 中出现阴影、高光或过强纹理，这与 PBR 解耦目标冲突。
   - MR reward 如果只看像素距离，可能导致通道塌缩。

4. 与当前 SFT objective 分布不一致
   - 当前训练在随机 timestep 上做 teacher-forced 噪声预测。
   - GRPO 优化完整或部分 denoising trajectory，梯度分布不同，学习率应显著小于 SFT。

5. CFG 与组内采样的稳定性
   - 当前 pipeline 的 CFG 使用三分支：uncond / ref / full。
   - GRPO 下如果对 CFG 后结果直接计算 log-prob，需要明确 policy 是哪个分支的 noise prediction。
   - 建议最小实现先关闭或固定 CFG 逻辑，或只对 full branch 做近似策略更新。

6. normal / position 是条件不是动作
   - 当前模型不生成 normal / position。
   - 如果想优化 normal / position 质量，必须另建生成头或几何预测模块；这不属于最小 GRPO 范围。

7. Renderer / bake reward 吞吐
   - 完整 `textureGenPipeline.py` 包含 remesh、UV wrap、super-resolution、bake、inpaint 和导出 mesh，不适合作为每步训练 reward。
   - 适合作为周期性 eval 或离线 reward 数据生成。

## 9. 一个最小可行实现方案

建议分四阶段推进。

### 阶段 0：保持当前 SFT 作为初始化

- 使用现有 `train.py + cfgs/v1_lora_1000.yaml` 得到一个 LoRA SFT checkpoint。
- 确认 validation 图中 albedo / MR 基本可用。
- 固定 5-20 个 validation cases 作为 RL 前后比较集。

### 阶段 1：做 RL rollout smoke test，不更新参数

新增 `train_grpo.py` 和 `materialmvp/rl/rollout.py`，先实现：

- 对同一个 batch 条件采样 `group_size=2`。
- 输出 albedo / MR 图。
- 计算简单 reward：
  - `-L1(generated_albedo, gt_albedo)`
  - `-L1(generated_mr, gt_mr)`
  - PBR range penalty
- 打印每个 group 的 reward、均值、标准差、advantage。

这一阶段不需要 log-prob，不做 backward，只验证采样、decode、reward shape 和显存。

### 阶段 2：加入可计算 log-prob 的短步数 DDPM/DDIM 采样

- 使用训练 scheduler 或新增 RL scheduler。
- 采样步数先设 `4-8`，分辨率先用 `256`。
- 记录每一步：
  - `latents_t`
  - `t`
  - `noise_pred`
  - `latents_{t-1}`
  - `log_prob_old`
- 冻结一份 reference policy 计算 KL 或 reference log-prob。
- 只训练 LoRA 参数和 learned text token。

### 阶段 3：实现 GRPO loss

对同一条件的 `G` 个样本：

```text
advantage_i = (reward_i - mean(reward_group)) / (std(reward_group) + eps)
ratio_i,t = exp(log_prob_new_i,t - log_prob_old_i,t)
loss_i,t = -min(
    ratio_i,t * advantage_i,
    clip(ratio_i,t, 1 - eps, 1 + eps) * advantage_i
)
total_loss = grpo_loss + beta_kl * kl_to_reference
```

建议初始超参：

| 参数 | 建议值 |
| --- | --- |
| `group_size` | 2 |
| `num_inference_steps` | 4 或 8 |
| `resolution` | 256 |
| `learning_rate` | `5e-6` 到 `2e-5` |
| `clip_range` | 0.1 或 0.2 |
| `kl_beta` | 0.01 起步 |
| `max_grad_norm` | 0.5 或 1.0 |
| `trainable` | LoRA + learned text tokens |

### 阶段 4：逐步加入 MaterialMVP 专用 reward

按稳定性顺序加入：

1. albedo / MR GT similarity。
2. PBR range / saturation penalty。
3. DINO reference similarity。
4. 多视角一致性 reward。
5. 周期性 bake eval，不进每步训练。
6. 视情况加入 re-render reward 或人工 preference reward。

### 最小实现边界

第一版不要追求完整产品级 mesh reward。最小目标应是：

- 能从 SFT LoRA checkpoint 启动。
- 每个条件生成一组 albedo / MR。
- reward 有组内差异。
- GRPO loss 能稳定 backward。
- 训练 20-100 step 后 reward 日志不 NaN、不塌缩。
- 固定 validation case 的 base vs RL 输出可比较。

如果这个闭环成立，再把 reward 从像素级扩展到 PBR / bake / re-render 级别，风险会小很多。

## 总结判断

MaterialMVP 接入 GRPO / DanceGRPO 的技术路线是可行的，但不建议直接改现有监督 `training_step`。最稳妥的路线是新增独立 RL 训练入口，复用当前模型、LoRA、数据和推理 pipeline，先做低分辨率、短步数、LoRA-only 的 GRPO smoke test。reward 先从有 GT 的 albedo / MR 相似度开始，再逐步加入 reference 保真、多视角一致性、PBR 合理性和 bake/re-render 评价。

normal / position 当前是几何条件，不是模型生成目标；RL 首阶段应优化 albedo / MR 生成质量和跨视角一致性，而不是把 normal / position 纳入 action space。
