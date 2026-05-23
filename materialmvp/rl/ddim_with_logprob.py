import math
from typing import Optional, Tuple

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


def _as_index_tensor(value, device):
    if torch.is_tensor(value):
        value = value.to(device=device, dtype=torch.long)
        if value.ndim == 0:
            value = value[None]
        return value
    return torch.tensor([int(value)], device=device, dtype=torch.long)


def _left_broadcast(tensor, shape):
    if tensor.ndim == 0:
        tensor = tensor[None]
    return tensor.reshape(tensor.shape + (1,) * (len(shape) - tensor.ndim)).broadcast_to(shape)


def _get_variance(scheduler: DDIMScheduler, timestep, prev_timestep, device):
    timestep = _as_index_tensor(timestep, device)
    prev_timestep = _as_index_tensor(prev_timestep, device)
    prev_timestep_clamped = prev_timestep.clamp(0, scheduler.config.num_train_timesteps - 1)

    alpha_prod_t = scheduler.alphas_cumprod.gather(0, timestep.cpu()).to(device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep_clamped.cpu()),
        scheduler.final_alpha_cumprod,
    ).to(device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    eta: float = 1.0,
    generator: Optional[torch.Generator] = None,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(scheduler, DDIMScheduler):
        raise TypeError("ddim_step_with_logprob expects a diffusers DDIMScheduler.")
    if scheduler.num_inference_steps is None:
        raise ValueError("Call scheduler.set_timesteps(...) before ddim_step_with_logprob.")
    if eta <= 0:
        raise ValueError("eta must be > 0 to compute a non-degenerate DDIM transition log-prob.")

    sample_dtype = sample.dtype
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    device = sample.device
    timestep_tensor = _as_index_tensor(timestep, device)
    prev_timestep = timestep_tensor - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep_clamped = prev_timestep.clamp(0, scheduler.config.num_train_timesteps - 1)

    alpha_prod_t = scheduler.alphas_cumprod.gather(0, timestep_tensor.cpu()).to(device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep_clamped.cpu()),
        scheduler.final_alpha_cumprod,
    ).to(device)
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape)
    beta_prod_t = 1 - alpha_prod_t

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    else:
        raise ValueError(f"Unsupported DDIM prediction_type: {prediction_type}")

    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )

    variance = _get_variance(scheduler, timestep_tensor, prev_timestep, device).clamp_min(0)
    std_dev_t = eta * variance.sqrt()
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).float().clamp_min(1e-5)

    direction_scale = (1 - alpha_prod_t_prev - std_dev_t**2).clamp_min(0).sqrt()
    pred_sample_direction = direction_scale * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=torch.float32,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise
    elif generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    normalizer = torch.sqrt(torch.as_tensor(2.0 * math.pi, device=device, dtype=torch.float32))
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(normalizer)
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    log_prob = torch.nan_to_num(log_prob, nan=-1e4, posinf=1e4, neginf=-1e4)
    return prev_sample.to(sample_dtype), log_prob
