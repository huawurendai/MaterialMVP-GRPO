import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


DEFAULT_TARGET_SUFFIXES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_q_mr",
    "to_k_mr",
    "to_v_mr",
    "to_out_mr.0",
)


DEFAULT_EXTRA_TRAINABLE_KEYWORDS = (
    "learned_text_clip_albedo",
    "learned_text_clip_mr",
    "learned_text_clip_ref",
    "image_proj_model_dino.proj",
)


@dataclass
class LoRAInjectReport:
    module_count: int
    module_names: List[str]
    trainable_params: int
    total_params: int
    trainable_names: List[str]
    config: Dict


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_down = nn.Linear(base.in_features, rank, bias=False).to(device=device, dtype=dtype)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False).to(device=device, dtype=dtype)

        for param in self.base.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        return base_out + lora_out


def _as_plain_dict(config) -> Dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "items"):
        return {k: v for k, v in config.items()}
    return {}


def _as_tuple(value, default=()):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def normalize_lora_config(config) -> Dict:
    cfg = _as_plain_dict(config)
    enabled = bool(cfg.get("enabled", False))
    rank = int(cfg.get("rank", 8))
    alpha = int(cfg.get("alpha", rank))
    dropout = float(cfg.get("dropout", 0.0))
    target_suffixes = _as_tuple(cfg.get("target_suffixes"), DEFAULT_TARGET_SUFFIXES)
    include_keywords = _as_tuple(cfg.get("include_keywords"), ())
    exclude_keywords = _as_tuple(cfg.get("exclude_keywords"), ())
    extra_trainable_keywords = _as_tuple(
        cfg.get("extra_trainable_keywords"),
        DEFAULT_EXTRA_TRAINABLE_KEYWORDS if bool(cfg.get("train_extra_params", True)) else (),
    )
    return {
        "enabled": enabled,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_suffixes": target_suffixes,
        "include_keywords": include_keywords,
        "exclude_keywords": exclude_keywords,
        "extra_trainable_keywords": extra_trainable_keywords,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
        "print_trainable": bool(cfg.get("print_trainable", True)),
        "save_extra_trainable": bool(cfg.get("save_extra_trainable", True)),
    }


def lora_enabled(config) -> bool:
    return normalize_lora_config(config)["enabled"]


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _matches_target(
    name: str,
    target_suffixes: Iterable[str],
    include_keywords: Iterable[str] = (),
    exclude_keywords: Iterable[str] = (),
) -> bool:
    target_suffixes = tuple(target_suffixes)
    include_keywords = tuple(include_keywords)
    exclude_keywords = tuple(exclude_keywords)
    if not any(name.endswith(suffix) for suffix in target_suffixes):
        return False
    if include_keywords and not any(keyword in name for keyword in include_keywords):
        return False
    if exclude_keywords and any(keyword in name for keyword in exclude_keywords):
        return False
    return True


def _set_extra_trainable(root: nn.Module, extra_trainable_keywords: Iterable[str]) -> List[str]:
    keywords = tuple(extra_trainable_keywords)
    opened = []
    if not keywords:
        return opened
    for name, param in root.named_parameters():
        if any(keyword in name for keyword in keywords):
            param.requires_grad = True
            opened.append(name)
    return opened


def inject_lora_into_attention(
    root: nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout: float = 0.0,
    target_suffixes: Iterable[str] = DEFAULT_TARGET_SUFFIXES,
    include_keywords: Iterable[str] = (),
    exclude_keywords: Iterable[str] = (),
    extra_trainable_keywords: Iterable[str] = DEFAULT_EXTRA_TRAINABLE_KEYWORDS,
    freeze_first: bool = True,
) -> LoRAInjectReport:
    target_suffixes = tuple(target_suffixes)
    include_keywords = tuple(include_keywords)
    exclude_keywords = tuple(exclude_keywords)
    extra_trainable_keywords = tuple(extra_trainable_keywords)

    if freeze_first:
        freeze_module(root)

    targets = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and _matches_target(name, target_suffixes, include_keywords, exclude_keywords):
            targets.append((name, module))

    injected_names = []
    for name, module in targets:
        parent, child_name = _get_parent_module(root, name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        if child_name.isdigit():
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)
        injected_names.append(name)

    for module in root.modules():
        if isinstance(module, LoRALinear):
            module.lora_down.weight.requires_grad = True
            module.lora_up.weight.requires_grad = True

    _set_extra_trainable(root, extra_trainable_keywords)

    total_params = sum(p.numel() for p in root.parameters())
    trainable = [(name, p) for name, p in root.named_parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for _name, p in trainable)
    return LoRAInjectReport(
        module_count=len(injected_names),
        module_names=injected_names,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_names=[name for name, _p in trainable],
        config={
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_suffixes": list(target_suffixes),
            "include_keywords": list(include_keywords),
            "exclude_keywords": list(exclude_keywords),
            "extra_trainable_keywords": list(extra_trainable_keywords),
        },
    )


def setup_lora_for_model(model, lora_config) -> LoRAInjectReport:
    cfg = normalize_lora_config(lora_config)
    if not cfg["enabled"]:
        return None

    target_unet = model.unet.unet if hasattr(model.unet, "unet") else model.unet
    freeze_module(model.unet)
    if cfg["gradient_checkpointing"] and hasattr(target_unet, "enable_gradient_checkpointing"):
        target_unet.enable_gradient_checkpointing()

    report = inject_lora_into_attention(
        model.unet,
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        target_suffixes=cfg["target_suffixes"],
        include_keywords=cfg["include_keywords"],
        exclude_keywords=cfg["exclude_keywords"],
        extra_trainable_keywords=cfg["extra_trainable_keywords"],
        freeze_first=False,
    )
    model.lora_report = report
    model.lora_config_resolved = report.config
    print(
        "LoRA enabled: "
        f"modules={report.module_count}, "
        f"trainable={report.trainable_params}, "
        f"total={report.total_params}, "
        f"ratio={report.trainable_params / max(report.total_params, 1):.6f}"
    )
    if cfg["print_trainable"]:
        print("Trainable parameters:")
        for name in report.trainable_names:
            param = dict(model.unet.named_parameters())[name]
            print(f"  {name}: shape={tuple(param.shape)}, dtype={param.dtype}, device={param.device}")
    return report


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if ".lora_down." in name or ".lora_up." in name
    }


def adapter_state_dict(module: nn.Module, extra_trainable_keywords: Iterable[str] = ()) -> Dict[str, torch.Tensor]:
    keywords = tuple(extra_trainable_keywords)
    state = {}
    for name, tensor in module.state_dict().items():
        if ".lora_down." in name or ".lora_up." in name or any(keyword in name for keyword in keywords):
            state[name] = tensor.detach().cpu()
    return state


def save_lora_checkpoint(path: str, module: nn.Module, lora_config: Dict):
    torch.save(
        {
            "lora_config": lora_config,
            "state_dict": adapter_state_dict(module, lora_config.get("extra_trainable_keywords", ())),
        },
        path,
    )
