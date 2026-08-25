from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DINOV2_CONFIGS = {
    "small": dict(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        mlp_ratio=4,
        image_size=518,
        patch_size=14,
    ),
    "base": dict(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        image_size=518,
        patch_size=14,
    ),
    "large": dict(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        image_size=518,
        patch_size=14,
    ),
}
DINOV2_HUB_IDS = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
}


class ResNet50SlotEncoder(nn.Module):

    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        backbone = torchvision.models.resnet50(weights=weights)
        self.body = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        feat = self.body(x).flatten(1)
        return self.proj(feat)


class RadImageNetResNet50SlotEncoder(nn.Module):

    def __init__(self, out_dim=512, weights_path=None, unfreeze_last_frac=0.2):
        super().__init__()
        if not weights_path:
            raise FileNotFoundError(
                "RadImageNet ONNX weights required - convert RadImageNet-ResNet50_notop.h5 via tf2onnx + onnx2torch (see yosukeyama/onnx-convert-radimagenet-to-pth for the reference recipe), then pass the resulting .onnx file's path via --radimagenet-weights."
            )
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"RadImageNet ONNX weights not found at {weights_path} - expected a .onnx file produced by the tf2onnx + onnx2torch conversion, mounted as a Kaggle dataset input."
            )
        from onnx2torch import convert as onnx2torch_convert

        tmp_dir = Path(tempfile.mkdtemp())
        tmp_weights_path = tmp_dir / weights_path.name
        shutil.copy(weights_path, tmp_weights_path)
        self.body = onnx2torch_convert(str(tmp_weights_path))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, out_dim)
        self._frozen_bn_modules = []
        if unfreeze_last_frac is not None and unfreeze_last_frac < 1.0:
            body_params = list(self.body.named_parameters())
            total_numel = sum((p.numel() for _, p in body_params))
            target_trainable = int(round(total_numel * unfreeze_last_frac))
            unfreeze_names = set()
            cum = 0
            for name, p in reversed(body_params):
                if cum >= target_trainable:
                    break
                unfreeze_names.add(name)
                cum += p.numel()
            for name, p in body_params:
                p.requires_grad = name in unfreeze_names
            for m in self.body.modules():
                if hasattr(m, "running_mean") and hasattr(m, "running_var"):
                    own_params = list(m.parameters(recurse=False))
                    if own_params and all((not p.requires_grad for p in own_params)):
                        m.eval()
                        self._frozen_bn_modules.append(m)
            n_trainable_tensors = sum((p.requires_grad for _, p in body_params))
            n_trainable_params = sum(
                (p.numel() for _, p in body_params if p.requires_grad)
            )
            print(
                f"[INFO] RadImageNetResNet50SlotEncoder: unfreeze_last_frac={unfreeze_last_frac} -> {n_trainable_tensors}/{len(body_params)} body param tensors trainable ({n_trainable_params / 1000000.0:.1f}M/{total_numel / 1000000.0:.1f}M params, {n_trainable_params / total_numel:.1%}), {len(self._frozen_bn_modules)} BN layer(s) pinned to eval mode",
                flush=True,
            )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in self._frozen_bn_modules:
                m.eval()
        return self

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        feat = self.body(x)
        feat = feat.permute(0, 3, 1, 2)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)


def find_dinov2_checkpoint(variant="small", search_root="/kaggle/input"):
    root = Path(search_root)
    if not root.is_dir():
        return None
    hits = []
    for path in root.rglob("config.json"):
        if "dinov2" in str(path.parent).lower():
            hits.append(path.parent)
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None


class Dinov2SlotEncoder(nn.Module):

    def __init__(
        self,
        out_dim=512,
        pretrained=True,
        variant="small",
        unfreeze_last=2,
        source=None,
    ):
        super().__init__()
        from transformers import AutoModel, Dinov2Config, Dinov2Model

        if pretrained:
            path = source or find_dinov2_checkpoint(variant) or DINOV2_HUB_IDS[variant]
            self.backbone = AutoModel.from_pretrained(str(path))
        else:
            cfg = Dinov2Config(**DINOV2_CONFIGS[variant])
            self.backbone = Dinov2Model(cfg)
        n_layers = len(self.backbone.encoder.layer)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[max(0, n_layers - unfreeze_last) :]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        cls_token = out.last_hidden_state[:, 0]
        return self.proj(cls_token)


def build_encoder(backbone, out_dim=512, pretrained=True, **kwargs):
    if backbone == "resnet50":
        return ResNet50SlotEncoder(out_dim=out_dim, pretrained=pretrained)
    if backbone == "radimagenet_resnet50":
        return RadImageNetResNet50SlotEncoder(
            out_dim=out_dim,
            weights_path=kwargs.get("radimagenet_weights"),
            unfreeze_last_frac=kwargs.get("unfreeze_last_frac", 0.2),
        )
    if backbone == "dinov2":
        return Dinov2SlotEncoder(out_dim=out_dim, pretrained=pretrained, **kwargs)
    raise ValueError(
        f"unknown backbone {backbone!r}, expected 'resnet50', 'radimagenet_resnet50', or 'dinov2'"
    )


class KneeMRIModel(nn.Module):

    def __init__(
        self,
        n_labels=12,
        feat_dim=512,
        backbone="resnet50",
        pretrained=True,
        train_img_size=224,
        backbone_kwargs=None,
    ):
        super().__init__()
        self.encoder = build_encoder(
            backbone, out_dim=feat_dim, pretrained=pretrained, **backbone_kwargs or {}
        )
        self.backbone_name = backbone
        self.train_img_size = train_img_size
        self.norm_mode = (
            "scale_pm1" if backbone == "radimagenet_resnet50" else "imagenet"
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, n_labels),
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, slots, mask):
        B, S, C, H, W = slots.shape
        x = slots.reshape(B * S, C, H, W).float() / 255.0
        if x.shape[-1] != self.train_img_size:
            x = F.interpolate(
                x,
                size=(self.train_img_size, self.train_img_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.norm_mode == "scale_pm1":
            x = x * 2.0 - 1.0
        else:
            x = (x - self.mean) / self.std
        feat = self.encoder(x).reshape(B, S, -1)
        mask_f = mask.unsqueeze(-1)
        mean_feat = (feat * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        neg_inf = torch.finfo(feat.dtype).min
        masked_for_max = feat.masked_fill(mask_f == 0, neg_inf)
        max_feat = masked_for_max.max(1).values
        no_slots = (mask.sum(1) == 0).unsqueeze(-1)
        max_feat = torch.where(no_slots, torch.zeros_like(max_feat), max_feat)
        pooled = torch.cat([mean_feat, max_feat], dim=-1)
        return self.classifier(pooled)
