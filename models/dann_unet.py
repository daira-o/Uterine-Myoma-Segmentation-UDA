"""
DANN wrapper around the existing Attention U-Net segmenter.

Phase 2 flow:
    image -> AttentionUNetDANN.encode() -> bottleneck features
          -> AttentionUNetDANN.decode() -> segmentation logits
          -> GRL -> DomainDiscriminator -> domain logits
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from models.attention_unet_dann import AttentionUNetDANN
from models.domain_discriminator import (
    DomainDiscriminator,
    MultiScaleDomainDiscriminator,
)
from models.grl import GradientReversalLayer, grad_reverse


FeatureMode = Literal["bottleneck", "multiscale"]


class DANNUNet(nn.Module):
    """
    Attention U-Net plus Gradient Reversal Layer and Domain Discriminator.

    Args:
        in_channels: Number of U-Net input channels.
        num_classes: Number of segmentation output channels.
        base_filters: Base channel count in the Attention U-Net.
        discriminator_hidden_dim: Hidden size in the domain discriminator.
        discriminator_dropout: Dropout in the domain discriminator.
        feature_mode: "bottleneck" is recommended; "multiscale" uses
            enc3, enc4, and bottleneck features.

    Forward:
        seg_logits, domain_logits = model(x, alpha=alpha)

    For target/US batches without masks:
        _, domain_logits = model(x_us, alpha=alpha, return_segmentation=False)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        discriminator_hidden_dim: int = 512,
        discriminator_dropout: float = 0.5,
        feature_mode: FeatureMode = "bottleneck",
    ) -> None:
        super().__init__()
        self.base_filters = base_filters
        self.feature_mode = feature_mode

        self.segmenter = AttentionUNetDANN(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
        )
        self.grl = GradientReversalLayer(lambda_init=0.0)

        bottleneck_channels = base_filters * 16
        if feature_mode == "bottleneck":
            self.domain_discriminator: nn.Module = DomainDiscriminator(
                in_channels=bottleneck_channels,
                hidden_dim=discriminator_hidden_dim,
                num_domains=2,
                dropout_rate=discriminator_dropout,
            )
        elif feature_mode == "multiscale":
            self.domain_discriminator = MultiScaleDomainDiscriminator(
                channels_list=[base_filters * 4, base_filters * 8, bottleneck_channels],
                num_domains=2,
                dropout_rate=discriminator_dropout,
            )
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    def _domain_logits(
        self,
        features: dict[str, torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        """Apply GRL to deep features and predict the input domain."""
        self.grl.set_lambda(alpha)
        if self.feature_mode == "bottleneck":
            reversed_features = self.grl(features["bottleneck"])
            return self.domain_discriminator(reversed_features)

        reversed_features = [
            grad_reverse(features["enc3"], lambda_=alpha),
            grad_reverse(features["enc4"], lambda_=alpha),
            grad_reverse(features["bottleneck"], lambda_=alpha),
        ]
        return self.domain_discriminator(reversed_features)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        return_segmentation: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Return segmentation logits and domain logits.

        When `return_segmentation=False`, the decoder is skipped. This is useful
        for ultrasound batches used only for domain supervision. The encoder
        still receives adversarial gradients through the GRL.
        """
        features = self.segmenter.encode(x)
        seg_logits = self.segmenter.decode(features) if return_segmentation else None
        domain_logits = self._domain_logits(features, alpha=alpha)
        return seg_logits, domain_logits

    def load_pretrained_segmenter(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Load phase-1 segmenter weights into `self.segmenter`.

        Supports plain state_dict checkpoints and dictionaries with common keys:
        `model_state_dict`, `state_dict`, or `segmenter_state_dict`.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("segmenter_state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        cleaned_state = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module.") :]
            if clean_key.startswith("segmenter."):
                clean_key = clean_key[len("segmenter.") :]
            cleaned_state[clean_key] = value

        incompatible = self.segmenter.load_state_dict(cleaned_state, strict=strict)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
