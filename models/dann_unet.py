"""
dann_unet.py
Integracion DANN sobre la Attention U-Net existente.

Fase 2:
    imagen -> AttentionUNet.encode() -> bottleneck features
        -> AttentionUNet.decode() -> segmentation logits
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
    Attention U-Net + Gradient Reversal Layer + Domain Discriminator.

    Args:
        in_channels: canales de entrada de la U-Net.
        num_classes: canales de salida de segmentacion.
        base_filters: filtros base de la Attention U-Net.
        discriminator_hidden_dim: dimension oculta del discriminator.
        discriminator_dropout: dropout del discriminator.
        feature_mode: "bottleneck" recomendado; "multiscale" usa enc3/enc4/bottleneck.

    Forward:
        seg_logits, domain_logits = model(x, alpha=alpha)

    Para target/US sin mascara:
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
            raise ValueError(f"feature_mode no soportado: {feature_mode}")

    def _domain_logits(
        self,
        features: dict[str, torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        """Aplica GRL a features profundas y predice dominio."""
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
        Devuelve logits de segmentacion y logits de dominio.

        Si return_segmentation=False se evita ejecutar el decoder, util para US.
        El encoder siempre recibe gradientes adversariales a traves del GRL.
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
        Carga pesos de Fase 1 dentro de self.segmenter.

        Acepta checkpoints guardados como state_dict puro o como dict con claves
        comunes: model_state_dict, state_dict, segmenter_state_dict.
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
