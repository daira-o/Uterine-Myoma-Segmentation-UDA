"""
Domain discriminators for DANN experiments.

These classifiers operate on encoder or bottleneck features from the Attention
U-Net after gradient reversal.

Recommended attachment point:
    For `base_filters=64`, the bottleneck has 1024 channels at 16 x 16 for a
    256 x 256 input. This is the most abstract representation and is usually
    the most stable domain-discriminator input.

Alternative attachment point:
    Multi-scale features from enc3, enc4, and the bottleneck may capture lower
    level MRI-vs-US texture differences, but they add complexity and can make
    adversarial gradients harder to balance.

Both modes are implemented and selected by `feature_mode`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    """
    Domain classifier for DANN.

    The discriminator receives post-GRL encoder features and predicts:
        0 = MRI source domain
        1 = ultrasound target domain

    Spatial feature maps are reduced with global average pooling, then passed
    through fully connected layers with dropout regularization.

    Use `CrossEntropyLoss(logits, labels)` with integer labels. Do not apply
    softmax before the loss, because CrossEntropyLoss includes it internally.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim: int = 512,
        num_domains: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, num_domains),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Use Xavier initialization for adversarial-training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W] or already-pooled tensor [B, C].

        Returns:
            Domain logits [B, num_domains].
        """
        if x.dim() == 4:
            x = self.gap(x)
            x = x.view(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError(f"DomainDiscriminator expects 2D or 4D input, got {x.dim()}D")

        return self.classifier(x)


class MultiScaleDomainDiscriminator(nn.Module):
    """
    Multi-scale discriminator over enc3, enc4, and bottleneck features.

    This variant is useful when MRI-to-US differences appear at multiple levels,
    such as granular ultrasound texture versus smoother MRI boundaries.

    For an Attention U-Net with `base_filters=64`:
        enc3:       256 channels, 64 x 64
        enc4:       512 channels, 32 x 32
        bottleneck: 1024 channels, 16 x 16

    Each branch applies global average pooling and a linear projection. The
    projected vectors are concatenated before the final classifier.
    """

    def __init__(
        self,
        channels_list: list[int] = None,
        proj_dim: int = 256,
        num_domains: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        if channels_list is None:
            channels_list = [256, 512, 1024]

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, proj_dim),
                nn.ReLU(inplace=True),
            )
            for ch in channels_list
        ])

        fused_dim = proj_dim * len(channels_list)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim // 2, num_domains),
        )

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: Tensors [B, C_i, H_i, W_i] in `channels_list` order.

        Returns:
            Domain logits [B, num_domains].
        """
        projected = []
        for feat, proj in zip(features_list, self.projections):
            pooled = self.gap(feat).view(feat.size(0), -1)
            projected.append(proj(pooled))

        fused = torch.cat(projected, dim=1)
        return self.classifier(fused)
