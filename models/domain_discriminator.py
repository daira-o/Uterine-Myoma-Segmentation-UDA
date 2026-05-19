"""
domain_discriminator.py — Domain Discriminator para DANN
═══════════════════════════════════════════════════════════════════════════════
Discriminador de dominio que opera sobre features del encoder/bottleneck
de la Attention U-Net. Recibe features ya procesadas por el GRL.

DÓNDE CONECTAR EL DISCRIMINATOR
─────────────────────────────────
Análisis de tu Attention U-Net (base_filters=64):

    Nivel       Módulo          Canales    Resolución (entrada 256×256)
    ─────────── ─────────────── ────────── ──────────────────────────────
    enc1        ConvBlock       64         256×256   ← muy detallado, poco semántico
    enc2        ConvBlock       128        128×128
    enc3        ConvBlock       256        64×64
    enc4        ConvBlock       512        32×32
    bottleneck  ConvBlock       1024       16×16     ← MÁS ABSTRACTO, MEJOR

RECOMENDACIÓN: BOTTLENECK (1024 canales, 16×16)
    ✅ Features más semánticas y abstractas
    ✅ Menor resolución → GAP reduce a [B, 1024] sin pérdida de info
    ✅ El encoder ENTERO aprende a ser domain-invariant
    ✅ No interfiere con skip connections del decoder
    ✅ Balance óptimo entre capacidad discriminativa y estabilidad
    ⚠ Desventaja: no captura diferencias de textura de bajo nivel

ALTERNATIVA: MULTI-SCALE (enc3 + enc4 + bottleneck)
    ✅ Discriminación en múltiples niveles de abstracción
    ✅ Puede capturar diferencias de textura MRI vs US
    ⚠ Más complejo, mayor riesgo de conflictos con decoder
    ⚠ Gradientes más difíciles de balancear

IMPLEMENTADO AQUÍ: ambos modos, configurable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    """
    Discriminador de dominio para DANN.

    Recibe features del encoder (post-GRL) y predice el dominio:
        0 = MRI (source)
        1 = Ultrasound (target)

    Se aplica Global Average Pooling para aplanar los feature maps espaciales,
    luego pasa por capas FC con Dropout para regularización.

    Args:
        in_channels:  Canales del feature map de entrada.
                      Para bottleneck con base_filters=64 → 1024.
        hidden_dim:   Dimensión de la capa oculta FC (default 512).
        num_domains:  Número de clases de dominio (default 2: MRI / US).
        dropout_rate: Tasa de dropout para regularización.

    Flujo:
        features [B, C, H, W]
            → GAP → [B, C]
            → FC(C, hidden_dim) → BN → ReLU → Dropout
            → FC(hidden_dim, hidden_dim//2) → BN → ReLU → Dropout
            → FC(hidden_dim//2, num_domains)
            → logits [B, num_domains]

    Nota sobre pérdida:
        Usar CrossEntropyLoss(logits, labels) donde labels es LongTensor.
        NO aplicar softmax antes — la loss lo hace internamente.
    """

    def __init__(
        self,
        in_channels:  int   = 1024,
        hidden_dim:   int   = 512,
        num_domains:  int   = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)   # [B, C, H, W] → [B, C, 1, 1]

        self.classifier = nn.Sequential(
            # Capa 1
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # Capa 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # Salida
            nn.Linear(hidden_dim // 2, num_domains),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización Xavier para estabilidad adversarial."""
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
            x: Feature map [B, C, H, W] o tensor 2D [B, C] (ya aplanado).

        Returns:
            logits [B, num_domains] — pasar por CrossEntropyLoss.
        """
        if x.dim() == 4:
            x = self.gap(x)         # [B, C, 1, 1]
            x = x.view(x.size(0), -1)  # [B, C]
        elif x.dim() == 2:
            pass  # ya aplanado
        else:
            raise ValueError(f"DomainDiscriminator espera 2D o 4D, recibió {x.dim()}D")

        return self.classifier(x)   # [B, num_domains]


class MultiScaleDomainDiscriminator(nn.Module):
    """
    Discriminador multi-escala: fusiona features de enc3, enc4 y bottleneck.

    Útil cuando las diferencias MRI→US son visibles en múltiples niveles
    (textura granulosa en US vs. bordes suaves en MRI).

    Para tu Attention U-Net con base_filters=64:
        enc3:       256 canales, 64×64
        enc4:       512 canales, 32×32
        bottleneck: 1024 canales, 16×16

    Cada rama tiene su propio GAP + proyección lineal.
    Los vectores proyectados se concatenan y pasan por el clasificador final.

    Args:
        channels_list: Lista de canales de cada nivel. Default: [256, 512, 1024].
        proj_dim:      Dimensión de proyección por rama (default 256).
        num_domains:   Número de dominios (default 2).
        dropout_rate:  Dropout en el clasificador final.
    """

    def __init__(
        self,
        channels_list: list[int] = None,
        proj_dim:      int        = 256,
        num_domains:   int        = 2,
        dropout_rate:  float      = 0.5,
    ) -> None:
        super().__init__()

        if channels_list is None:
            channels_list = [256, 512, 1024]  # enc3, enc4, bottleneck para base=64

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Proyecciones por rama
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
            features_list: Lista de tensores [B, C_i, H_i, W_i].
                           Deben estar en el mismo orden que channels_list.

        Returns:
            logits [B, num_domains].
        """
        projected = []
        for feat, proj in zip(features_list, self.projections):
            pooled = self.gap(feat).view(feat.size(0), -1)  # [B, C_i]
            projected.append(proj(pooled))                   # [B, proj_dim]

        fused = torch.cat(projected, dim=1)   # [B, proj_dim * n_scales]
        return self.classifier(fused)