"""
attention_unet_dann.py
Extension minima de AttentionUNet para exponer encoder/bottleneck/decoder en DANN.

La arquitectura base sigue viviendo en models/attention_unet.py. Esta clase hereda
sus modulos sin duplicarlos, por lo que los checkpoints de Fase 1 mantienen los
mismos nombres de parametros.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.attention_unet import AttentionUNet


class AttentionUNetDANN(AttentionUNet):
    """Attention U-Net compatible con DANN sin modificar la clase original."""

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Ejecuta encoder + bottleneck y expone features profundas para DANN.

        Los nombres de modulos son los mismos que en AttentionUNet:
        enc1/enc2/enc3/enc4/bottleneck.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        return {"enc1": e1, "enc2": e2, "enc3": e3, "enc4": e4, "bottleneck": b}

    def decode(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Ejecuta el decoder de segmentacion a partir de features del encoder."""
        e1 = features["enc1"]
        e2 = features["enc2"]
        e3 = features["enc3"]
        e4 = features["enc4"]
        b = features["bottleneck"]

        g4 = self.gate4(b)
        a4 = self.att4(e4, g4)
        d4 = self.dec4(torch.cat([self.up4(b), a4], dim=1))

        g3 = self.gate3(d4)
        a3 = self.att3(e3, g3)
        d3 = self.dec3(torch.cat([self.up3(d4), a3], dim=1))

        g2 = self.gate2(d3)
        a2 = self.att2(e2, g2)
        d2 = self.dec2(torch.cat([self.up2(d3), a2], dim=1))

        g1 = self.gate1(d2)
        a1 = self.att1(e1, g1)
        d1 = self.dec1(torch.cat([self.up1(d2), a1], dim=1))

        return self.output_conv(d1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve las features del bottleneck para adaptacion de dominio."""
        return self.encode(x)["bottleneck"]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.encode(x)
        logits = self.decode(features)
        if return_features:
            return logits, features
        return logits

