"""
grl.py — Gradient Reversal Layer (GRL)
═══════════════════════════════════════════════════════════════════════════════
Implementación del GRL según Ganin & Lempitsky (ICML 2015).
"Unsupervised Domain Adaptation by Backpropagation."

MATEMÁTICA
──────────
Forward pass:  GRL(x) = x                        (identidad)
Backward pass: ∂L/∂x_grl = −λ · ∂L/∂x           (gradiente negado y escalado)

Donde λ (lambda) es el factor de reversión que puede crecer progresivamente
durante el entrenamiento usando el schedule estándar de Ganin et al.:

    λ(p) = 2 / (1 + exp(−γ · p)) − 1

    p ∈ [0, 1]: progreso relativo del entrenamiento
    γ = 10 por defecto

INTUICIÓN DEL FLUJO BACKWARD
──────────────────────────────
1. El discriminator minimiza: L_d = CE(d(GRL(f(x))), domain_label)
2. Durante backward, los gradientes llegan al GRL multiplicados por −λ.
3. Esto hace que el encoder *maximice* L_d en lugar de minimizarla.
4. El encoder aprende features que "confunden" al discriminator → invarianza.

IMPLEMENTACIÓN
──────────────
Se usa torch.autograd.Function para definir el comportamiento custom
en backward, que es la única forma correcta de implementar esto en PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    """
    Función autograd con forward=identidad, backward=negación escalada.

    Nota: Esta clase no se instancia directamente; se accede a través de
    GradientReversalLayer o de la función auxiliar `grad_reverse`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass: operación identidad.
        Almacena λ en el contexto para el backward.

        Args:
            ctx:     Contexto autograd (almacena estado para backward).
            x:       Tensor de entrada (features del encoder).
            lambda_: Factor de escala de inversión de gradiente.

        Returns:
            x sin modificar (view para evitar copy).
        """
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: invierte y escala el gradiente.

        ∂L/∂x = −λ · grad_output

        Returns:
            Tupla (grad_x, None) — None para el escalar lambda_ (no tiene grad).
        """
        lambda_ = ctx.saved_tensors[0].item()
        return -lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """
    Aplica reversión de gradiente como función standalone.

    Uso conveniente cuando no se quiere instanciar un módulo:
        features_reversed = grad_reverse(encoder_features, lambda_=0.5)

    Args:
        x:       Tensor de features [B, C, H, W] o [B, C].
        lambda_: Factor de escala (≥0). 0 = sin reversión, 1 = inversión total.

    Returns:
        Tensor con mismo shape que x (forward identidad).
    """
    return _GradientReversalFunction.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """
    GRL como módulo nn.Module, compatible con Sequential y cualquier pipeline.

    Permite λ dinámico: se puede actualizar en cada paso de entrenamiento
    usando el schedule de Ganin et al. o cualquier otro esquema.

    Args:
        lambda_init: Valor inicial de λ (default: 0.0 — empieza suave).

    Ejemplo:
        grl = GradientReversalLayer(lambda_init=0.0)
        # ... durante el entrenamiento:
        grl.set_lambda(compute_lambda(epoch, max_epochs))
        reversed_feat = grl(encoder_features)
    """

    def __init__(self, lambda_init: float = 0.0) -> None:
        super().__init__()
        self.lambda_ = lambda_init

    def set_lambda(self, value: float) -> None:
        """
        Actualiza λ dinámicamente (llamar antes de cada forward de entrenamiento).

        Args:
            value: Nuevo valor de λ (float ≥ 0).
        """
        self.lambda_ = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_:.4f}"


def compute_lambda_schedule(
    current_step: int,
    total_steps: int,
    gamma: float = 10.0,
    lambda_max: float = 1.0,
) -> float:
    """
    Schedule progresivo de λ según Ganin et al. (2015).

    Fórmula:
        p = current_step / total_steps          ∈ [0, 1]
        λ(p) = λ_max · (2 / (1 + exp(−γ · p)) − 1)

    Propiedades:
        λ(0)   ≈ 0.0        → inicio suave, segmentación primero
        λ(0.5) ≈ 0.46       → equilibrio progresivo
        λ(1.0) ≈ λ_max      → adversarial total al final

    Args:
        current_step: Paso global actual (épocas × batches_por_época).
        total_steps:  Total de pasos del entrenamiento.
        gamma:        Velocidad de crecimiento (10.0 recomendado).
        lambda_max:   Valor máximo de λ (default 1.0).

    Returns:
        float: λ en [0, lambda_max].
    """
    import math
    if total_steps <= 0:
        return lambda_max
    p = float(current_step) / float(total_steps)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)