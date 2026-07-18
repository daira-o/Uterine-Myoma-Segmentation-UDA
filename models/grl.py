"""
Gradient Reversal Layer (GRL).

This implementation follows Ganin and Lempitsky, "Unsupervised Domain
Adaptation by Backpropagation" (ICML 2015).

Mathematics:
    Forward pass:  GRL(x) = x
    Backward pass: dL/dx = -lambda * grad_output

The reversal factor can be increased during training with the schedule from the
paper:

    lambda(p) = 2 / (1 + exp(-gamma * p)) - 1

where `p` is relative training progress in [0, 1] and gamma is usually 10.

Backward intuition:
    1. The discriminator minimizes domain classification loss.
    2. During backpropagation, GRL multiplies the feature gradient by -lambda.
    3. The encoder therefore maximizes the discriminator loss.
    4. Encoder features become harder to classify by domain.

PyTorch requires a custom `torch.autograd.Function` for this exact forward and
backward behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    """
    Autograd function with identity forward and scaled gradient negation.

    This class is not instantiated directly; use `GradientReversalLayer` or
    the helper function `grad_reverse`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Return `x` unchanged and store lambda for the backward pass."""
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Reverse and scale the incoming gradient."""
        lambda_ = ctx.saved_tensors[0].item()
        return -lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """
    Apply gradient reversal as a standalone function.

    Args:
        x: Feature tensor [B, C, H, W] or [B, C].
        lambda_: Reversal scale. 0 disables reversal; 1 applies full reversal.

    Returns:
        A tensor with the same shape and forward values as `x`.
    """
    return _GradientReversalFunction.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """
    `nn.Module` wrapper for GRL, compatible with ordinary model pipelines.

    `lambda_` can be updated before each training forward pass with a schedule
    such as `compute_lambda_schedule`.
    """

    def __init__(self, lambda_init: float = 0.0) -> None:
        super().__init__()
        self.lambda_ = lambda_init

    def set_lambda(self, value: float) -> None:
        """Update the reversal factor used by subsequent forward passes."""
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
    Progressive GRL schedule from Ganin et al. (2015).

    Formula:
        p = current_step / total_steps
        lambda(p) = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)

    Properties:
        lambda(0)   ~= 0.0       starts gently, prioritizing segmentation.
        lambda(0.5) ~= 0.46      increases adversarial pressure gradually.
        lambda(1.0) ~= lambda_max reaches full adversarial strength.
    """
    import math

    if total_steps <= 0:
        return lambda_max
    p = float(current_step) / float(total_steps)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
