import torch
from torch import nn


def langevin_gradient_step(
        energy_function: nn.Module,
        batch_of_points: torch.Tensor,
        step_size: float,
        rng: torch.Generator):

    # Energy descent operation.
    batch_of_points.requires_grad = True
    if batch_of_points.grad is not None:
        batch_of_points.grad.zero_()    # Defend against previously accumulated gradients
    energy = energy_function(batch_of_points)
    energy = torch.sum(energy)
    energy.backward()
    batch_of_points.requires_grad = False   # Needed at this point since we're about to mutate this variable

    batch_of_points -= (0.5*step_size) * batch_of_points.grad
    batch_of_points.grad.zero_()    # Defend against accidentally accumulating.

    # Noise operation
    noise = torch.normal(mean=torch.zeros_like(batch_of_points),
                         std=torch.sqrt(torch.Tensor([step_size])),
                         generator=rng)
    batch_of_points += noise
