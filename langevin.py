import torch
from torch import nn
from typing import Optional


def langevin_gradient_step(
        energy_function: nn.Module,
        elementwise_min: float,
        elementwise_max: float,
        batch_of_points: torch.Tensor,
        step_size: float,
        noise_sigma: Optional[float],
        gradient_clipping: float,
        rng: torch.Generator):

    # Set the energy function to eval mode to avoid propagating gradients back into its parameters.
    is_training_mode = energy_function.training
    energy_function.eval()
    for p in energy_function.parameters():
        p.requires_grad = False

    # Energy descent operation.
    batch_of_points.requires_grad = True
    if batch_of_points.grad is not None:
        batch_of_points.grad.zero_()    # Defend against previously accumulated gradients
    energy = energy_function(batch_of_points)
    energy = torch.sum(energy)
    energy.backward()

    # Restore the state of the energy function
    for p in energy_function.parameters():
        p.requires_grad = True
    energy_function.train(is_training_mode)

    batch_of_points.requires_grad = False   # Needed at this point since we're about to mutate this variable

    clipped_grad = batch_of_points.grad.detach().clamp(-gradient_clipping, +gradient_clipping)
    batch_of_points -= (0.5*step_size) * clipped_grad
    batch_of_points.grad.zero_()    # Defend against accidentally accumulating.

    # Noise operation
    sigma = torch.Tensor([noise_sigma]) if noise_sigma is not None else torch.sqrt(torch.Tensor([step_size]))
    noise = torch.normal(mean=torch.zeros_like(batch_of_points),
                         std=torch.Tensor([sigma]),
                         generator=rng)
    batch_of_points += noise

    # Restrict to lie within the domain
    batch_of_points.clamp_(min=elementwise_min, max=elementwise_max)
