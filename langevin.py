from torch import nn, Tensor


def langevin_gradient_step(
        energy_function: nn.Module,
        batch_of_points: Tensor,
        step_size: float) -> Tensor:

    # FIXME - how do you compute gradients in here?
    #  Remember that this whole thing is going to be wrapped in a no_grad block!
