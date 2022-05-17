import torch
from torch import nn
from langevin import langevin_gradient_step
import numpy as np


def test_req_grad_behaviour():
    """
    Test to assert that pytorch behaves the way I think it does.
    """

    class FooMod(nn.Module):
        def __init__(self):
            super().__init__()

        @staticmethod
        def forward(x):
            return 2 * x**2

    x = torch.Tensor([1.0, 2.0])
    foo_mod = FooMod()

    expected_xs = [
        torch.Tensor([5.0, 10.0]),
        torch.Tensor([25.0, 50.0]),
        torch.Tensor([125.0, 250.0]),
    ]

    for i in range(3):
        x.requires_grad = True
        y = torch.sum(foo_mod(x))
        y.backward()
        # The following line is required, otherwise we will get an error about a leaf Variable that
        # requires grad being use in an in-place operation.
        x.requires_grad = False
        x += x.grad
        # The backward() operation accumulated gradients. Therefore zeroing the gradient is necessary.
        x.grad.zero_()

        torch.testing.assert_allclose(x, expected_xs[i])


def test_langevin_simple_dist():
    """
    Consider the distribution
    rho_T(x) = (x + 1) * 2 / 3 for x in [-1, 0]
             = (1 - 0.5*x) * 2 / 3 for x in [0, 2]
             = 0 otherwise
    This should have a mass of 13/24 in the interval [-0.5, +0.5].
    However, we can't run Langevin dynamics on this, since it does not have support everywhere.
    Instead consider the distribution
    rho(x) = 0.5*rho_T(x) + 0.5*rho_G(x)
    where rho_G is a standard Gaussian, and has cumulative mass 0.38292492254802624 in the interval [-0.5, 0.5]
    (use scipy.stats.norm.cdf(0.5) - scipy.stats.norm.cdf(-0.5) to verify this).
    Therefore we would expect approximately 0.4623 of the sample points to lie in the interval [-0.5, 0.5]
    after many Langevin steps.
    """

    n_points = 1000
    n_steps = 10000
    step_size = 1e-2

    # Seed a generator for repeatability
    g = torch.Generator()
    g.manual_seed(100)

    # Initialize sample points
    x = torch.normal(torch.Tensor([0.0 for _ in range(n_points)]), std=2.0, generator=g)

    class EnergyModule(nn.Module):
        def __init__(self):
            self._norm_const = torch.Tensor([1.0 / np.sqrt((2.0*np.pi))])
            super().__init__()

        def forward(self, x):
            """
            PDF described at start of test.
            """
            pdf = 0.5 * self._norm_const * torch.exp(-0.5 * x**2) + \
                0.5 * torch.where(x > 0.0,
                                  torch.where(x > 2.0,
                                              torch.zeros_like(x),
                                              (2.0/3.0)*(1.0 - 0.5*x)),
                                  torch.where(x < -1.0,
                                              torch.zeros_like(x),
                                              (2.0/3.0)*(1.0 + x))
                                  )
            return -torch.log(pdf)

    energy_module = EnergyModule()

    for i in range(n_steps):

        langevin_gradient_step(
            energy_function=energy_module,
            batch_of_points=x,
            step_size=step_size,
            rng=g
        )

    assert torch.any(torch.isnan(x)).item() is False
    frac_in_interval = torch.sum((x > -0.5) & (x < +0.5)) / n_points
    assert frac_in_interval > 0.4
    assert frac_in_interval < 0.5
