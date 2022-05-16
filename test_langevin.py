import torch
from torch import nn


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
