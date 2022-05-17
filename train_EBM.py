from collections import OrderedDict, deque

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision
from os import path
import numpy as np
from tqdm import tqdm
from langevin import langevin_gradient_step


def get_energy_network(
        data_size_channels: int,
        data_size_x: int,
        data_size_y: int,
        conv1_channels: int,
        conv1_kernel_size: int,
        conv2_channels: int,
        conv2_kernel_size: int) -> nn.Module:

    final_layer_size = (conv2_channels *
                        (data_size_x - (conv1_kernel_size - 1) - (conv2_kernel_size - 1)) *
                        (data_size_y - (conv1_kernel_size - 1) - (conv2_kernel_size - 1)))
    # FIXME - seed RNGs for weights
    energy_network: nn.Module = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(
            in_channels=data_size_channels,
            out_channels=conv1_channels,
            kernel_size=conv1_kernel_size
        )),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=conv2_kernel_size
        )),
        ('relu2', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('linear', nn.Linear(
            in_features=final_layer_size,
            out_features=1,
        ))
    ]))

    return energy_network


def main(dataset_name: str,
         batch_size: int,
         conv1_channels: int,
         conv1_kernel_size: int,
         conv2_channels: int,
         conv2_kernel_size: int,
         epochs: int,
         buffer_sample_probability: float,
         buffer_size: int,
         alpha_l2: float,
         langevin_step_size: float,
         langevin_gradient_clipping: float,
         adam_learning_rate: float,
         adam_beta1: float,
         adam_beta2: float):

    # Setup PRNGs
    rng_langevin = torch.Generator()
    rng_langevin.manual_seed(1234)

    # Get the dataset, downloading and caching it locally if necessary
    dataset_fn = {
        'mnist': torchvision.datasets.MNIST,
    }[dataset_name]

    # FIXME - scaling of data to be between 0 and 1?
    data_path = path.join(path.expanduser('~'), 'data')
    dataset_train = dataset_fn(path.join(data_path, f'{dataset_name}_root'), train=True,
                               download=True, transform=torchvision.transforms.ToTensor())
    dataset_valid = dataset_fn(path.join(data_path, f'{dataset_name}_root'), train=False,
                               download=True, transform=torchvision.transforms.ToTensor())

    data_loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=True, drop_last=True, num_workers=8)
    data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=8)

    data_size_channels, data_size_x, data_size_y = dataset_train[0][0].shape

    # Setup network
    energy_network = get_energy_network(
        data_size_channels=data_size_channels,
        data_size_x=data_size_x,
        data_size_y=data_size_y,
        conv1_channels=conv1_channels,
        conv1_kernel_size=conv1_kernel_size,
        conv2_channels=conv2_channels,
        conv2_kernel_size=conv2_kernel_size
    )

    # FIXME - how do I implement gradient clipping?
    optimizer = optim.Adam(energy_network.parameters(), lr=adam_learning_rate, betas=(adam_beta1, adam_beta1))

    # This buffer will store samples that we evolve by Langevin dynamics.
    # These are needed as the 'negative' samples in the gradient step.
    sample_buffer = deque(maxlen=buffer_size)
    # FIXME - can I replace this with a pytorch RNG that I get to seed?
    rng_buffer = np.random.RandomState(1234)

    for epoch in range(epochs):
        for positive_images, _ in tqdm(iter(data_loader_train)):
            # Sample from buffer or uniform distribution
            negative_images = torch.rand((batch_size, data_size_channels, data_size_x, data_size_y))
            buffer_sample_idx_batch = []
            use_buffer = rng_buffer.choice([True, False],
                                           p=[buffer_sample_probability, 1.0 - buffer_sample_probability],
                                           size=(batch_size,))
            for i in range(batch_size):
                if use_buffer[i]:
                    buffer_sample_idx = rng_buffer.choice(sample_buffer.maxlen)
                    if buffer_sample_idx < len(sample_buffer):
                        negative_images[i, :, :, :] = sample_buffer[buffer_sample_idx]
                        buffer_sample_idx_batch.append(buffer_sample_idx)
                else:
                    buffer_sample_idx_batch.append(None)

            # Execute in-place Langevin dynamics on the negative samples
            langevin_gradient_step(
                energy_function=energy_network,
                batch_of_points=negative_images,
                step_size=langevin_step_size,
                rng=rng_langevin
            )

            # Write points (after Langevin evolution) back to the buffer
            for i in range(batch_size):
                sample_buffer.append(negative_images[i, :, :, :])

            # Compute loss function and take gradient step
            optimizer.zero_grad()
            e_pos = energy_network(positive_images)
            e_neg = energy_network(negative_images)
            loss = (1/batch_size) * torch.sum(alpha_l2 * (e_pos**2 + e_neg**2) + e_pos - e_neg, dim=0)
            loss.backward()
            optimizer.step()

        pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--conv1-channels', type=int, default=20)
    parser.add_argument('--conv1-kernel-size', type=int, default=3)
    parser.add_argument('--conv2-channels', type=int, default=20)
    parser.add_argument('--conv2-kernel-size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--buffer-sample-probability', type=float, default=0.95)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--alpha-l2', type=float, default=1.0)
    parser.add_argument('--langevin-step-size', type=float, default=10.0)
    parser.add_argument('--langevin-gradient-clipping', type=float, default=0.01)
    parser.add_argument('--adam-learning-rate', type=float, default=1e-4)
    parser.add_argument('--adam-beta1', type=float, default=0.0)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    args = parser.parse_args()
    args = vars(args)

    main(**args)
