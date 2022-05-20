from collections import OrderedDict, deque

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.utils.parametrizations import spectral_norm
import torchvision
from os import path, makedirs
import numpy as np
from tqdm import tqdm
from langevin import langevin_gradient_step
import matplotlib.pyplot as plt
from datasets import JointGaussianDataset, CircleDistribution


def make_single_plot(data: torch.Tensor) -> plt.Figure:

    fig, ax = plt.subplots()

    ax.plot(
        [data[i, 0] for i in range(data.shape[0])],
        [data[i, 1] for i in range(data.shape[0])],
        '.'
    )

    return fig


def make_plots(data: torch.Tensor) -> plt.Figure:
    ncols = 3
    nrows = (data.shape[0] // ncols) + int(bool(data.shape[0] % ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    # Remove tick labels and scales
    for ax in [x for y in axs for x in y]:
        ax.tick_params(left=False, right=False, bottom=False, top=False)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

    for i in range(data.shape[0]):
        row = i // ncols
        col = i - row*ncols
        axs[row, col].imshow(data[i].squeeze())

    return fig


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
        ('conv1', spectral_norm(nn.Conv2d(
            in_channels=data_size_channels,
            out_channels=conv1_channels,
            kernel_size=conv1_kernel_size
        ))),
        ('relu1', nn.ReLU()),
        ('conv2', spectral_norm(nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=conv2_kernel_size
        ))),
        ('relu2', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('linear', spectral_norm(nn.Linear(
            in_features=final_layer_size,
            out_features=1,
        )))
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
         langevin_num_steps: int,
         langevin_gradient_clipping: float,
         adam_learning_rate: float,
         adam_beta1: float,
         adam_beta2: float,
         output_dir: str):

    # Setup PRNGs
    rng_langevin = torch.Generator()
    rng_langevin.manual_seed(1234)
    rng_buffer = np.random.RandomState(5678)

    data_path = path.join(path.expanduser('~'), 'data')
    makedirs(output_dir, exist_ok=True)

    # Get the dataset, downloading and caching it locally if necessary
    dataset_train = {
        'mnist': torchvision.datasets.MNIST(path.join(data_path, f'{dataset_name}_root'), train=True,
                                            download=True, transform=torchvision.transforms.ToTensor()),
        'gaussian': JointGaussianDataset(length=50000, seed=12345),
        'circle': CircleDistribution(length=50000, seed=12345),
    }[dataset_name]

    # Note that the ToTensor transform scales the pixel intensities to lie in [0, 1]
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=True, drop_last=True, num_workers=8)

    data_shape = dataset_train[0][0].shape
    if len(data_shape) == 3:
        data_size_channels, data_size_x, data_size_y = data_shape

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
    elif len(data_shape) == 1:
        (data_dim,) = data_shape

        hidden_size_1 = 128
        hidden_size_2 = 128

        energy_network = nn.Sequential(OrderedDict([
            ('linear_1', spectral_norm(nn.Linear(
                in_features=data_dim,
                out_features=hidden_size_1,
            ))),
            ('relu1', nn.ReLU()),
            ('linear_2', spectral_norm(nn.Linear(
                in_features=hidden_size_1,
                out_features=hidden_size_2,
            ))),
            ('relu2', nn.ReLU()),
            ('linear_3', spectral_norm(nn.Linear(
                in_features=hidden_size_2,
                out_features=1,
            ))),
        ]))
    else:
        raise ValueError(f'Unknown data shape: {data_shape}')

    # FIXME - how do I implement gradient clipping?
    optimizer = optim.Adam(energy_network.parameters(), lr=adam_learning_rate, betas=(adam_beta1, adam_beta2))

    # This buffer will store samples that we evolve by Langevin dynamics.
    # These are needed as the 'negative' samples in the gradient step.
    sample_buffer = deque(maxlen=buffer_size)

    for epoch in range(epochs):
        for positive_images, _ in tqdm(iter(data_loader_train)):
            # Sample from buffer or uniform distribution
            negative_images = torch.Tensor(*((batch_size,) + tuple(data_shape))).uniform_(-1.0, 1.0)
            buffer_sample_idx_batch = []
            use_buffer = rng_buffer.choice([True, False],
                                           p=[buffer_sample_probability, 1.0 - buffer_sample_probability],
                                           size=(batch_size,))
            for i in range(batch_size):
                if use_buffer[i]:
                    buffer_sample_idx = rng_buffer.choice(sample_buffer.maxlen)
                    if buffer_sample_idx < len(sample_buffer):
                        if len(data_shape) == 3:
                            negative_images[i, :, :, :] = sample_buffer[buffer_sample_idx]
                        elif len(data_shape) == 1:
                            negative_images[i, :] = sample_buffer[buffer_sample_idx]
                        buffer_sample_idx_batch.append(buffer_sample_idx)
                else:
                    buffer_sample_idx_batch.append(None)

            # Execute in-place Langevin dynamics on the negative samples
            for _ in range(langevin_num_steps):
                langevin_gradient_step(
                    energy_function=energy_network,
                    batch_of_points=negative_images,
                    step_size=langevin_step_size,
                    gradient_clipping=langevin_gradient_clipping,
                    rng=rng_langevin
                )

            # Write points (after Langevin evolution) back to the buffer
            for i in range(batch_size):
                if len(data_shape) == 3:
                    sample_buffer.append(negative_images[i, :, :, :])
                elif len(data_shape) == 1:
                    sample_buffer.append(negative_images[i, :])

            # Compute loss function and take gradient step
            optimizer.zero_grad()
            e_pos = energy_network(positive_images)
            e_neg = energy_network(negative_images)
            loss = (1/batch_size) * torch.sum(alpha_l2 * (e_pos**2 + e_neg**2) + e_pos - e_neg, dim=0)
            loss.backward()
            optimizer.step()

        # End of epoch, write some examples to disc
        print(f'Completed epoch {epoch}')
        if len(data_shape) == 3:
            samples_to_output = torch.Tensor(*((9,) + tuple(data_shape))).uniform_(-1.0, 1.0)
        else:
            samples_to_output = torch.Tensor(*((100,) + tuple(data_shape))).uniform_(-1.0, 1.0)
        for _ in range(langevin_num_steps):
            langevin_gradient_step(
                energy_function=energy_network,
                batch_of_points=samples_to_output,
                step_size=langevin_step_size,
                gradient_clipping=langevin_gradient_clipping,
                rng=rng_langevin
            )
        if len(data_shape) == 3:
            fig = make_plots(samples_to_output)
        else:
            fig = make_single_plot(samples_to_output)
        fig.savefig(path.join(output_dir, f'epoch_{epoch}_samples.png'))
        plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='circle')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--conv1-channels', type=int, default=20)
    parser.add_argument('--conv1-kernel-size', type=int, default=3)
    parser.add_argument('--conv2-channels', type=int, default=20)
    parser.add_argument('--conv2-kernel-size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--buffer-sample-probability', type=float, default=0.95)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--alpha-l2', type=float, default=1.0)
    parser.add_argument('--langevin-step-size', type=float, default=10.0)
    parser.add_argument('--langevin-num-steps', type=int, default=120)
    parser.add_argument('--langevin-gradient-clipping', type=float, default=0.01)
    parser.add_argument('--adam-learning-rate', type=float, default=1e-4)
    parser.add_argument('--adam-beta1', type=float, default=0.0)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()
    args = vars(args)

    main(**args)
