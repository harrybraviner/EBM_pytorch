from collections import OrderedDict
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from os import path


def main(dataset_name: str,
         batch_size: int,
         conv1_channels: int,
         conv1_kernel_size: int,
         conv2_channels: int,
         conv2_kernel_size: int):

         # Get the dataset, downloading and caching it locally if necessary
    dataset_fn = {
        'mnist': torchvision.datasets.MNIST,
    }[dataset_name]

    data_path = path.join(path.expanduser('~'), 'data')
    dataset_train = dataset_fn(path.join(data_path, f'{dataset_name}_root'), train=True,
                               download=True, transform=torchvision.transforms.ToTensor())
    dataset_valid = dataset_fn(path.join(data_path, f'{dataset_name}_root'), train=False,
                               download=True, transform=torchvision.transforms.ToTensor())

    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=8)

    data_size_channels, data_size_x, data_size_y = dataset_train[0][0].shape

    # Setup network
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
            in_features=,
            out_features=1,
        ))
    ]))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--conv1-channels', type=int, default=20)
    parser.add_argument('--conv1-kernel-size', type=int, default=3)
    args = parser.parse_args()
    args = vars(args)