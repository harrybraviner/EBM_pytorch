import torch
import train_EBM


def test_network_passes_data():
    # Property based testing. We assert that we get the correct output shape for a wide variety of settings.
    for data_size_channels in range(1, 3):
        for kernel_size in range(1, 3):
            for conv_channels in [1, 5, 10]:
                for data_size_x in [10, 15, 25]:
                    for data_size_y in [10, 15, 25]:

                        energy_network = train_EBM.get_energy_network(
                            data_size_channels=data_size_channels,
                            data_size_x=data_size_x,
                            data_size_y=data_size_y,
                            conv1_channels=conv_channels,
                            conv1_kernel_size=kernel_size,
                            conv2_channels=conv_channels,
                            conv2_kernel_size=kernel_size
                        )

                        for batch_size in [1, 3]:
                            x = torch.Tensor(torch.rand((batch_size, data_size_channels, data_size_x, data_size_y)))
                            y = energy_network(x)
                            assert y.shape == (batch_size, 1)
