import torch
from torch.utils.data import Dataset


class JointGaussianDataset(Dataset):

    def __init__(self, length: int, seed: int):
        g = torch.Generator()
        g.manual_seed(seed)
        self._data = torch.normal(
            mean=torch.full(size=(length, 2), fill_value=0.0),
            std=torch.Tensor([1.0, 2.0]).repeat([length, 1]),
            generator=g
        )

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx, :], None
