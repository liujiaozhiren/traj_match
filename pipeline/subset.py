from torch.utils.data import Sampler
import torch

class EpochRandomSubsetSampler(Sampler[int]):
    def __init__(self, dataset_size: int, subset_size: int, seed: int = 0):
        assert subset_size <= dataset_size
        self.dataset_size = dataset_size
        self.subset_size = subset_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randperm(self.dataset_size, generator=g)[:self.subset_size]
        return iter(idx.tolist())

    def __len__(self):
        return self.subset_size
