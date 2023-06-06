from torch.utils.data import Dataset

class DummyDataset(Dataset):

    def __init__(self, mask, length):
        self.mask = mask
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.mask