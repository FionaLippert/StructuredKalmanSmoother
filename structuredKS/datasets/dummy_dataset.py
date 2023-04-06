from torch.utils.data import Dataset

class DummyDataset(Dataset):

    def __init__(self, index, length):
        self.index = index
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.index