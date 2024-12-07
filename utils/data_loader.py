import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_data(batch_size=32):
    data = pd.read_csv("data/processed/train.csv")
    inputs = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    targets = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)
    dataset = TimeSeriesDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
