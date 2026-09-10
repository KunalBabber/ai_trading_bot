import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, features, target_return, target_long, target_short,
                 sequence_length=96):
        self.X, self.y_reg, self.y_long, self.y_short = [], [], [], []

        for i in range(sequence_length - 1, len(features)):
            self.X.append(features[i-sequence_length+1:i+1])
            self.y_reg.append(target_return[i])
            self.y_long.append(target_long[i])
            self.y_short.append(target_short[i])

        self.X = np.asarray(self.X, dtype=np.float32)
        self.y_reg = np.asarray(self.y_reg, dtype=np.float32)
        self.y_long = np.asarray(self.y_long, dtype=np.float32)
        self.y_short = np.asarray(self.y_short, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y_reg[idx]),
            torch.tensor(self.y_long[idx]),
            torch.tensor(self.y_short[idx]),
        )
