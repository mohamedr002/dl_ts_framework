from torch.utils.data import Dataset
import torch
import numpy as np

class TSDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(TSDataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        self.len = X_train.shape[0]
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train)
        else:
            self.x_data = X_train.float()
            self.y_data = y_train
    def __getitem__(self, index):

        return self.x_data[index].float(), self.y_data[index]

    def __len__(self):
        return self.len