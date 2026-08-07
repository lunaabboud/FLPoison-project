import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class RT_IOT2022(Dataset):
    """
    RT-IoT 2022 Dataset
    Expects:
        data/
            RT_IOT2022_train.csv
            RT_IOT2022_test.csv
    """

    def __init__(self,
                 root,
                 train=True,
                 download=False,
                 transform=None):

        self.root = root
        self.transform = transform
        self.train = train

        filename = "RT_IOT2022_train.csv" if train else "RT_IOT2022_test.csv"
        csv_path = os.path.join(root, filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        # Labels
        self.targets = torch.tensor(
            df["Attack_type"].values,
            dtype=torch.long
        )

        # Features
        self.data = torch.tensor(
            df.drop(columns=["Attack_type"]).values,
            dtype=torch.float32
        )

        self.classes = list(range(12))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y
