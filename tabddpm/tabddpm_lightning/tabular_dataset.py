import pandas as pd
import torch
from torch.utils.data import Dataset as torchDataset, DataLoader

# df = pd.read_csv(
#     "~/Downloads/HIGGS.csv.gz",
#     compression="gzip",
#     nrows=100000,
#     index_col=False,
#     header=None,
# )
# df = df.drop(labels=0, axis=1)
# print(df.head())
# df.to_csv("higgs.csv", header=False, index=False)

# df = pd.read_csv("./higgs.csv", header=None, index_col=False)
# print(df.head())
# print(df.shape)
# print(df.values.shape)


class Dataset(torchDataset):
    def __init__(
        self,
        filepath: str,
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.tab = torch.from_numpy(pd.read_csv(self.filepath).values)
        (self.length, self.tabular_width) = self.tab.shape
        self.tab = self.tab[:,None,:]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        return self.tab[index]


# dataset = Dataset("./higgs.csv")
# print(dataset[0])
# print(dataset[[1, 2]])
# print(len(dataset))
