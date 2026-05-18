import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

def create_datasets(df: pd.DataFrame, 
                    window_size: int = 5,
                    target = 'Close',
                    sequence_length: int = 5,
                    test_size: float = 0.2,) -> tuple[SequenceDataset, SequenceDataset]:
    df = df.ffill().dropna()
    rolling_min = df.rolling(window=window_size).min() + 1e-8
    rolling_max = df.rolling(window=window_size).max()
    df = (df - rolling_min) / (rolling_max - rolling_min)
    df = df.ffill().dropna()
    features = df.columns
    df_train, df_test = train_test_split(df ,test_size=test_size, random_state=42, shuffle=False)
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    return train_dataset, test_dataset