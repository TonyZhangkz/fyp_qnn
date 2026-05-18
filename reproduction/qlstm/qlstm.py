# solve path for imports
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from reproduction.qlstm.data_utils import create_datasets
from reproduction.qlstm.model import ShallowRegressionLSTM, QLSTM, QShallowRegressionLSTM
from reproduction.qlstm.run import train_model, test_model, predict

PARQUET_DIR = 'E:\\fyp_qnn\\data\\yfinance'



def main():
    ticker = "AAPL"
    learning_rate=0.01
    num_hidden_units=7
    epochs = 50
    df_yf = pd.read_parquet(f'{PARQUET_DIR}\\{ticker}.parquet', engine='pyarrow')
    features = df_yf.columns
    dataset_train, dataset_test = create_datasets(df_yf, test_size=0.2, sequence_length=3)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    model_classic = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units,num_layers=1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model_classic.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_model(train_loader, model_classic, loss_function, optimizer = optimizer)
        test_loss = test_model(test_loader, model_classic, loss_function)
        print(f"Train Loss: {train_loss}, Test Loss: {test_loss}")
    

if __name__ == "__main__":
    main()
