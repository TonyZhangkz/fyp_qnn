import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


class SimpleCustomQNNRegressor:
    """
    Minimal, dependency-light CustomQNN-style regressor.
    - Angle-encoding style transform (arcsin/arccos over projected features)
    - Lightweight nearest-neighbor "entanglement" interaction term
    - Closed-form linear readout training for stability
    """

    def __init__(self, n_qubits: int = 3, lookback: int = 2, seed: int = 42):
        self.n_qubits = n_qubits
        self.lookback = lookback
        self.seed = seed
        self.proj = None
        self.readout_w = None
        self.readout_b = None

    def _flatten(self, X_seq: np.ndarray) -> np.ndarray:
        # X_seq: (N, lookback, n_features)
        return X_seq.reshape(X_seq.shape[0], -1)

    def _encode(self, X_flat: np.ndarray) -> np.ndarray:
        projected = np.tanh(X_flat @ self.proj)  # range in (-1, 1)
        projected = np.clip(projected, -1.0, 1.0)

        theta = np.arcsin(projected)
        phi = np.arccos(projected)

        base = np.sin(theta) + np.cos(phi)
        entangled = 0.5 * base * np.roll(base, shift=-1, axis=1)
        return base + entangled

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        X_flat = self._flatten(X_seq)
        self.proj = rng.normal(0.0, 0.5, size=(X_flat.shape[1], self.n_qubits))

        q_features = self._encode(X_flat)
        A = np.column_stack([q_features, np.ones((q_features.shape[0], 1))])
        w = np.linalg.lstsq(A, y.reshape(-1), rcond=None)[0]
        self.readout_w = w[:-1]
        self.readout_b = w[-1]
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X_seq)
        q_features = self._encode(X_flat)
        pred = q_features @ self.readout_w + self.readout_b
        return pred.reshape(-1, 1)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = out["close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=out.index).rolling(14).mean() / (atr + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=out.index).rolling(14).mean() / (atr + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    out["adx_14"] = dx.rolling(14).mean()

    return out.dropna()


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback : i])
        y_seq.append(y[i])
    return np.asarray(X_seq), np.asarray(y_seq)


def main():
    data_path = r"E:\fyp_qnn\data\yfinance\AAPL.parquet"
    lookback = 2
    k_features = 3
    n_qubits = 3
    holdout_ratio = 0.2
    n_splits = 3

    df = pd.read_parquet(data_path).dropna()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df = add_indicators(df)

    candidate_features = [c for c in ["open", "high", "low", "close", "Volume", "rsi_14", "macd", "macd_signal", "adx_14"] if c in df.columns]
    X_df = df[candidate_features]
    y = df["close"].values.reshape(-1, 1)

    split_idx = int(len(df) * (1 - holdout_ratio))
    X_trainval_df, X_test_df = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_trainval, y_test = y[:split_idx], y[split_idx:]

    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_trainval_sel = selector.fit_transform(X_trainval_df, y_trainval.ravel())
    X_test_sel = selector.transform(X_test_df)
    selected = X_trainval_df.columns[selector.get_support()].tolist()

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    X_trainval_scaled = x_scaler.fit_transform(X_trainval_sel)
    X_test_scaled = x_scaler.transform(X_test_sel)
    y_trainval_scaled = y_scaler.fit_transform(y_trainval)
    y_test_scaled = y_scaler.transform(y_test)

    X_trainval_seq, y_trainval_seq = make_sequences(X_trainval_scaled, y_trainval_scaled, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test_scaled, lookback)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse = []
    best_model = None
    best_rmse = np.inf

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_trainval_seq), start=1):
        model = SimpleCustomQNNRegressor(n_qubits=n_qubits, lookback=lookback, seed=42 + fold)
        model.fit(X_trainval_seq[tr_idx], y_trainval_seq[tr_idx])
        va_pred = model.predict(X_trainval_seq[va_idx])
        rmse = mean_squared_error(y_trainval_seq[va_idx], va_pred, squared=False)
        fold_rmse.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    y_test_pred_scaled = best_model.predict(X_test_seq)
    y_test_true = y_scaler.inverse_transform(y_test_seq)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)
    test_rmse = mean_squared_error(y_test_true, y_test_pred, squared=False)

    print("Selected features:", selected)
    print("Fold RMSE (scaled):", [round(float(v), 6) for v in fold_rmse])
    print("Mean fold RMSE (scaled):", round(float(np.mean(fold_rmse)), 6))
    print("Held-out test RMSE (price scale):", round(float(test_rmse), 6))
    print("Smoke test complete: pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
