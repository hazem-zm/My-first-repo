"""Milk price forecasting pipeline with proper time-series handling.

This module addresses several issues in the original Colab notebook:

1. Train/validation leakage caused by overlapping date filters is removed.
2. Feature scaling is added to stabilise training.
3. True temporal sequences are constructed so that the LSTM receives
   `(timesteps, n_features)` windows instead of treating features as timesteps.
4. DataFrame slicing is performed with `.copy()` to avoid chained assignment.
5. Imports are consolidated and unused ones removed for clarity.

The code can be executed as a standalone script (expects the Excel file to be
available locally) or the helper functions can be imported into a notebook.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.simplefilter("ignore", category=FutureWarning)

plt.style.use("fivethirtyeight")
sns.set_palette("tab10")


@dataclass
class SequenceData:
    """Container for time-series sequences and their metadata."""

    X: np.ndarray
    y: np.ndarray
    index: Sequence[pd.Timestamp]


def load_milk_price_data(
    excel_path: Path,
    crisis_start: pd.Timestamp | str = "2023-01-01",
    crisis_end: pd.Timestamp | str = "2024-07-31",
) -> pd.DataFrame:
    """Load the milk price dataset and add the crisis indicator column."""

    df = pd.read_excel(excel_path)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the Excel sheet.")

    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    crisis_start = pd.to_datetime(crisis_start)
    crisis_end = pd.to_datetime(crisis_end)

    df["is_crisis"] = (
        (df.index >= crisis_start) & (df.index <= crisis_end)
    ).astype(int)

    return df


def filter_governorate(df: pd.DataFrame, gov_name: str) -> pd.DataFrame:
    """Return a copy of rows for the requested governorate."""

    if "GovName" not in df.columns:
        raise ValueError("Expected a 'GovName' column in the data frame.")

    subset = df[df["GovName"] == gov_name].copy()
    if subset.empty:
        raise ValueError(f"No records found for governorate '{gov_name}'.")

    return subset.sort_index()


def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: pd.Timestamp | str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train/test without overlapping indices."""

    split_date = pd.to_datetime(split_date)
    train_df = df[df.index < split_date].copy()
    test_df = df[df.index >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/test split resulted in an empty partition; "
            "adjust 'split_date'."
        )

    return train_df, test_df


def scale_features_and_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
]:
    """Fit scalers on the training split and transform both splits."""

    feature_cols = list(feature_cols)

    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values

    train_target = train_df[[target_col]].values
    test_target = test_df[[target_col]].values

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)

    train_target_scaled = target_scaler.fit_transform(train_target).ravel()
    test_target_scaled = target_scaler.transform(test_target).ravel()

    return (
        train_features_scaled,
        test_features_scaled,
        train_target_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler,
    )


def build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    index: Sequence[pd.Timestamp],
    window: int,
) -> SequenceData:
    """Create rolling windows for LSTM input."""

    if len(features) != len(targets) or len(features) != len(index):
        raise ValueError("Features, targets, and index must be aligned.")

    if window < 1:
        raise ValueError("Window length must be at least 1.")

    if len(features) <= window:
        raise ValueError(
            "Not enough rows to create sequences; reduce the window length."
        )

    X, y, idx = [], [], []

    for i in range(window, len(features)):
        X.append(features[i - window : i])
        y.append(targets[i])
        idx.append(index[i])

    return SequenceData(X=np.array(X), y=np.array(y), index=idx)


def build_lstm_model(
    window: int,
    n_features: int,
    lstm_units: int = 64,
    learning_rate: float = 1e-4,
) -> Sequential:
    """Construct a simple LSTM regressor."""

    model = Sequential()
    model.add(LSTM(lstm_units, activation="relu", input_shape=(window, n_features)))
    model.add(Dense(1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return RMSE, MAE, MSE, and R² scores."""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2}


def inverse_transform_targets(
    scaler: StandardScaler,
    values: np.ndarray,
) -> np.ndarray:
    """Return predictions in the original target scale."""

    reshaped = values.reshape(-1, 1)
    return scaler.inverse_transform(reshaped).ravel()


def plot_results(
    full_series: pd.Series,
    test_index: Sequence[pd.Timestamp],
    predictions: np.ndarray,
    split_date: pd.Timestamp,
    crisis_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> None:
    """Plot historical data, test window, and predictions."""

    crisis_start, crisis_end = crisis_range

    fig, ax = plt.subplots(figsize=(15, 5))
    full_series.plot(ax=ax, label="Observed")

    ax.plot(
        test_index,
        predictions,
        label="Predictions",
        color="tab:orange",
        linewidth=2,
    )

    ax.axvspan(crisis_start, crisis_end, color="red", alpha=0.3, label="Crisis")
    ax.axvline(split_date, color="black", linestyle="--", label="Train/Test Split")
    ax.set_title("Milk Price Forecasting with LSTM")
    ax.legend()
    plt.tight_layout()
    plt.show()


def predict_price(
    model: Sequential,
    recent_data: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    window: int,
) -> float:
    """Predict the next price using the most recent `window` observations.

    Parameters
    ----------
    model
        Trained Keras model.
    recent_data
        DataFrame containing the most recent observations sorted chronologically.
    feature_cols
        Columns used as model inputs.
    feature_scaler
        Fitted scaler for the input features.
    target_scaler
        Fitted scaler for the target variable.
    window
        Number of timesteps expected by the network.
    """

    if len(recent_data) != window:
        raise ValueError(
            f"`recent_data` must contain exactly {window} rows to match the window size."
        )

    raw_features = recent_data[feature_cols].values
    scaled_features = feature_scaler.transform(raw_features)
    scaled_features = scaled_features.reshape(1, window, len(feature_cols))

    scaled_prediction = model.predict(scaled_features)
    prediction = inverse_transform_targets(target_scaler, scaled_prediction)

    return float(prediction[0])


def run_pipeline(
    excel_path: Path,
    gov_name: str = "القاهرة",
    split_date: str = "2024-01-01",
    window: int = 12,
    epochs: int = 300,
    batch_size: int = 64,
) -> None:
    """Execute the full training and evaluation loop."""

    df = load_milk_price_data(excel_path)
    gov_df = filter_governorate(df, gov_name)

    feature_cols: List[str] = [
        "World Milk Price(US)",
        "Exchange rate",
        "Petroleum (US)",
        "CPI",
        "is_crisis",
    ]
    target_col = "Avg"

    gov_df = gov_df.dropna(subset=feature_cols + [target_col])

    train_df, test_df = train_test_split_by_date(gov_df, split_date)

    (
        train_features_scaled,
        test_features_scaled,
        train_target_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler,
    ) = scale_features_and_target(train_df, test_df, feature_cols, target_col)

    train_sequences = build_sequences(
        train_features_scaled, train_target_scaled, train_df.index, window
    )
    test_sequences = build_sequences(
        test_features_scaled, test_target_scaled, test_df.index, window
    )

    X_train, X_val, y_train, y_val = train_test_split(
        train_sequences.X,
        train_sequences.y,
        test_size=0.2,
        shuffle=False,
    )

    model = build_lstm_model(window, len(feature_cols))
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate on training sequences (full) and test sequences
    train_pred_scaled = model.predict(train_sequences.X)
    test_pred_scaled = model.predict(test_sequences.X)

    train_pred = inverse_transform_targets(target_scaler, train_pred_scaled)
    test_pred = inverse_transform_targets(target_scaler, test_pred_scaled)

    y_train_actual = inverse_transform_targets(target_scaler, train_sequences.y)
    y_test_actual = inverse_transform_targets(target_scaler, test_sequences.y)

    train_metrics = evaluate_predictions(y_train_actual, train_pred)
    test_metrics = evaluate_predictions(y_test_actual, test_pred)

    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    prediction_series = pd.Series(
        test_pred,
        index=pd.DatetimeIndex(test_sequences.index),
        name="prediction",
    )
    actual_series = pd.Series(
        y_test_actual,
        index=pd.DatetimeIndex(test_sequences.index),
        name="Avg",
    )

    results_df = pd.concat([actual_series, prediction_series], axis=1)
    print("\nSample predictions:")
    print(results_df.head())

    crisis_start = pd.to_datetime("2023-01-01")
    crisis_end = pd.to_datetime("2024-07-31")
    plot_results(
        full_series=gov_df[target_col],
        test_index=prediction_series.index,
        predictions=test_pred,
        split_date=pd.to_datetime(split_date),
        crisis_range=(crisis_start, crisis_end),
    )


if __name__ == "__main__":
    data_path = Path("./Milk_CPI.xlsx")

    if not data_path.exists():
        print(
            "Data file 'Milk_CPI.xlsx' not found in the current directory. "
            "Update 'data_path' or place the file alongside this script."
        )
    else:
        run_pipeline(data_path)
