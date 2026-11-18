from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import get_data
from src.utils import PROJECT_ROOT, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLS = [
    "log_ret",
    "hl_range",
    "vol_chg",
    "rsi14",
    "price_sma20",
    "vol20",
]
TARGET_COL = "target"


@dataclass
class SplitData:
    """Container for time-split, normalized features and targets.

    Attributes:
        features_df: Full DataFrame with scaled features, target, and split labels.
        train_df: Subset of features_df where split == 'train'.
        val_df: Subset of features_df where split == 'val'.
        test_df: Subset of features_df where split == 'test'.
        scaler: StandardScaler fitted strictly on train features.
        split_indices: Tuple of (train_end_idx, val_end_idx) in features_df.
    """

    features_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    scaler: StandardScaler
    split_indices: tuple[int, int]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using simple rolling mean (causal)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0, RSI is 100
    rsi = rsi.fillna(100.0).where(avg_gain.notna() & avg_loss.notna())
    return rsi


def compute_features_only(
    df: pd.DataFrame,
    rsi_period: int = 14,
    sma_period: int = 20,
    vol_period: int = 20,
) -> pd.DataFrame:
    """Compute the 6 technical features at day t using only data up to day t."""
    out = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out["log_ret"] = np.log(close / close.shift(1))
    out["hl_range"] = (high - low) / close
    out["vol_chg"] = np.log(volume / volume.shift(1))
    out["rsi14"] = compute_rsi(close, period=rsi_period)
    sma20 = close.rolling(window=sma_period, min_periods=sma_period).mean()
    out["price_sma20"] = close / sma20
    out["vol20"] = (
        out["log_ret"].rolling(window=vol_period, min_periods=vol_period).std()
    )

    return out[FEATURE_COLS]


def build_feature_table(
    df: pd.DataFrame,
    rsi_period: int = 14,
    sma_period: int = 20,
    vol_period: int = 20,
) -> pd.DataFrame:
    """Compute 6 features and target y_t = 1(close_{t+1} > close_t), drop NaNs and last row."""
    feat = compute_features_only(
        df, rsi_period=rsi_period, sma_period=sma_period, vol_period=vol_period
    )
    next_close = df["Close"].shift(-1)
    feat[TARGET_COL] = (next_close > df["Close"]).astype(np.float32)

    # Last row has no next_close (NaN), remove it explicitly along with warm-up NaNs
    feat.loc[df.index[-1], TARGET_COL] = np.nan
    feat = feat.dropna().copy()
    feat[TARGET_COL] = feat[TARGET_COL].astype(int)
    return feat


def check_no_leakage(df: pd.DataFrame, t_idx: int = 100, offset: int = 5) -> None:
    """Leakage test: modifying price at day t + offset must NOT change features at day t."""
    f_orig = compute_features_only(df)

    df_mod = df.copy()
    target_date = df_mod.index[t_idx + offset]
    df_mod.loc[target_date, ["Open", "High", "Low", "Close"]] *= 2.5
    df_mod.loc[target_date, "Volume"] *= 3

    f_mod = compute_features_only(df_mod)

    day_t = df.index[t_idx]
    orig_up_to_t = f_orig.loc[:day_t].dropna()
    mod_up_to_t = f_mod.loc[:day_t].dropna()

    if not np.allclose(orig_up_to_t.values, mod_up_to_t.values, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"Future leakage detected at or before index {t_idx} ({day_t})!"
        )


def split_and_scale(
    feat_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitData:
    """Split strictly by time (70/15/15) and fit StandardScaler ONLY on train."""
    n = len(feat_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    scaled_df = feat_df.copy()
    split_labels = np.array(["test"] * n, dtype=object)
    split_labels[:train_end] = "train"
    split_labels[train_end:val_end] = "val"
    scaled_df["split"] = split_labels

    scaler = StandardScaler()
    train_features = scaled_df.iloc[:train_end][FEATURE_COLS].values
    scaler.fit(train_features)

    all_scaled = scaler.transform(scaled_df[FEATURE_COLS].values)
    scaled_df[FEATURE_COLS] = all_scaled

    train_df = scaled_df.iloc[:train_end].copy()
    val_df = scaled_df.iloc[train_end:val_end].copy()
    test_df = scaled_df.iloc[val_end:].copy()

    return SplitData(
        features_df=scaled_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaler=scaler,
        split_indices=(train_end, val_end),
    )


def prepare_data(save_processed: bool = True) -> SplitData:
    """Full Stage 2 pipeline: load raw data, test for leakage, build features, split & scale."""
    cfg = load_config()
    raw_df = get_data()

    f_cfg = cfg["features"]
    d_cfg = cfg["data"]

    check_no_leakage(raw_df, t_idx=150, offset=5)

    feat_df = build_feature_table(
        raw_df,
        rsi_period=f_cfg["rsi_period"],
        sma_period=f_cfg["sma_period"],
        vol_period=f_cfg["vol_period"],
    )

    split_data = split_and_scale(
        feat_df,
        train_ratio=d_cfg["train_ratio"],
        val_ratio=d_cfg["val_ratio"],
    )

    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        split_data.features_df.to_csv(PROCESSED_DIR / "features_scaled.csv")

    return split_data


if __name__ == "__main__":
    data = prepare_data()
    print("Leakage test (price change at t+5 -> features at t unchanged): PASSED")
    print(f"Total clean rows: {len(data.features_df)}")
    for name, part in [
        ("Train", data.train_df),
        ("Val  ", data.val_df),
        ("Test ", data.test_df),
    ]:
        pos_rate = part[TARGET_COL].mean()
        start_d = part.index[0].date()
        end_d = part.index[-1].date()
        print(
            f"{name}: {len(part):4d} rows ({start_d} -> {end_d}) | "
            f"Up ratio (y=1): {pos_rate:.2%}"
        )
