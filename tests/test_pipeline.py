import numpy as np
import torch

from src.data import get_data
from src.dataset import create_dataloaders
from src.features import FEATURE_COLS, TARGET_COL, check_no_leakage, prepare_data
from src.model import GRUClassifier
from src.utils import load_config


def test_no_future_leakage() -> None:
    df = get_data()
    check_no_leakage(df, t_idx=100, offset=5)
    check_no_leakage(df, t_idx=500, offset=1)


def test_split_and_scaler_isolation() -> None:
    split_data = prepare_data(save_processed=False)
    train_means = split_data.train_df[FEATURE_COLS].mean().values
    train_stds = split_data.train_df[FEATURE_COLS].std(ddof=0).values

    assert np.allclose(train_means, 0.0, atol=1e-6)
    assert np.allclose(train_stds, 1.0, atol=1e-6)


def test_window_dataset_alignment() -> None:
    cfg = load_config()
    window = cfg["features"]["window"]
    split_data = prepare_data(save_processed=False)
    datasets, _loaders = create_dataloaders(split_data, window=window, batch_size=32)

    # First window in train corresponds to rows [0 : window], target at row `window - 1`
    x0, y0 = datasets["train"][0]
    assert x0.shape == (window, len(FEATURE_COLS))
    expected_x0 = split_data.features_df[FEATURE_COLS].iloc[:window].values
    expected_y0 = split_data.features_df[TARGET_COL].iloc[window - 1]
    assert np.allclose(x0.numpy(), expected_x0, atol=1e-6)
    assert np.isclose(y0.item(), expected_y0)

    # First window in val has target at the very first row of val_df,
    # using the last (window - 1) rows of train_df as history
    val_x0, val_y0 = datasets["val"][0]
    train_end, _ = split_data.split_indices
    expected_val_x0 = (
        split_data.features_df[FEATURE_COLS]
        .iloc[train_end - window + 1 : train_end + 1]
        .values
    )
    expected_val_y0 = split_data.features_df[TARGET_COL].iloc[train_end]
    assert np.allclose(val_x0.numpy(), expected_val_x0, atol=1e-6)
    assert np.isclose(val_y0.item(), expected_val_y0)


def test_gru_classifier_forward() -> None:
    cfg = load_config()["model"]
    model = GRUClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        fc_size=cfg["fc_size"],
        dropout=cfg["dropout"],
    )
    x = torch.randn(8, 30, cfg["input_size"])
    logits = model(x)
    assert logits.shape == (8,)
