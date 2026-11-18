from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import FEATURE_COLS, TARGET_COL, SplitData, prepare_data
from src.utils import load_config


class WindowDataset(Dataset):
    """Sliding window dataset for time-series binary classification.

    Rule:
        The window for target at index `t` covers `[t - window + 1 : t + 1]` (inclusive of t),
        so it may include days from the preceding split as input context,
        while the target at `t` strictly belongs to the specified split ('train', 'val', or 'test').
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        split: Literal["train", "val", "test"],
        window: int = 30,
    ) -> None:
        super().__init__()
        self.window = window
        self.split = split

        self.features = features_df[FEATURE_COLS].values.astype(np.float32)
        self.targets = features_df[TARGET_COL].values.astype(np.float32)
        self.dates = features_df.index

        # Valid target indices `t` are those where:
        # 1) features_df["split"] == split
        # 2) t >= window - 1 (so a full `window` of past/current rows exists)
        split_mask = features_df["split"].values == split
        all_indices = np.arange(len(features_df))
        self.target_indices = all_indices[split_mask & (all_indices >= window - 1)]

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.target_indices[idx]
        x = self.features[t - self.window + 1 : t + 1]
        y = self.targets[t]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    def get_target_dates(self) -> pd.DatetimeIndex:
        """Return the dates corresponding to the target day t of each window."""
        return self.dates[self.target_indices]


def create_dataloaders(
    split_data: SplitData,
    window: int = 30,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[
    dict[str, WindowDataset],
    dict[str, DataLoader],
]:
    """Create WindowDatasets and DataLoaders for train, val, and test splits.

    Shuffling is enabled ONLY for train. Val and test strictly preserve chronological order.
    """
    datasets: dict[str, WindowDataset] = {}
    loaders: dict[str, DataLoader] = {}

    generator = torch.Generator()
    generator.manual_seed(seed)

    for split in ("train", "val", "test"):
        ds = WindowDataset(split_data.features_df, split=split, window=window)
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
            drop_last=False,
        )

    return datasets, loaders


if __name__ == "__main__":
    cfg = load_config()
    window = cfg["features"]["window"]
    batch_size = cfg["train"]["batch_size"]
    seed = cfg["train"]["seed"]

    split_data = prepare_data()
    datasets, loaders = create_dataloaders(
        split_data, window=window, batch_size=batch_size, seed=seed
    )

    for name, ds in datasets.items():
        dates = ds.get_target_dates()
        print(
            f"{name.upper():5s} dataset: {len(ds):4d} windows | "
            f"target dates: {dates[0].date()} -> {dates[-1].date()}"
        )

    # Check batch shapes
    xb, yb = next(iter(loaders["train"]))
    print(f"\nSample train batch shapes: X={tuple(xb.shape)}, y={tuple(yb.shape)}")

    # Verify first train window against source features_df
    first_x, first_y = datasets["train"][0]
    expected_x = split_data.features_df[FEATURE_COLS].iloc[:window].values
    expected_y = split_data.features_df[TARGET_COL].iloc[window - 1]
    assert np.allclose(first_x.numpy(), expected_x, atol=1e-6), (
        "First window X mismatch!"
    )
    assert np.isclose(first_y.item(), expected_y), "First window y mismatch!"
    print("First window verification against source table: PASSED")
