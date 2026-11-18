import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn

from src.dataset import create_dataloaders
from src.features import prepare_data
from src.model import GRUClassifier
from src.train import compute_metrics, evaluate_baselines, evaluate_loader
from src.utils import PROJECT_ROOT, get_device, load_config, set_seed


def plot_evaluation_figures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    history_path: Path,
    runs_dir: Path,
) -> None:
    """Save ROC curve, Confusion Matrix, and training curves to runs/."""
    # 1. ROC curve + Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob)
    axes[0].plot(fpr, tpr, label=f"GRU (AUC = {metrics['roc_auc']:.4f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", label="Random / Always Up (AUC = 0.5000)")
    axes[0].set_title("Test ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Down/Flat (0)", "Up (1)"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False
    )
    axes[1].set_title("Test Confusion Matrix")

    fig.tight_layout()
    fig.savefig(runs_dir / "test_evaluation.png", dpi=150)
    plt.close(fig)

    # 2. Training curves if history exists
    if history_path.exists():
        hist = pd.read_csv(history_path)
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))

        axes2[0].plot(hist["epoch"], hist["train_loss"], label="Train Loss")
        axes2[0].plot(hist["epoch"], hist["val_loss"], label="Val Loss")
        axes2[0].set_title("Loss per Epoch")
        axes2[0].set_xlabel("Epoch")
        axes2[0].set_ylabel("BCE Loss")
        axes2[0].legend()
        axes2[0].grid(alpha=0.3)

        axes2[1].plot(hist["epoch"], hist["train_auc"], label="Train ROC-AUC")
        axes2[1].plot(hist["epoch"], hist["val_auc"], label="Val ROC-AUC")
        axes2[1].axhline(0.5, color="gray", linestyle="--", label="Baseline 0.5")
        axes2[1].set_title("ROC-AUC per Epoch")
        axes2[1].set_xlabel("Epoch")
        axes2[1].set_ylabel("ROC-AUC")
        axes2[1].legend()
        axes2[1].grid(alpha=0.3)

        fig2.tight_layout()
        fig2.savefig(runs_dir / "training_curves.png", dpi=150)
        plt.close(fig2)


def analyze_errors_by_regime(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    scaled_vol20: np.ndarray,
) -> dict[str, Any]:
    """Break down test performance by calendar year and by volatility regime (high vs low vol20)."""
    df_eval = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "y_true": y_true,
            "y_prob": y_prob,
            "vol20_scaled": scaled_vol20,
        }
    )

    by_year: dict[str, dict[str, Any]] = {}
    for year, grp in df_eval.groupby("year"):
        m = compute_metrics(grp["y_true"].values, grp["y_prob"].values)
        m["count"] = len(grp)
        m["up_ratio"] = float(grp["y_true"].mean())
        by_year[str(year)] = m

    med_vol = float(np.median(scaled_vol20))
    low_mask = df_eval["vol20_scaled"] <= med_vol
    high_mask = df_eval["vol20_scaled"] > med_vol

    by_volatility: dict[str, dict[str, Any]] = {}
    for label, mask in [("low_volatility", low_mask), ("high_volatility", high_mask)]:
        sub = df_eval[mask]
        m = compute_metrics(sub["y_true"].values, sub["y_prob"].values)
        m["count"] = len(sub)
        m["up_ratio"] = float(sub["y_true"].mean())
        by_volatility[label] = m

    return {"by_year": by_year, "by_volatility": by_volatility}


def evaluate_pipeline() -> dict[str, Any]:
    """Evaluate best saved GRU model on Val and Test, compare with baselines, and run error analysis."""
    cfg = load_config()
    t_cfg = cfg["train"]
    m_cfg = cfg["model"]
    f_cfg = cfg["features"]

    seed = t_cfg["seed"]
    set_seed(seed)
    device = get_device()

    runs_dir = PROJECT_ROOT / t_cfg["runs_dir"]
    best_model_path = runs_dir / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {best_model_path}. Run `uv run python -m src.train` first."
        )

    split_data = prepare_data()
    datasets, loaders = create_dataloaders(
        split_data,
        window=f_cfg["window"],
        batch_size=t_cfg["batch_size"],
        seed=seed,
    )

    baselines = evaluate_baselines(datasets, seed=seed)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model = GRUClassifier(
        input_size=m_cfg["input_size"],
        hidden_size=m_cfg["hidden_size"],
        num_layers=m_cfg["num_layers"],
        fc_size=m_cfg["fc_size"],
        dropout=m_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    _, val_metrics, _, _ = evaluate_loader(model, loaders["val"], criterion, device)
    _, test_metrics, y_true_test, y_prob_test = evaluate_loader(
        model, loaders["test"], criterion, device
    )

    # Extract vol20 of target day t for each test window
    test_ds = datasets["test"]
    test_dates = test_ds.get_target_dates()
    test_vol20 = split_data.features_df.loc[test_dates, "vol20"].values

    regime_analysis = analyze_errors_by_regime(
        dates=test_dates,
        y_true=y_true_test,
        y_prob=y_prob_test,
        scaled_vol20=test_vol20,
    )

    plot_evaluation_figures(
        y_true=y_true_test,
        y_prob=y_prob_test,
        history_path=runs_dir / "training_log.csv",
        runs_dir=runs_dir,
    )

    comparison = {
        "always_up": baselines["always_up"],
        "logreg": baselines["logreg"],
        "gru": {
            "best_epoch": int(checkpoint["epoch"]),
            "val": val_metrics,
            "test": test_metrics,
        },
        "error_analysis": regime_analysis,
    }

    with open(runs_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("=== Stage 6: Model Comparison (Val & Test) ===")
    header = f"{'Model':20s} | {'Val Acc':>8s} {'Val F1':>8s} {'Val AUC':>8s} | {'Test Acc':>8s} {'Test F1':>8s} {'Test AUC':>8s}"
    print(header)
    print("-" * len(header))
    for key, label in [
        ("always_up", "Always Up"),
        ("logreg", "LogReg (last day)"),
        ("gru", f"GRU (epoch {checkpoint['epoch']})"),
    ]:
        v = comparison[key]["val"]
        te = comparison[key]["test"]
        print(
            f"{label:20s} | {v['accuracy']:8.4f} {v['f1']:8.4f} {v['roc_auc']:8.4f} | "
            f"{te['accuracy']:8.4f} {te['f1']:8.4f} {te['roc_auc']:8.4f}"
        )

    print("\n=== Error Analysis on Test by Year ===")
    for yr, m in regime_analysis["by_year"].items():
        print(
            f"Year {yr} (n={m['count']:3d}, up={m['up_ratio']:.1%}) | "
            f"Acc: {m['accuracy']:.4f}, F1: {m['f1']:.4f}, ROC-AUC: {m['roc_auc']:.4f}"
        )

    print("\n=== Error Analysis on Test by Volatility Regime ===")
    for reg, m in regime_analysis["by_volatility"].items():
        print(
            f"{reg:16s} (n={m['count']:3d}, up={m['up_ratio']:.1%}) | "
            f"Acc: {m['accuracy']:.4f}, F1: {m['f1']:.4f}, ROC-AUC: {m['roc_auc']:.4f}"
        )

    print(
        f"\nPlots saved to: {runs_dir / 'test_evaluation.png'} and {runs_dir / 'training_curves.png'}"
    )
    return comparison


if __name__ == "__main__":
    evaluate_pipeline()
