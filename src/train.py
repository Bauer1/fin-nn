import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import WindowDataset, create_dataloaders
from src.features import prepare_data
from src.model import GRUClassifier
from src.utils import PROJECT_ROOT, get_device, load_config, set_seed


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute binary classification metrics: accuracy, F1, and ROC-AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def evaluate_baselines(
    datasets: dict[str, WindowDataset],
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate Stage 4 baselines on val and test:
    1. 'Always Up': predicts class 1 with probability = train positive rate.
    2. 'Logistic Regression': trained on the last day features of each window.
    """
    train_ds = datasets["train"]
    X_train_last = np.array([train_ds[i][0][-1].numpy() for i in range(len(train_ds))])
    y_train = np.array([train_ds[i][1].item() for i in range(len(train_ds))], dtype=int)

    train_pos_rate = float(y_train.mean())

    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train_last, y_train)

    results: dict[str, dict[str, dict[str, float]]] = {
        "always_up": {},
        "logreg": {},
    }

    for split in ("val", "test"):
        ds = datasets[split]
        X_last = np.array([ds[i][0][-1].numpy() for i in range(len(ds))])
        y_true = np.array([ds[i][1].item() for i in range(len(ds))], dtype=int)

        # 1. Always Up baseline (constant probability >= 0.5 if train_pos_rate >= 0.5, always predicting 1)
        always_up_probs = np.full_like(
            y_true, fill_value=max(train_pos_rate, 0.5001), dtype=float
        )
        results["always_up"][split] = compute_metrics(y_true, always_up_probs)

        # 2. Logistic Regression on last day of window
        logreg_probs = clf.predict_proba(X_last)[:, 1]
        results["logreg"][split] = compute_metrics(y_true, logreg_probs)

    return results


def sanity_check_overfit_one_batch(
    loader: DataLoader,
    m_cfg: dict[str, Any],
    device: torch.device,
    steps: int = 150,
) -> float:
    """Stage 5 Sanity Check: verify that GRUClassifier can overfit a single mini-batch."""
    xb, yb = next(iter(loader))
    xb = xb[:16].to(device)
    yb = yb[:16].to(device)

    model = GRUClassifier(
        input_size=m_cfg["input_size"],
        hidden_size=m_cfg["hidden_size"],
        num_layers=m_cfg["num_layers"],
        fc_size=m_cfg["fc_size"],
        dropout=0.0,  # Disable dropout for fast memorization check
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    final_loss = 1.0
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    return final_loss


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return loss, metrics, y_true, and y_prob."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        batch_size = xb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(yb.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(int)
    avg_loss = total_loss / max(total_count, 1)
    metrics = compute_metrics(y_true, y_prob)
    return avg_loss, metrics, y_true, y_prob


def train_pipeline() -> dict[str, Any]:
    """Run full training pipeline with baselines, sanity check, early stopping, and logging."""
    cfg = load_config()
    t_cfg = cfg["train"]
    m_cfg = cfg["model"]
    f_cfg = cfg["features"]

    seed = t_cfg["seed"]
    set_seed(seed)
    device = get_device()

    runs_dir = PROJECT_ROOT / t_cfg["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data & Windows
    split_data = prepare_data()
    datasets, loaders = create_dataloaders(
        split_data,
        window=f_cfg["window"],
        batch_size=t_cfg["batch_size"],
        seed=seed,
    )

    # 2. Baselines
    baselines = evaluate_baselines(datasets, seed=seed)
    with open(runs_dir / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    print("=== Stage 4: Baselines ===")
    for b_name, b_Label in [
        ("always_up", "Always Up"),
        ("logreg", "LogReg (last day)"),
    ]:
        v = baselines[b_name]["val"]
        te = baselines[b_name]["test"]
        print(
            f"{b_Label:18s} | Val  Acc: {v['accuracy']:.4f}, F1: {v['f1']:.4f}, AUC: {v['roc_auc']:.4f} | "
            f"Test Acc: {te['accuracy']:.4f}, F1: {te['f1']:.4f}, AUC: {te['roc_auc']:.4f}"
        )

    # 3. Sanity check: overfit 1 small batch
    set_seed(seed)
    overfit_loss = sanity_check_overfit_one_batch(loaders["train"], m_cfg, device)
    print("\n=== Stage 5: Sanity Check ===")
    print(
        f"Single-batch overfit final loss (16 windows, 150 steps): {overfit_loss:.6f}"
    )

    # 4. Initialize model, loss, optimizer, scheduler
    set_seed(seed)
    model = GRUClassifier(
        input_size=m_cfg["input_size"],
        hidden_size=m_cfg["hidden_size"],
        num_layers=m_cfg["num_layers"],
        fc_size=m_cfg["fc_size"],
        dropout=m_cfg["dropout"],
    ).to(device)

    train_targets = np.array(
        [datasets["train"][i][1].item() for i in range(len(datasets["train"]))]
    )
    num_pos = float(train_targets.sum())
    num_neg = float(len(train_targets) - num_pos)
    if t_cfg.get("use_pos_weight", True) and num_pos > 0:
        pos_weight_val = num_neg / num_pos
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    writer = SummaryWriter(log_dir=str(runs_dir / "tb"))
    history: list[dict[str, float]] = []

    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = t_cfg["patience"]
    best_model_path = runs_dir / "best_model.pt"

    print(f"\n=== Training GRUClassifier on {device} ===")
    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += float(loss.item()) * xb.size(0)
            train_count += xb.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        _, train_metrics, _, _ = evaluate_loader(
            model, loaders["train"], criterion, device
        )
        val_loss, val_metrics, _, _ = evaluate_loader(
            model, loaders["val"], criterion, device
        )

        val_auc = val_metrics["roc_auc"]
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["roc_auc"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_auc,
        }
        history.append(row)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("AUC/train", train_metrics["roc_auc"], epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                best_model_path,
            )
            marker = " * [BEST]"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{t_cfg['epochs']:02d} | "
            f"Train Loss: {train_loss:.4f} (AUC: {train_metrics['roc_auc']:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_auc:.4f}{marker}"
        )

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, best val AUC: {best_val_auc:.4f})"
            )
            break

    writer.close()

    history_df = pd.DataFrame(history)
    history_df.to_csv(runs_dir / "training_log.csv", index=False)
    print(
        f"Saved best model weights to {best_model_path} and logs to {runs_dir / 'training_log.csv'}"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "baselines": baselines,
        "best_model_path": best_model_path,
    }


if __name__ == "__main__":
    train_pipeline()
