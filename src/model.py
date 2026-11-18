import torch
from torch import nn


class GRUClassifier(nn.Module):
    """GRU binary classifier for financial time-series windows.

    Architecture:
        - Input: [B, window, input_size] (e.g., [B, 30, 6])
        - GRU: hidden_size=64, num_layers=2, dropout=0.2, batch_first=True
        - Last hidden state from top layer: [B, 64]
        - Linear(64 -> 32) + ReLU + Dropout(0.2)
        - Linear(32 -> 1) -> output: single logit [B]
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        fc_size: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        # h_n: [num_layers, B, hidden_size]
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # [B, hidden_size]
        logits = self.head(last_hidden).squeeze(-1)  # [B]
        return logits


if __name__ == "__main__":
    model = GRUClassifier()
    dummy_x = torch.randn(16, 30, 6)
    out = model(dummy_x)
    print(model)
    print(f"\nInput shape: {tuple(dummy_x.shape)} -> Output shape: {tuple(out.shape)}")
