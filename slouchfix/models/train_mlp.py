"""Core neural model: a small feed-forward net (per the tech stack doc)
mapping standardized landmark features to a posture class and a distance
regression, trained end-to-end with backprop. Cross-entropy for the class
head, Huber/SmoothL1 for the distance head, Adam with a plateau LR
scheduler.

Expects `x_train`/`x_val` to already be standardized (see
`dataset.fit_standard_scaler`) -- scaling is done once in the training
orchestrator so every model variant that wants standardized input shares
the exact same scaler.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .. import config


class PostureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_classes: int = len(config.LABELS)) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.classifier_head = nn.Linear(hidden, num_classes)
        self.distance_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.classifier_head(h), self.distance_head(h).squeeze(-1)


def train(
    x_train: np.ndarray,
    y_label_train: np.ndarray,
    y_distance_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_label_val: np.ndarray | None = None,
    y_distance_val: np.ndarray | None = None,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    distance_loss_weight: float = 0.02,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    label_encoder = LabelEncoder()
    label_encoder.fit(config.LABELS)  # fixed ordering regardless of which labels appear in this split

    model = PostureMLP(in_dim=x_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    ce_loss = nn.CrossEntropyLoss()
    huber_loss = nn.SmoothL1Loss()

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_cls_t = torch.tensor(label_encoder.transform(y_label_train), dtype=torch.long)
    y_dist_t = torch.tensor(y_distance_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_cls_t, y_dist_t), batch_size=batch_size, shuffle=True)

    has_val = x_val is not None and len(x_val) > 0
    if has_val:
        xv_t = torch.tensor(x_val, dtype=torch.float32)
        yv_cls_t = torch.tensor(label_encoder.transform(y_label_val), dtype=torch.long)
        yv_dist_t = torch.tensor(y_distance_val, dtype=torch.float32)

    for _epoch in range(epochs):
        model.train()
        for xb, yb_cls, yb_dist in loader:
            optimizer.zero_grad()
            logits, dist_pred = model(xb)
            loss = ce_loss(logits, yb_cls) + distance_loss_weight * huber_loss(dist_pred, yb_dist)
            loss.backward()
            optimizer.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                logits, dist_pred = model(xv_t)
                val_loss = ce_loss(logits, yv_cls_t) + distance_loss_weight * huber_loss(dist_pred, yv_dist_t)
            scheduler.step(val_loss.item())

    return {"model": model, "label_encoder": label_encoder}


def predict(models: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = models["model"]
    model.eval()
    with torch.no_grad():
        logits, dist_pred = model(torch.tensor(x, dtype=torch.float32))
        pred_idx = logits.argmax(dim=1).numpy()
    labels = models["label_encoder"].inverse_transform(pred_idx)
    distances = dist_pred.numpy()
    return labels, distances
