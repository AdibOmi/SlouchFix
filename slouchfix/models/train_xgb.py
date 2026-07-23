"""XGBoost: classifier for posture label, regressor for distance_cm."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


def train(
    x_train: np.ndarray,
    y_label_train: np.ndarray,
    y_distance_train: np.ndarray,
    seed: int = 42,
) -> dict:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_label_train)

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    classifier.fit(x_train, y_encoded)

    regressor = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )
    regressor.fit(x_train, y_distance_train)

    return {"classifier": classifier, "regressor": regressor, "label_encoder": label_encoder}


def predict(models: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = models["classifier"].predict(x)
    labels = models["label_encoder"].inverse_transform(encoded)
    distances = models["regressor"].predict(x)
    return labels, distances
