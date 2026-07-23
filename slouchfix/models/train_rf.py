"""Random Forest: classifier for posture label, regressor for distance_cm."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train(
    x_train: np.ndarray,
    y_label_train: np.ndarray,
    y_distance_train: np.ndarray,
    n_estimators: int = 300,
    seed: int = 42,
) -> dict:
    classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    classifier.fit(x_train, y_label_train)

    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    regressor.fit(x_train, y_distance_train)

    return {"classifier": classifier, "regressor": regressor}


def predict(models: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = models["classifier"].predict(x)
    distances = models["regressor"].predict(x)
    return labels, distances
