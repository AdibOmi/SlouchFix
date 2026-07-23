"""Linear baseline: logistic regression for the posture class, linear
regression for distance. This is the "beat this" bar from the tech stack
doc before the MLP/RF/XGBoost results can be called an improvement.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def train(x_train: np.ndarray, y_label_train: np.ndarray, y_distance_train: np.ndarray) -> dict:
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(x_train, y_label_train)

    regressor = LinearRegression()
    regressor.fit(x_train, y_distance_train)

    return {"classifier": classifier, "regressor": regressor}


def predict(models: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = models["classifier"].predict(x)
    distances = models["regressor"].predict(x)
    return labels, distances
