from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np

SEED = 42

FEATURE_BOUNDS = {
    "p2": (-2.0, -0.5),
    "p3": (-2.0, -0.5),
    "p4": (-2.0, -0.5),
    "tau1": (0.5, 10.0),
    "tau2": (0.5, 10.0),
    "tau3": (0.5, 10.0),
    "tau4": (0.5, 10.0),
    "g1": (0.05, 1.00),
    "g2": (0.05, 1.00),
    "g3": (0.05, 1.00),
    "g4": (0.05, 1.00),
}


def load_data():
    dataset = fetch_ucirepo(id=471)
    X = dataset.data.features
    y_raw = dataset.data.targets

    # "stabf" is the categorical label: "stable" / "unstable"
    y = (y_raw["stabf"] == "stable").astype(int).values
    X = X.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Stable ratio — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
