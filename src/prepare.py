import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from data import load_data


def prepare():
    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # X_test kept in physical space for attack initialisation
    return X_train_scaled, X_test_scaled, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_tr, X_te, X_te_phys, y_tr, y_te, scaler = prepare()
    print(f"Train scaled  : {X_tr.shape}  mean={X_tr.mean():.4f}  std={X_tr.std():.4f}")
    print(f"Test  scaled  : {X_te.shape}  mean={X_te.mean():.4f}  std={X_te.std():.4f}")
    print(f"Test  physical: {X_te_phys.shape}")
    print(f"Stable ratio  — train: {y_tr.mean():.3f}, test: {y_te.mean():.3f}")
    print(f"Scaler saved  -> outputs/scaler.pkl")
