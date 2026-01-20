"""
Exercise 8b — Logistic Regression vs Random Forest pe expresie genică

Scop:
- să antrenăm și să comparăm două modele:
  - Logistic Regression (multiclass, liniar)
  - Random Forest (non-liniar, bazat pe arbori)
- să vedem dacă performanța și erorile sunt similare sau diferite

TODO:
- Încărcați expresia pentru HANDLE
- Împărțiți în X (gene) și y (Label)
- Encodați etichetele
- Împărțiți în train/test
- Scalați features pentru logistic regression
- Antrenați RF și Logistic Regression
- Comparați classification_report pentru ambele modele
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Config
# --------------------------
HANDLE = "LeachAntonia"

DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_ITER_LOGREG = 1000

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT_TXT = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"File found: {path}")


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, LogisticRegression, StandardScaler]:
    """
    Antrenează RF (pe date raw) și LogReg (pe date scalate).
    """
    # 1. Scalare (necesară pentru Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 2. Antrenare Random Forest
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 3. Antrenare Logistic Regression
    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=MAX_ITER_LOGREG,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)

    return rf, logreg, scaler


def compare_models(
    rf: RandomForestClassifier,
    logreg: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_txt: Path,
) -> None:
    X_test_scaled = scaler.transform(X_test)

    y_pred_rf = rf.predict(X_test)
    y_pred_logreg = logreg.predict(X_test_scaled)

    target_names = [str(cls) for cls in label_encoder.classes_]

    report_rf = classification_report(y_test, y_pred_rf, target_names=target_names)
    report_logreg = classification_report(y_test, y_pred_logreg, target_names=target_names)

    print("=== Random Forest ===")
    print(report_rf)
    print("\n=== Logistic Regression ===")
    print(report_logreg)

    combined_text = (
        f"Exercise 8b Comparison Report\nHandle: {HANDLE}\n\n"
        f"=== Random Forest ===\n{report_rf}\n\n"
        f"=== Logistic Regression ===\n{report_logreg}\n"
    )
    out_txt.write_text(combined_text)
    print(f"Report saved in: {out_txt}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DATA_CSV)

    X, y = load_dataset(DATA_CSV)

    y_enc, le = encode_labels(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    rf, logreg, scaler = train_models(X_train, y_train)

    compare_models(rf, logreg, scaler, X_test, y_test, le, OUT_REPORT_TXT)

    print("Files for exercise 8 are saved")