"""
Exercise 8 — Supervised ML pipeline pentru expresie genică (Random Forest)

TODO-uri principale:
- Încărcați matricea de expresie (ex. subset TP53 / GTEx) pentru HANDLE-ul vostru
- Separați features (gene) și label (ultima coloană)
- Encodați etichetele
- Împărțiți în train/test
- Antrenați un RandomForestClassifier (model de bază)
- Evaluați: classification_report + matrice de confuzie (salvate)
- Calculați importanța trăsăturilor și salvați în CSV
- (Opțional) Aplicați KMeans pe X și comparați clustere vs etichete reale
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Config — completați cu valorile voastre
# --------------------------

HANDLE = "LeachAntonia"

DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
TOPK_FEATURES = 20

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / f"confusion_rf_{HANDLE}.png"
OUT_REPORT = OUT_DIR / f"classification_report_{HANDLE}.txt"
OUT_FEATIMP = OUT_DIR / f"feature_importance_{HANDLE}.csv"
OUT_CLUSTER_CROSSTAB = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------

def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
        )
    print(f"Found input file: {path}")


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    for i, cls in enumerate(le.classes_):
        count = (y_enc == i).sum()
        print(f"  {cls} → {i} ({count} samples)")
    
    return y_enc, le


def split_data(
    X: pd.DataFrame, 
    y_enc: np.ndarray, 
    test_size: float, 
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc
    )
    
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> RandomForestClassifier:
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    
    rf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"  Training accuracy: {train_acc:.4f}")
    
    return rf


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_png: Path,
    out_txt: Path,
) -> dict:
    
    # Predictions
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_
    
    # Test accuracy
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\n=== Classification Report ===")
    print(report)

    out_txt.write_text(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Random Forest Confusion Matrix\n(Accuracy: {test_acc:.3f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {out_png}")
    
    return {
        'accuracy': test_acc,
        'predictions': y_pred,
        'confusion_matrix': cm
    }


def compute_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    out_csv: Path,
    top_k: int = 20,
) -> pd.DataFrame:
    importances = model.feature_importances_
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    df_imp["Rank"] = range(1, len(df_imp) + 1)
    df_imp = df_imp[["Rank", "Feature", "Importance"]]
    
    df_imp.to_csv(out_csv, index=False)
    print(f"Saved feature importance: {out_csv}")
    
    print(f"\n  Top {top_k} most important features (genes):")
    for _, row in df_imp.head(top_k).iterrows():
        print(f"    {row['Rank']:3d}. {row['Feature']:15s} : {row['Importance']:.4f}")
    
    return df_imp


def run_kmeans_and_crosstab(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    label_encoder: LabelEncoder,
    n_clusters: int,
    out_csv: Path,
) -> pd.DataFrame:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init="auto"
    )
    clusters = kmeans.fit_predict(X.values)
    
    # Crosstab
    df_comparison = pd.DataFrame({
        "Label": label_encoder.inverse_transform(y_enc),
        "Cluster": clusters
    })
    
    ctab = pd.crosstab(df_comparison["Label"], df_comparison["Cluster"])
    ctab.columns = [f"Cluster_{c}" for c in ctab.columns]
    
    print("\nCrosstab Label vs Cluster:")
    print(ctab)
    
    ctab.to_csv(out_csv)
    print(f"Saved cluster crosstab: {out_csv}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    ensure_exists(DATA_CSV)
    
    X, y = load_dataset(DATA_CSV)
    
    y_enc, le = encode_labels(y)

    X_train, X_test, y_train, y_test = split_data(
        X, y_enc, TEST_SIZE, RANDOM_STATE
    )
    
    rf = train_random_forest(X_train, y_train, N_ESTIMATORS, RANDOM_STATE)
    
    eval_results = evaluate_model(rf, X_test, y_test, le, OUT_CONFUSION, OUT_REPORT)
    
    feat_imp_df = compute_feature_importance(
        rf, X.columns, OUT_FEATIMP, TOPK_FEATURES
    )
    
    n_classes = len(le.classes_)
    ctab = run_kmeans_and_crosstab(
        X, y_enc, le, n_clusters=n_classes,
        out_csv=OUT_CLUSTER_CROSSTAB
    )

    print("Files for exercise 8 are saved")


if __name__ == "__main__":
    main()
