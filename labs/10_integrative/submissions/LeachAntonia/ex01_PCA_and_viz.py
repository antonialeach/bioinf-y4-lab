"""
Exercise 10 — PCA Single-Omics vs Joint

TODO:
- încărcați SNP și Expression
- normalizați fiecare strat (z-score)
- rulați PCA pe:
    1) strat SNP
    2) strat Expression
    3) strat Joint (concat)
- generați 3 figuri PNG
- comparați vizual distribuția probelor
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

HANDLE = "LeachAntonia"

SNP_CSV = Path(f"data/work/{HANDLE}/lab10/snp_matrix_{HANDLE}.csv")
EXP_CSV = Path(f"data/work/{HANDLE}/lab10/expression_matrix_{HANDLE}.csv")
OUT_DIR = Path(f"labs/10_integrative/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(snp_path: Path, exp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_snp = pd.read_csv(snp_path, index_col=0)
    df_exp = pd.read_csv(exp_path, index_col=0)
    
    return df_snp, df_exp

def align_samples(df_snp: pd.DataFrame, df_exp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_samples = df_snp.columns.intersection(df_exp.columns)
    
    df_snp_aligned = df_snp[common_samples]
    df_exp_aligned = df_exp[common_samples]
    
    return df_snp_aligned, df_exp_aligned

def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-score normalization per feature.
    z = (x - mean) / std
    """
    df_norm = (df - df.mean(axis=1).values[:, np.newaxis]) / df.std(axis=1).values[:, np.newaxis]
    
    df_norm = df_norm.fillna(0)
    
    return df_norm


def create_joint_matrix(df_snp: pd.DataFrame, df_exp: pd.DataFrame) -> pd.DataFrame:
    """Concatenate SNP and expression matrices vertically."""
    df_joint = pd.concat([df_snp, df_exp], axis=0)
    return df_joint


def run_pca(df: pd.DataFrame, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components)
    projection = pca.fit_transform(df.T)
    
    return projection, pca


def plot_pca(projection: np.ndarray, pca: PCA, title: str, output_path: Path, 
             color: str = "blue", sample_labels: list = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(projection[:, 0], projection[:, 1], 
                         c=color, alpha=0.7, edgecolors='black', linewidth=0.5, s=100)
    
    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            ax.annotate(label, (projection[i, 0], projection[i, 1]), 
                       fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    var_explained = pca.explained_variance_ratio_ * 100
    
    ax.set_xlabel(f"PC1 ({var_explained[0]:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.2f}% variance)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"- PC1 explains {var_explained[0]:.2f}% variance")
    print(f"- PC2 explains {var_explained[1]:.2f}% variance")
    print(f"- Total: {sum(var_explained):.2f}%\n")


def main():
    df_snp, df_exp = load_data(SNP_CSV, EXP_CSV)


    df_snp, df_exp = align_samples(df_snp, df_exp)
    sample_labels = list(df_snp.columns)

    df_snp_norm = zscore_normalize(df_snp)
    df_exp_norm = zscore_normalize(df_exp)
    print("SNP matrix normalized")
    print("Expression matrix normalized\n")
    
    df_joint = create_joint_matrix(df_snp_norm, df_exp_norm)
    
    joint_csv_path = OUT_DIR / f"multiomics_concat_{HANDLE}.csv"
    df_joint.to_csv(joint_csv_path)
    print(f"Saved joint matrix: {joint_csv_path}\n")
    
    proj_snp, pca_snp = run_pca(df_snp_norm)
    
    proj_exp, pca_exp = run_pca(df_exp_norm)
    
    proj_joint, pca_joint = run_pca(df_joint)
    
    plot_pca(proj_snp, pca_snp, 
             "PCA on SNP Data Only", 
             OUT_DIR / f"pca_snp_{HANDLE}.png",
             color="crimson", sample_labels=sample_labels)
    
    plot_pca(proj_exp, pca_exp, 
             "PCA on Expression Data Only", 
             OUT_DIR / f"pca_expression_{HANDLE}.png",
             color="forestgreen", sample_labels=sample_labels)
    
    plot_pca(proj_joint, pca_joint, 
             "PCA on Joint Multi-Omics Data", 
             OUT_DIR / f"pca_joint_{HANDLE}.png",
             color="royalblue", sample_labels=sample_labels)
    
    print("Exercises 10 done")


if __name__ == "__main__":
    main()
