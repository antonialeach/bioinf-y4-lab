"""
Exercise 10.2 — Identify top SNP–Gene correlations

TODO:
- încărcați matricea integrată multi-omics
- împărțiți rândurile în SNPs vs gene (după indice sau după nume)
- calculați corelații între fiecare SNP și fiecare genă
- filtrați |r| > 0.5
- exportați snp_gene_pairs_<handle>.csv
"""

from pathlib import Path
import pandas as pd

HANDLE = "LeachAntonia"

IN_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/multiomics_concat_{HANDLE}.csv")
OUT_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/snp_gene_pairs_{HANDLE}.csv")

def main():
    if not IN_CSV.exists():
        print(f"Error: Input file not found at {IN_CSV}")
        return

    print(f"Loading {IN_CSV}.")
    df = pd.read_csv(IN_CSV, index_col=0)

    snp_mask = df.index.str.startswith("rs")
    df_snps = df[snp_mask]
    df_genes = df[~snp_mask]

    print(f"Identified {len(df_snps)} SNPs and {len(df_genes)} Genes.")

    print("Calculating correlations.")
    
    full_corr = df.T.corr() 
    snp_gene_corr = full_corr.loc[df_snps.index, df_genes.index]

    pairs = snp_gene_corr.unstack().reset_index()
    pairs.columns = ['Gene', 'SNP', 'correlation']

    significant_pairs = pairs[pairs['correlation'].abs() > 0.5].copy()
    
    significant_pairs = significant_pairs.sort_values('correlation', key=abs, ascending=False)

    print(f"Found {len(significant_pairs)} pairs with |r| > 0.5")
    significant_pairs.to_csv(OUT_CSV, index=False)
    print(f"Saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()