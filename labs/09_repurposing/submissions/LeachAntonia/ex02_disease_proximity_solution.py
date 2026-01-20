"""
Exercise 9.2 — Disease Proximity and Drug Ranking

Scop:
- să calculați distanța medie dintre fiecare medicament și un set de gene asociate unei boli
- să ordonați medicamentele în funcție de proximitate (network-based prioritization)

TODO-uri principale:
- încărcați graful bipartit drug–gene (din exercițiul 9.1) sau reconstruiți-l
- încărcați lista de disease genes
- pentru fiecare medicament, calculați distanța minimă / medie până la genele bolii
- exportați un tabel cu medicamente și scorul lor de proximitate
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List, Tuple
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------
# Config
# --------------------------
HANDLE = "LeachAntonia"

GRAPH_DRUG_GENE = Path(f"labs/09_repurposing/submissions/{HANDLE}/network_drug_gene_{HANDLE}.gpickle")
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

DISEASE_GENES_TXT = Path(f"data/work/{HANDLE}/lab09/disease_genes_{HANDLE}.txt")

OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_PRIORITY = OUT_DIR / f"drug_priority_{HANDLE}.csv"
OUT_REPURPOSING_REPORT = OUT_DIR / f"REPURPOSING_{HANDLE}.csv"

MAX_DISTANCE = 10


# --------------------------
# Utils
# --------------------------

def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
        )
    print(f"Found: {path}")


def load_bipartite_graph_or_build() -> nx.Graph:
    if GRAPH_DRUG_GENE.exists():
        print(f"Loading from pickle: {GRAPH_DRUG_GENE}")
        with open(GRAPH_DRUG_GENE, 'rb') as f:
            B = pickle.load(f)
        print(f"Loaded: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
        return B
    
    print(f"  Pickle not found, rebuilding from CSV...")
    ensure_exists(DRUG_GENE_CSV)
    
    df = pd.read_csv(DRUG_GENE_CSV)
    
    B = nx.Graph()
    
    # Add drug nodes
    drugs = df['drug'].unique()
    for drug in drugs:
        B.add_node(drug, bipartite='drug', node_type='drug')
    
    # Add gene nodes and edges
    for _, row in df.iterrows():
        drug, gene = row['drug'], row['gene']
        B.add_node(gene, bipartite='gene', node_type='gene')
        B.add_edge(drug, gene)
    
    print(f"  Built graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    
    return B


def load_disease_genes(path: Path) -> Set[str]:
    with open(path, 'r') as f:
        genes = set(line.strip() for line in f if line.strip())
    
    print(f"Loaded {len(genes)} disease genes")
    print(f"Genes: {sorted(genes)}")
    
    return genes


def get_drug_nodes(B: nx.Graph) -> List[str]:
    drugs = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 'drug']
    return drugs


def get_gene_nodes(B: nx.Graph) -> List[str]:
    genes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 'gene']
    return genes


def compute_drug_disease_distance(
    B: nx.Graph,
    drug: str,
    disease_genes: Set[str],
    mode: str = "mean",
    max_dist: int = MAX_DISTANCE,
) -> Tuple[float, dict]:
    distances = {}
    
    for gene in disease_genes:
        if gene not in B:
            distances[gene] = max_dist + 1
        elif drug not in B:
            distances[gene] = max_dist + 1
        else:
            try:
                d = nx.shortest_path_length(B, drug, gene)
                distances[gene] = d
            except nx.NetworkXNoPath:
                distances[gene] = max_dist + 1
    
    # Aggregate distances
    dist_values = list(distances.values())
    
    if mode == "mean":
        agg_dist = np.mean(dist_values)
    elif mode == "min":
        agg_dist = np.min(dist_values)
    elif mode == "sum":
        agg_dist = np.sum(dist_values)
    elif mode == "median":
        agg_dist = np.median(dist_values)
    else:
        agg_dist = np.mean(dist_values)
    
    return agg_dist, distances


def rank_drugs_by_proximity(
    B: nx.Graph,
    disease_genes: Set[str],
    mode: str = "mean",
) -> pd.DataFrame:
    print(f"\nRanking drugs by disease proximity (mode={mode})...")
    
    drugs = get_drug_nodes(B)
    print(f"Evaluating {len(drugs)} drugs against {len(disease_genes)} disease genes")
    
    results = []
    
    for drug in drugs:
        agg_dist, details = compute_drug_disease_distance(
            B, drug, disease_genes, mode=mode
        )
        
        dist_values = list(details.values())
        reachable = sum(1 for d in dist_values if d <= MAX_DISTANCE)
        direct_targets = sum(1 for d in dist_values if d == 1)  # Distance 1 = direct target
        
        results.append({
            'drug': drug,
            'distance': agg_dist,
            'min_distance': min(dist_values),
            'max_distance': max(dist_values),
            'mean_distance': np.mean(dist_values),
            'reachable_genes': reachable,
            'direct_targets': direct_targets,
            'total_disease_genes': len(disease_genes)
        })
    
    df = pd.DataFrame(results)
    
    # Sort by distance (ascending)
    df = df.sort_values('distance', ascending=True)
    df['rank'] = range(1, len(df) + 1)
    
    cols = ['rank', 'drug', 'distance', 'min_distance', 'mean_distance', 
            'direct_targets', 'reachable_genes', 'total_disease_genes']
    df = df[cols]
    
    return df


def generate_repurposing_report(
    df_ranking: pd.DataFrame,
    disease_genes: Set[str],
    out_csv: Path
) -> None:
    df = df_ranking.copy()
    
    def categorize(row):
        if row['direct_targets'] > 0:
            return "HIGH - Direct target"
        elif row['min_distance'] <= 2:
            return "MEDIUM - Close proximity"
        elif row['min_distance'] <= 4:
            return "LOW - Moderate proximity"
        else:
            return "VERY LOW - Distant"
    
    df['priority'] = df.apply(categorize, axis=1)
    
    df['disease_coverage'] = (df['reachable_genes'] / df['total_disease_genes'] * 100).round(1)
    
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return df


# --------------------------
# Main
# --------------------------

def main():
    B = load_bipartite_graph_or_build()
    
    ensure_exists(DISEASE_GENES_TXT)
    disease_genes = load_disease_genes(DISEASE_GENES_TXT)
    
    graph_genes = set(get_gene_nodes(B))
    overlap = disease_genes & graph_genes
    print(f"\nDisease genes in network: {len(overlap)}/{len(disease_genes)}")
    
    if not overlap:
        print("Make sure disease gene names match the network gene names.")
    
    df_ranking = rank_drugs_by_proximity(B, disease_genes, mode="mean")
    
    df_ranking.to_csv(OUT_DRUG_PRIORITY, index=False)
    print(f"\nSaved drug priority ranking: {OUT_DRUG_PRIORITY}")

    df_report = generate_repurposing_report(df_ranking, disease_genes, OUT_REPURPOSING_REPORT)

if __name__ == "__main__":
    main()
