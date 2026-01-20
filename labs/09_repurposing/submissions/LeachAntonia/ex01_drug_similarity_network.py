"""
Exercise 9.1 — Drug–Gene Bipartite Network & Drug Similarity Network

Scop:
- să construiți o rețea bipartită drug–gene plecând de la un CSV
- să proiectați layer-ul de medicamente folosind similaritatea dintre seturile de gene
- să exportați un fișier cu muchiile de similaritate între medicamente

TODO:
- încărcați datele drug–gene
- construiți dict-ul drug -> set de gene țintă
- construiți graful bipartit drug–gene (NetworkX)
- calculați similaritatea dintre medicamente (ex. Jaccard)
- construiți graful drug similarity
- exportați tabelul cu muchii: drug1, drug2, weight
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List
import itertools
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# --------------------------
# Config
# --------------------------
HANDLE = "LeachAntonia"

DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_SUMMARY = OUT_DIR / f"drug_summary_{HANDLE}.csv"
OUT_DRUG_SIMILARITY = OUT_DIR / f"drug_similarity_{HANDLE}.csv"
OUT_GRAPH_DRUG_GENE = OUT_DIR / f"network_drug_gene_{HANDLE}.gpickle"
OUT_BIPARTITE_PNG = OUT_DIR / f"network_drug_gene_{HANDLE}.png"

MIN_SIMILARITY = 0.0

def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
        )
    print(f"Found input file: {path}")


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    required_cols = ['drug', 'gene']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    
    n_drugs = df['drug'].nunique()
    n_genes = df['gene'].nunique()
    n_interactions = len(df)

    return df


def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    drug2genes = df.groupby('drug')['gene'].apply(set).to_dict()
    
    target_counts = [len(genes) for genes in drug2genes.values()]
    return drug2genes


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    B = nx.Graph()
    
    # Add drug nodes
    for drug in drug2genes.keys():
        B.add_node(drug, bipartite='drug', node_type='drug')
    
    # Add gene nodes and edges
    all_genes = set()
    for drug, genes in drug2genes.items():
        for gene in genes:
            all_genes.add(gene)
            B.add_node(gene, bipartite='gene', node_type='gene')
            B.add_edge(drug, gene)
    
    print(f"Total nodes: {B.number_of_nodes()}")
    print(f"- Drug nodes: {len(drug2genes)}")
    print(f"- Gene nodes: {len(all_genes)}")
    print(f" Total edges: {B.number_of_edges()}")
    
    return B


def summarize_drugs(drug2genes: Dict[str, Set[str]]) -> pd.DataFrame:
    summary_data = []
    for drug, genes in drug2genes.items():
        summary_data.append({
            'drug': drug,
            'num_targets': len(genes),
            'targets': ','.join(sorted(genes))
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('num_targets', ascending=False)
    
    print(f"Top 5 drugs by target count:")
    for _, row in df_summary.head(5).iterrows():
        print(f"{row['drug']}: {row['num_targets']} targets")
    
    return df_summary


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    if not s1 and not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def compute_drug_similarity_edges(
    drug2genes: Dict[str, Set[str]],
    min_sim: float = 0.0,
) -> List[Tuple[str, str, float]]:
    drugs = list(drug2genes.keys())
    n_drugs = len(drugs)
    n_pairs = n_drugs * (n_drugs - 1) // 2
    
    edges = []
    similarities = []
    
    for d1, d2 in itertools.combinations(drugs, 2):
        sim = jaccard_similarity(drug2genes[d1], drug2genes[d2])
        similarities.append(sim)
        
        if sim >= min_sim and sim > 0:  # Only include non-zero similarities
            edges.append((d1, d2, sim))
    
    return edges


def edges_to_dataframe(edges: List[Tuple[str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(edges, columns=['drug1', 'drug2', 'similarity'])
    df = df.sort_values('similarity', ascending=False)
    return df


def build_drug_similarity_network(edges: List[Tuple[str, str, float]]) -> nx.Graph:
    G = nx.Graph()
    
    drugs = set()
    for d1, d2, _ in edges:
        drugs.add(d1)
        drugs.add(d2)
    
    G.add_nodes_from(drugs)
    
    for d1, d2, sim in edges:
        G.add_edge(d1, d2, weight=sim, similarity=sim)
    
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    # Network statistics
    if G.number_of_edges() > 0:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        print(f"Edge weights: min={min(weights):.3f}, max={max(weights):.3f}")
    
    return G


def visualize_bipartite_network(
    B: nx.Graph,
    drug2genes: Dict[str, Set[str]],
    out_png: Path,
    max_nodes: int = 50
) -> None:
    if B.number_of_nodes() > max_nodes:
        top_drugs = sorted(drug2genes.keys(), 
                          key=lambda d: len(drug2genes[d]), reverse=True)[:15]
        genes_to_keep = set()
        for d in top_drugs:
            genes_to_keep.update(drug2genes[d])
        
        B = B.subgraph(top_drugs + list(genes_to_keep)).copy()
    
    drugs = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 'drug']
    genes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 'gene']
    
    pos = {}
    for i, drug in enumerate(drugs):
        pos[drug] = (0, i)
    for i, gene in enumerate(genes):
        pos[gene] = (1, i * len(drugs) / max(len(genes), 1))
    
    pos = nx.spring_layout(B, seed=42, k=2)
    
    plt.figure(figsize=(14, 10))
    
    # Draw drug nodes
    drug_sizes = [300 + 50 * len(drug2genes.get(d, [])) for d in drugs]
    nx.draw_networkx_nodes(B, pos, nodelist=drugs, node_color='steelblue',
                          node_size=drug_sizes, alpha=0.8, label='Drugs')
    
    # Draw gene nodes
    nx.draw_networkx_nodes(B, pos, nodelist=genes, node_color='salmon',
                          node_size=200, alpha=0.7, label='Genes')
    
    # Draw edges
    nx.draw_networkx_edges(B, pos, alpha=0.3, edge_color='gray')
    
    nx.draw_networkx_labels(B, pos, font_size=8)
    
    plt.title("Bipartite Drug-Gene Network", fontsize=14, fontweight='bold')
    plt.legend(scatterpoints=1, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {out_png}")


def save_graph(G: nx.Graph, path: Path) -> None:
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved graph: {path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    ensure_exists(DRUG_GENE_CSV)
    
    df = load_drug_gene_table(DRUG_GENE_CSV)
    
    drug2genes = build_drug2genes(df)
    
    B = build_bipartite_graph(drug2genes)
    save_graph(B, OUT_GRAPH_DRUG_GENE)
    
    df_summary = summarize_drugs(drug2genes)
    df_summary.to_csv(OUT_DRUG_SUMMARY, index=False)
    print(f"\nSaved drug summary: {OUT_DRUG_SUMMARY}")
    
    edges = compute_drug_similarity_edges(drug2genes, min_sim=MIN_SIMILARITY)
    
    df_similarity = edges_to_dataframe(edges)
    df_similarity.to_csv(OUT_DRUG_SIMILARITY, index=False)
    print(f"\nSaved drug similarity edges: {OUT_DRUG_SIMILARITY}")
    
    if len(df_similarity) > 0:
        print(f"\nTop 10 most similar drug pairs:")
        for _, row in df_similarity.head(10).iterrows():
            print(f"{row['drug1']} - {row['drug2']}: {row['similarity']:.3f}")
    
    G_sim = build_drug_similarity_network(edges)
    
    visualize_bipartite_network(B, drug2genes, OUT_BIPARTITE_PNG)

if __name__ == "__main__":
    main()
