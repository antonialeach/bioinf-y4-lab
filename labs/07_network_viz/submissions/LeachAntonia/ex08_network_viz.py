"""
Exercițiu 8 — Vizualizarea rețelelor de co-expresie + gene hub

Obiectiv:
- Incărcați modulele detectate în Lab 6 și reconstruiți o rețea (din corelații)
- Vizualizați graful, colorând nodurile după modul
- Evidențiați genele hub (grad mare) și exportați figura (.png)

Intrări:
- Matricea de expresie folosită în Lab 6: data/work/<handle>/lab06/expression_matrix.csv
- Mapping gene→modul din Lab 6: labs/06_networks/submissions/<handle>/modules_<handle>.csv

Ieșiri:
- labs/07_networkviz/submissions/<handle>/network.png
- labs/07_networkviz/submissions/<handle>/hubs.csv  (opțional, listă gene hub)

Notă:
- Dacă aveți deja o matrice de adiacență salvată din Lab 6, o puteți încărca în loc să o reconstruiți.
- În acest exercițiu ne concentrăm pe VIZUALIZARE (nu refacem detectarea de module).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



HANDLE = "LeachAntonia"

# Fișiere de intrare (aceleași ca în Lab 6)
EXPR_CSV = Path(f"data/work/{HANDLE}/lab06/expression_matrix.csv")
MODULES_CSV = Path(f"labs/06_wgcna/submissions/{HANDLE}/modules_{HANDLE}.csv")

# (Opțional) încărcați o adiacență pregătită; altfel, reconstruiți din corelații
PRECOMPUTED_ADJ_CSV: Optional[Path] = None  # ex: Path(f"labs/06_networks/submissions/{HANDLE}/adj_{HANDLE}.csv")

# Parametri pentru reconstrucția adiacenței (dacă nu aveți CSV)
CORR_METHOD = "spearman"   # "pearson" sau "spearman"
USE_ABS_CORR = True        # True => folosiți |cor|
ADJ_THRESHOLD = 0.6      # prag pentru |cor| (ex: 0.6)
WEIGHTED = False           # False => 0/1; True => păstrează valorile corr peste prag

# Parametri de vizualizare
SEED = 42                  # pentru layout determinist
TOPK_HUBS = 10             # câte gene hub etichetăm (după grad)
NODE_BASE_SIZE = 60        # mărimea de bază a nodurilor
MAX_NODES_TO_PLOT = 1500
EDGE_ALPHA = 0.3         # transparența muchiilor

# Ieșiri
OUT_DIR = Path(f"labs/07_network_viz/submissions/{HANDLE}")
OUT_PNG = OUT_DIR / f"network_{HANDLE}.png"
OUT_HUBS = OUT_DIR / f"hubs_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Nu am găsit: {path}")

def read_modules_csv(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path)
    return dict(zip(df["Gene"].astype(str), df["Module"].astype(int)))

def load_expression_filtered(path: Path, genes: Set[str]) -> pd.DataFrame:
    """
    Citește matricea de expresie DOAR pentru genele din module.
    """
    chunks = []

    for chunk in pd.read_csv(path, index_col=0, chunksize=2000):
        common = chunk.index.intersection(genes)
        if not common.empty:
            subset = chunk.loc[common].astype(np.float32)
            np.log2(subset + 1, out=subset)
            chunks.append(subset)

    if not chunks:
        raise ValueError("Nu s-au găsit gene comune între expresie și module.")

    return pd.concat(chunks)


def build_graph(expr: pd.DataFrame, threshold: float) -> nx.Graph:
    """
    Construiește graful direct din corelații (fără matrice de adiacență).
    """
    corr = expr.T.corr(method=CORR_METHOD).abs().values
    genes = expr.index.to_numpy()

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows, cols = np.where((corr >= threshold) & mask)

    G = nx.Graph()
    G.add_nodes_from(genes)
    G.add_edges_from(zip(genes[rows], genes[cols]))

    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def get_hubs(G: nx.Graph, topk: int) -> pd.DataFrame:
    deg = dict(G.degree())
    hubs = (
        pd.DataFrame(deg.items(), columns=["Gene", "Degree"])
        .sort_values("Degree", ascending=False)
        .head(topk)
    )
    return hubs


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(EXPR_CSV)
    ensure_exists(MODULES_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gene2module = read_modules_csv(MODULES_CSV)
    target_genes = set(gene2module.keys())

    expr = load_expression_filtered(EXPR_CSV, target_genes)

    G = build_graph(expr, ADJ_THRESHOLD)
    print(f"Graf inițial: {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii")

    if G.number_of_nodes() > MAX_NODES_TO_PLOT:
        deg = dict(G.degree())
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:MAX_NODES_TO_PLOT]
        G = G.subgraph(top_nodes).copy()

    hubs_df = get_hubs(G, TOPK_HUBS)
    hub_set = set(hubs_df["Gene"])

    cmap = plt.get_cmap("tab10")
    node_colors = [cmap((gene2module.get(n, 0) - 1) % 10) for n in G.nodes()]
    node_sizes = [150 if n in hub_set else 30 for n in G.nodes()]

    pos = nx.spring_layout(G, seed=SEED)

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in hub_set if n in G.nodes()},
        font_size=8
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    hubs_df.to_csv(OUT_HUBS, index=False)

    print(f"Salvat: {OUT_PNG}")
    print(f"Salvat: {OUT_HUBS}")