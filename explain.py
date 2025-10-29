import glob
import logging
import sys
import torch
import dgl

from core.model import Model

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from dgl import to_networkx
import numpy as np

from processor import process_data

def load_latest_checkpoint(model, device):
    model_files = sorted(glob.glob('./tmp/model/*.pt'))
    if not model_files:
        raise FileNotFoundError("No model checkpoint found in ./tmp/model/")
    latest = model_files[-1]
    logger.info(f"Loading model checkpoint: {latest}")
    model.load_state_dict(torch.load(latest, map_location=device, weights_only=False))
    model.eval()
    return model

def compute_edge_mask(alphas, reduce_heads='max', reduce_layers='max'):
    alpha = alphas.squeeze(-1)  # [num_edges, num_layers, num_heads]

    def apply_reduce(x, reduce, dim):
        fn = getattr(torch, reduce)
        result = fn(x, dim=dim)
        # handle (value, index)
        return result[0] if isinstance(result, tuple) else result

    alpha = apply_reduce(alpha, reduce_heads, dim=-1)  # [num_edges, num_layers]
    alpha = apply_reduce(alpha, reduce_layers, dim=-1) # [num_edges]

    return alpha

def remove_self_loops(graph: dgl.DGLGraph, edge_mask: torch.Tensor):
    src, dst = graph.edges(order='eid')
    non_self_loop_mask = src != dst
    filtered_edge_mask = edge_mask[non_self_loop_mask]
    edges_to_keep = non_self_loop_mask.nonzero(as_tuple=False).squeeze()
    filtered_graph = dgl.edge_subgraph(graph, edges_to_keep, relabel_nodes=False)

    return filtered_graph, filtered_edge_mask

def get_important_edges(edge_mask: torch.Tensor, percentile: float):
    threshold = torch.quantile(edge_mask, percentile)
    top_indices = (edge_mask >= threshold).nonzero(as_tuple=True)[0]
    top_values = edge_mask[top_indices]

    sorted_order = torch.argsort(top_values, descending=True)

    top_values = top_values[sorted_order]
    top_indices = top_indices[sorted_order]

    return top_values, top_indices

def threshold_edge_mask(edge_mask: torch.Tensor, percentile: float):
    threshold = torch.quantile(edge_mask, percentile)
    mask = edge_mask.clone()
    mask[mask < threshold] = 0.0
    return mask

def visualize_graph_explanation(
    graph: dgl.DGLGraph,
    edge_mask: torch.Tensor,
    node_labels: dict = None,
    figsize=(10, 8),
    title='Explanation',
    output_path=None,
    only_explained_subgraph: bool = False
):
    G_full = to_networkx(graph)
    src, dst = graph.edges(order='eid')
    edge_index_map = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist()))}

    explained_edges = []
    explained_nodes = set()

    for u, v in G_full.edges():
        eid = edge_index_map.get((u, v)) or edge_index_map.get((v, u))
        importance = edge_mask[eid].item() if eid is not None else 0.0

        if importance > 0:
            explained_edges.append((u, v))
            explained_nodes.update([u, v])

    if only_explained_subgraph:
        G = nx.DiGraph()
        for u, v in explained_edges:
            G.add_edge(u, v)
    else:
        G = G_full

    pos = graphviz_layout(G, prog='fdp')
    plt.figure(figsize=figsize)

    # Node drawing
    node_color = 'skyblue'
    node_size = 100
    if only_explained_subgraph:
        nx.draw_networkx_nodes(G, pos, nodelist=explained_nodes, node_color=node_color, node_size=node_size)
    else:
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)

    if node_labels is not None:
        labels = { i: node_labels[i] for i in G.nodes() if i < len(node_labels) }

        # selected = {9,21,23,42,26}
        # labels = {
        #     i: node_labels[i] for i in G.nodes() if i in selected and i < len(node_labels)
        # }

        if only_explained_subgraph:
            labels = {k: v for k, v in labels.items() if k in explained_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))

    # Edge drawing
    edge_colors = []
    edge_widths = []
    edge_styles = []
    edges_to_draw = []

    for u, v in G.edges():
        eid = edge_index_map.get((u, v)) or edge_index_map.get((v, u))
        importance = edge_mask[eid].item() if eid is not None else 0.0

        if importance > 0:
            edge_colors.append('red')
            edge_widths.append(1.5 + 3.0 * importance / edge_mask.max().item())
        else:
            if only_explained_subgraph:
                continue
            edge_colors.append('black')
            edge_widths.append(1.0)

        edge_styles.append('solid')
        edges_to_draw.append((u, v))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_to_draw,
        edge_color=edge_colors,
        width=edge_widths,
        style='solid'
    )

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


def calculate_metrics(model, graph: dgl.DGLGraph, edge_mask, device='cpu'): 
    model.eval()

    # original prediction
    with torch.no_grad():
        graph_original = graph.clone()
        graph_original = dgl.add_self_loop(graph_original).to(device)
        _, original_logits = model(graph_original)
        original_class = torch.argmax(original_logits, dim=1).item()

    # keep only important edges (fidelity+)
    boolean_mask = edge_mask > 0
    graph_plus = graph.clone()
    graph_plus = dgl.edge_subgraph(graph_plus, boolean_mask, relabel_nodes=False).to(device)

    with torch.no_grad():
        graph_plus = dgl.add_self_loop(graph_plus).to(device)
        _, logits_plus = model(graph_plus)
        pred_plus = torch.argmax(logits_plus, dim=1).item()

    fidelity_plus = int(pred_plus == original_class)

    # remove important edges (fidelity−)
    inverse_mask = ~boolean_mask
    graph_minus = graph.clone()
    graph_minus = dgl.edge_subgraph(graph_minus, inverse_mask, relabel_nodes=False).to(device)

    with torch.no_grad():
        graph_minus = dgl.add_self_loop(graph_minus).to(device)
        _, logits_minus = model(graph_minus)
        pred_minus = torch.argmax(logits_minus, dim=1).item()

    fidelity_minus = int(pred_minus != original_class)

    # sparsity
    total_edges = graph.num_edges()
    selected_edges = boolean_mask.sum().item()
    sparsity = 1 - (selected_edges / total_edges)

    # attack edge ratio
    attack_edge_ratio = None
    if 'attack' in graph.edata:
        selected_attack_flags = graph.edata['attack'][boolean_mask]
        if selected_attack_flags.numel() > 0:
            attack_edge_ratio = selected_attack_flags.float().mean().item()
        else:
            attack_edge_ratio = 0.0
    else:
            attack_edge_ratio = 0.0

    return fidelity_plus, fidelity_minus, sparsity, attack_edge_ratio


if __name__ == "__main__":
    try:
        p = float(sys.argv[1])
        valid = (0 <= p <= 1)
        if not valid:
            raise ValueError
        
    except (IndexError, ValueError):
        print("Error: First argument must be a number between 0 and 1.")
        sys.exit(1)      

    percentile = 1 - p

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
    logger = logging.getLogger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Match training config
    edge_feat_dim = 46
    label_dim = 7
    hidden_dim = 128
    num_heads = 8
    num_layers = 3
    dropout = 0

    model = Model(
        edge_feat_dim=edge_feat_dim,
        label_dim=label_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device
    ).to(device)

    model = load_latest_checkpoint(model, device)

    test_data = process_data(
        logger,
        '/mnt/d/NF-CICIDS2018-v3/explanation/X.csv',
        '/mnt/d/NF-CICIDS2018-v3/explanation/labels.csv',
        '/mnt/d/NF-CICIDS2018-v3/explanation/idx_label.csv'
    )

    for i, (graph, label) in enumerate(test_data):
        node_labels = graph.cpu().id2addr

        graph = dgl.remove_self_loop(graph)
        visualize_graph_explanation(
            graph.cpu(),
            np.zeros(graph.num_edges()),
            title="",
            output_path=f'./tmp/fig/input_{i}.png',
            #node_labels=node_labels,
            only_explained_subgraph=False)

        logger.info(f'[sample {i}] Edges: {graph.num_edges()}')

        graph = dgl.add_self_loop(graph).to(device)
        graph_rep, logits = model(graph)
        logits_np = logits.detach().cpu().numpy()
        logger.info(f'[sample {i}] Logits: {np.exp(logits_np)/np.sum(np.exp(logits_np))}') # softmax(logits)
        pred = torch.argmax(logits, dim=1)
        logger.info(f'[sample {i}] Label: {label}, prediction: {pred.item()}')

        alphas = graph.edata['alpha']
        edge_mask = compute_edge_mask(alphas, reduce_heads='mean', reduce_layers='max')
        graph, edge_mask = remove_self_loops(graph, edge_mask)
        edge_mask = threshold_edge_mask(edge_mask, percentile=percentile)

        visualize_graph_explanation(
            graph.cpu(),
            edge_mask,
            title="",
            output_path=f'./tmp/fig/explanation_{i}.png',
            #node_labels=node_labels,
            only_explained_subgraph=True)
