import torch
import torch.nn as nn
from dgl.nn.pytorch import EdgeGATConv, MaxPooling

class Encoder(nn.Module):
    def __init__(self, edge_feat_dim, graph_dim, num_layers=3, num_heads=8, dropout=0.0):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.num_heads = num_heads
        self.input_dim = edge_feat_dim

        # First EdgeGATConv layer
        self.layers.append(
            EdgeGATConv(
                in_feats=self.input_dim,
                out_feats=graph_dim // num_heads,
                edge_feats=edge_feat_dim,
                num_heads=num_heads,
                attn_drop=dropout,
            )
        )

        # Hidden EdgeGATConv layers
        for _ in range(num_layers - 1):
            self.layers.append(
                EdgeGATConv(
                    in_feats=graph_dim,
                    out_feats=graph_dim // num_heads,
                    edge_feats=edge_feat_dim,
                    num_heads=num_heads,
                    attn_drop=dropout,
                )
            )

        self.pool = MaxPooling()

    def forward(self, g):
        node_feats = g.ndata['feat']
        edge_feats = g.edata['feat']
        
        h = node_feats
        
        all_alphas = []

        for layer in self.layers:
            h, attn_weights = layer(g, h, edge_feats, get_attention=True)
            if not self.training:
                all_alphas.append(attn_weights)  # [num_edges, num_heads]
            h = h.flatten(1)  # [num_nodes, hidden_dim]

        g.ndata['h'] = h

        if not self.training:
            stacked = torch.stack(all_alphas, dim=1)  # [num_edges, num_layers, num_heads]
            g.edata['alpha'] = stacked

        hg = self.pool(g, h)
        return hg
