import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import EdgeGATConv

class Encoder(nn.Module):
    def __init__(self, edge_feat_dim, hidden_dim, output_dim, num_layers=3, num_heads=8, dropout=0.0):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.input_dim = edge_feat_dim

        # First EdgeGATConv layer
        self.layers.append(
            EdgeGATConv(
                in_feats=self.input_dim,
                out_feats=hidden_dim // num_heads,
                edge_feats=edge_feat_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
            )
        )

        # Hidden EdgeGATConv layers
        for _ in range(num_layers - 1):
            self.layers.append(
                EdgeGATConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim // num_heads,
                    edge_feats=edge_feat_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                )
            )

        # Final linear projection to desired output dimension
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, g):
        g = dgl.add_self_loop(g)
        h = g.ndata['feat']
        edge_feats = g.edata['feat']

        for layer in self.layers:
            h = layer(g, h, edge_feats)
            h = h.flatten(1)  # concat across heads

        out = self.project(h)
        return out
