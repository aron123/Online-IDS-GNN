import torch.nn as nn

from core.classifier import Classifier
from core.encoder import Encoder

class Model(nn.Module):
    def __init__(self, edge_feat_dim, label_dim, num_heads, num_layers, hidden_dim, dropout, device='cpu'):
        super(Model, self).__init__()
        self.encoder = Encoder(
            edge_feat_dim=edge_feat_dim,
            graph_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        self.classifier = Classifier(
            in_dim=hidden_dim,
            out_dim=label_dim,
            hiddens=[128, 64],
            device=device
        ).to(device)

    def forward(self, graph):
        rep = self.encoder(graph)
        return rep, self.classifier(rep)
