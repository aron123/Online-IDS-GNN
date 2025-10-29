# Derived from: https://github.com/waimorris/E-GraphSAGE

from pathlib import Path
import random

import dgl
import dgl.function as fn

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['feat'], edges.data['feat']], 1))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['feat'] = nfeats
            g.edata['feat'] = efeats
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            g.ndata['feat'] = self.activation(self.W_apply(th.cat([g.ndata['feat'], g.ndata['h_neigh']], 1)))
            return g.ndata['feat']

class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_classes)
        )

    def forward(self, graph_reps):
        return self.classifier(graph_reps)

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout, num_classes):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, num_classes)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        g.ndata['feat'] = h
        graph_reps = dgl.readout_nodes(g, 'feat', op='max')
        return self.pred(graph_reps)

def set_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    graphs = [dgl.add_self_loop(g) for g in graphs]
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    return batched_graph.to('cuda'), batched_labels.to('cuda')

out_dim = 128
num_classes = 2
binary = True
dropout = 0.0
batch_size = 128
max_epochs = 100
patience = 5
data_dir = '/mnt/d/NF-CICIDS2018-v3/complete'

if __name__ == "__main__":
    print(f'CUDA available: {th.cuda.is_available()}')
    th.device("cuda")

    set_seed(42)

    train_data = th.load('./tmp/train.pt', weights_only=False)
    test_data = th.load('./tmp/test.pt', weights_only=False)
    print("Data is loaded successfully.")

    edge_dim = train_data[0][0].edata['feat'].shape[1]
    node_dim = train_data[0][0].ndata['feat'].shape[1]

    model = Model(node_dim, out_dim, edge_dim, F.relu, dropout, num_classes).cuda()

    class_labels = [label for _, label in train_data]
    classes = np.unique(class_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=class_labels)
    weights = th.tensor(weights, dtype=th.float32).to('cuda')
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = th.optim.Adam(model.parameters())

    model.train()
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        print(f"E-GraphSAGE - Epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0.0

        for batch_graph, labels in train_dl:
            optimizer.zero_grad()
            pred_logits = model(batch_graph, batch_graph.ndata['feat'], batch_graph.edata['feat'])
            loss = criterion(pred_logits, labels)
            loss.backward()
            optimizer.step()

            print(f"E-GraphSAGE - batch_loss: {loss.item():.4f}")
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dl)
        print(f"E-GraphSAGE - Epoch [{epoch + 1}/{max_epochs}], avg_loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stop training.")
            break

    print("E-GraphSAGE - Training completed, start testing.")

    model.eval()
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

    all_preds, all_labels = [], []

    for graph, labels in test_dl:
        logits = model(graph, graph.ndata['feat'], graph.edata['feat'])
        preds = th.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    labels = pd.read_csv(Path(data_dir) / 'idx_label.csv')
    targets = labels['label'].tolist() if binary is False else [ 'attack', 'normal' ]

    cr = classification_report(all_labels, all_preds, digits=4, target_names=targets)
    print("Classification Report:")
    print(cr)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(targets))))
    print("Confusion matrix:")
    print(cm)
