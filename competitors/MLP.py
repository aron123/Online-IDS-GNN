from pathlib import Path
import random

import dgl
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, dropout=0.0):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def stats_pooling(edge_feats):
    mean = edge_feats.mean(dim=0)
    std = edge_feats.std(dim=0)
    min_ = edge_feats.min(dim=0).values
    max_ = edge_feats.max(dim=0).values
    return th.cat([mean, std, min_, max_], dim=0)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    
    pooled_vectors = []
    for g in graphs:
        g = dgl.add_self_loop(g)
        efeats = g.edata['feat']
        pooled = stats_pooling(efeats)
        pooled_vectors.append(pooled)
    
    pooled_tensor = th.stack(pooled_vectors)  # (num_inputs, pooled_feat_dim)
    labels_tensor = th.tensor(labels)

    return pooled_tensor.to('cuda'), labels_tensor.to('cuda')

num_classes = 2
binary = True
dropout = 0.0
batch_size = 128
max_epochs = 100
patience = 5
data_dir = '/mnt/d/NF-CICIDS2018-v3/complete'

if __name__ == "__main__":
    print(f'CUDA available: {th.cuda.is_available()}')
    device = 'cuda'
    th.device(device)
    set_seed(42)

    train_data = th.load('./tmp/train.pt', weights_only=False)
    test_data = th.load('./tmp/test.pt', weights_only=False)
    print("Data is loaded successfully.")

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

    model = MLPClassifier(input_dim=train_data[0][0].edata['feat'].shape[1] * 4, hidden=[256, 128], output_dim=num_classes, dropout=dropout).to(device)

    class_labels = [label for _, label in train_data]
    classes = np.unique(class_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=class_labels)
    weights = th.tensor(weights, dtype=th.float32).to('cuda')
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = th.optim.Adam(model.parameters())

    model.train()

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        print(f"MLP - Epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0.0

        for batch_input, labels in train_dl:
            optimizer.zero_grad()
            pred_logits = model(batch_input)
            loss = criterion(pred_logits, labels)
            loss.backward()
            optimizer.step()

            print(f"MLP - batch_loss: {loss.item():.4f}")
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dl)
        print(f"MLP - Epoch [{epoch + 1}/{max_epochs}], avg_loss: {avg_loss:.4f}")

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

    print("MLP - Training completed, start testing.")

    model.eval()

    all_preds, all_labels = [], []

    for input, labels in test_dl:
        logits = model(input)
        preds = th.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    labels = pd.read_csv(Path(data_dir) / 'idx_label.csv')
    targets = labels['label'].tolist() if binary is False else ['attack', 'normal']

    cr = classification_report(all_labels, all_preds, digits=4, target_names=targets)
    print("Classification Report:")
    print(cr)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(targets))))
    print("Confusion matrix:")
    print(cm)
