import ast
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch as th
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class TimeGroupDataset(Dataset):
    def __init__(self, data_df, label_df, group_col='time_group', feature_col='features'):
        self.samples = []
        self.labels = []

        label_map = {row['time_group']: row['label'] for _, row in label_df.iterrows()}
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(label_map.values())))}
        self.label_to_idx = label_to_idx

        grouped = data_df.groupby(group_col)
        for group_id, group_df in grouped:
            if group_id in label_map:
                seq = [th.tensor(ast.literal_eval(f), dtype=th.float32) for f in group_df[feature_col]]
                self.samples.append(th.stack(seq))
                self.labels.append(label_to_idx[label_map[group_id]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def collate(batch):
    sequences, labels = zip(*batch)
    lengths = th.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)  # (batch_size, max_len, feature_dim)
    labels = th.tensor(labels)
    return padded, lengths, labels

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths):
        x = x.contiguous()

        lengths_sorted, idx = lengths.sort(descending=True)
        x_sorted = x[idx]

        packed = nn.utils.rnn.pack_padded_sequence(x_sorted.contiguous(), lengths_sorted.cpu(), batch_first=True)
        packed_out, (hn, _) = self.lstm(packed)

        # Take last hidden state
        if self.lstm.bidirectional:
            out = th.cat((hn[-2], hn[-1]), dim=1)
        else:
            out = hn[-1]

        # Reorder back
        _, rev_idx = idx.sort()
        out = out[rev_idx]

        return self.fc(out)

data_folder = '/mnt/d/TON_IoT_Dataset/complete'
max_epochs = 100
patience = 6
binary = True
batch_size = 128
lr = 0.001
hidden_dim = 64
bidirectional = False

if __name__ == "__main__":
    print(f'CUDA available: {th.cuda.is_available()}')
    device = 'cuda'
    th.device(device)

    # cuDNN encountered issues due to variable-length sequences
    # fallback to more stable PyTorch implementation
    th.backends.cudnn.enabled = False 

    set_seed(42)

    train_df = pd.read_csv(f'{data_folder}/X_train.csv')
    test_df = pd.read_csv(f'{data_folder}/X_test.csv')
    label_df = pd.read_csv(f'{data_folder}/labels.csv')
    idx_label_df = pd.read_csv(f'{data_folder}/idx_label.csv')

    if binary:
        label_df['label'] = label_df['label'].where(label_df['label'] == 'normal', 'attack')
        idx_label_df = pd.DataFrame({ 'label': [ 'attack', 'normal' ] })

    print('Data is loaded.')

    train_data = train_df
    train_labels = label_df[label_df['type'] == 'train']

    train_dataset = TimeGroupDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)

    model = LSTMClassifier(
        input_dim=len(ast.literal_eval(train_data['features'].iloc[0])),
        hidden_dim=hidden_dim,
        num_classes=len(idx_label_df['label'].unique()),
        bidirectional=bidirectional
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    
    print('LSTM - Start training.')

    model.train()

    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        epoch_losses = []

        print(f"\n LSTM - Epoch {epoch+1}/{max_epochs}")
        for i, (inputs, lengths, labels) in enumerate(train_loader):
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f"LSTM - Batch {i+1:02d}: Loss = {loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"LSTM - Average loss for epoch {epoch+1}: {avg_loss:.4f}")

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

    print('LSTM - Training is finished, start evaluation.')

    model.eval()

    test_data = test_df
    test_labels = label_df[label_df['type'] == 'test']

    test_dataset = TimeGroupDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate, shuffle=False)

    all_preds = []
    all_targets = []

    with th.no_grad():
        for inputs, lengths, labels in test_loader:
            inputs, lengths = inputs.to(device), lengths.to(device)
            outputs = model(inputs, lengths)  # shape: (1, num_classes)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, digits=4, target_names=list(test_dataset.label_to_idx.keys())))

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(test_dataset.label_to_idx))))
    print("Confusion matrix:")
    print(cm)
