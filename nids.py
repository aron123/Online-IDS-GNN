import glob
import os
import pickle
import time
import dgl
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from core.model import Model

class NIDS(object):
    def __init__(self, logger, edge_feat_dim, label_dim, num_heads, num_layers, hidden_dim, dropout, device='cpu'):
        super(NIDS, self).__init__()
        self.logger = logger
        self.device = device
        self.model = Model(
            edge_feat_dim=edge_feat_dim,
            label_dim=label_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device
        ).to(device)

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        graphs = [dgl.add_self_loop(g) for g in graphs] # enable message passing for 0-indegree nodes
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph.to(self.device), batched_labels.to(self.device)
    
    def remove_previous_models(self):
        pt_files = glob.glob('./tmp/model/*.pt')
        for file_path in pt_files:
            os.remove(file_path)

    def load_last_model(self):
        matching_files = sorted(glob.glob(f'./tmp/model/*.pt'))
        
        if not matching_files:
            self.logger.warning("No model checkpoint found.")
            return None

        filename_to_load = matching_files[-1]
        self.logger.info(f'Loading model from "{filename_to_load}" ...')
        checkpoint = torch.load(filename_to_load, map_location=self.device)
        return checkpoint

    def train(self, train_data, batch_size, max_epochs, patience, lr, weight_decay):
        self.remove_previous_models()

        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate)

        self.logger.info(self.model)
        self.model.train()

        # class_labels = [label for _, label in train_data]
        # classes = np.unique(class_labels)
        # weights = compute_class_weight(class_weight='balanced', classes=classes, y=class_labels)
        # weights = torch.tensor(weights, dtype=torch.float32).to('cuda')
        # criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            epoch_loss = 0.0

            for batch_graph, labels in train_dl:
                optimizer.zero_grad()
                _, pred_logits = self.model(batch_graph)

                loss = criterion(pred_logits, labels)
                loss.backward()
                optimizer.step()

                self.logger.info(f"batch_loss: {loss.item():.4f}")

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_dl)
            self.logger.info(f"Epoch [{epoch + 1}/{max_epochs}], avg_loss: {avg_loss:.4f}")
            
            # Save model after every epoch
            torch.save(self.model.state_dict(), f"./tmp/model/model_epoch_{str(epoch+1).zfill(len(str(max_epochs)))}.pt")

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                self.logger.debug(f"No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= patience:
                self.logger.info(f"No improvement for {patience} epochs, stop training.")
                break

    def test(self, test_data, target_names=[], model=None, display_cm=False):
        if model is None:
            checkpoint = self.load_last_model()
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)

        if self.model is None:
            self.logger.error("No model found to test.")
            return

        self.model.eval()

        test_dl = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=self.collate)

        all_preds, all_labels = [], []
        graph_reps = []

        total_time = 0

        for graph, labels in test_dl:
            start_time = time.time()
            
            graph_rep, logits = self.model(graph)  # (1, num_classes)
            preds = torch.argmax(logits, dim=1)

            end_time = time.time()
            total_time += end_time - start_time

            graph_reps.append(graph_rep.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_time_per_graph = total_time / len(test_data)
        self.logger.info(f"Average inference time: {avg_time_per_graph:.6f} secs (total: {total_time:.6f} secs).")

        with open('./tmp/graph_reps.pkl', 'wb') as f:
            pickle.dump({'graphs': graph_reps, 'labels': all_labels}, f)

        # Classification report
        report = classification_report(
            all_labels, all_preds,
            digits=4,
            target_names=target_names,
            labels=list(range(len(target_names)))
        )
        self.logger.info("Classification Report:\n" + report)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(target_names))))
        self.logger.info("Confusion matrix:\n")
        self.logger.info(cm)

        if display_cm:
            norm_cm = confusion_matrix(
                all_labels, all_preds,
                labels=list(range(len(target_names))),
                normalize='true'
            )
            disp = ConfusionMatrixDisplay(confusion_matrix=norm_cm, display_labels=target_names)
            fig, ax = plt.subplots()
            disp.plot(ax=ax, cmap='Blues')
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            ax.set_title("Confusion Matrix", fontsize=14)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            plt.show()