import dgl
import torch
from torch import nn
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self, logger, device='cpu'):
        self.logger = logger
        self.device = device

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def train(self, train_data, batch_size, max_epochs):
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate)

        for epoch in range(max_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            for batch_graphs, batch_labels in train_dl:
                # TODO: implement the training logic
                continue


        self.logger.error("NotImplementedError: train method is not implemented yet.")
    
    def test(self, test_data, model=None):
        self.logger.error("NotImplementedError: test method is not implemented yet.")

