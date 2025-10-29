from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self):
        self.data = []

    def add_item(self, graph, label):
        self.data.append({
            'graph': graph,
            'label': label
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph = self.data[idx]['graph']
        label = self.data[idx]['label']
        return graph, label