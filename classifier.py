import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64, 128], device='cpu'):
        super(Classifier, self).__init__()
        layers = []
        for i, hidden in enumerate(hiddens):
            input_size = in_dim if i == 0 else hiddens[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(hiddens[-1], out_dim)]
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor):
        return self.net(x)