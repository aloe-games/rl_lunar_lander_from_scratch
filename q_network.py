from torch import nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.network(x)