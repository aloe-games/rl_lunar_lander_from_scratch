from torch import nn


class QNetwork(nn.Module):
    def __init__(self, states, actions):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(states, 64),
            nn.Linear(64, 64),
            nn.Linear(64, actions),
        )

    def forward(self, x):
        return self.network(x)