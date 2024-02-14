from torch import nn


class QNetwork(nn.Module):
    def __init__(self, states, actions):
        super(QNetwork, self).__init__()
        self.network = nn.Linear(states, actions)

    def forward(self, x):
        return self.network(x)
