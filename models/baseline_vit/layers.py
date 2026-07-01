import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=48, drop=0.):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.up = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return x + self.up(self.drop(self.act(self.down(x))))


class BottleneckMlp(nn.Module):
    def __init__(self, dim, hidden_dim=48, drop=0.):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.up = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.up(self.drop(self.act(self.down(x))))
