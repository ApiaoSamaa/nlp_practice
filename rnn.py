import torch
from torch import nn
import torch.nn.functional as F

class RNNBlock(nn.Module):
    def __init__(self, reverse=False, in_dim=200, h_dim=100, out_dim=150):
        self.w1 = nn.Linear(in_dim, h_dim)
        self.w2 = nn.Linear(h_dim, h_dim)
        self.w3 = nn.Linear(h_dim, out_dim)
        self.reverse = reverse

    # (seq, 100) 
    def forward(self, x, h = torch.zeros((100, ))):
        y = []
        idx = range(len(x))
        if self.reverse:
            idx = reversed(idx)
        for i in idx:
            h = F.relu(self.w1(x[i]) + self.w2(h))
            y.append(self.w3(h))
        # y: (seq, 100)
        # h: (100, )
        # a: (1, 100)
        return y, h


class BiDirectionRNN(nn.Module):
    def __init__(self, in_dim=200, h_dim=100, out_dim=150):
        self.rnn1 = RNNBlock(False, in_dim, h_dim, out_dim)
        self.rnn2 = RNNBlock(True, in_dim, h_dim, out_dim)

    def forward(self, x, h0 = torch.zeros((100, ))):
        y1, h1 = self.rnn1(x, h0)
        y2, h2 = self.rnn2(x, h0)
        return [a + b for a, b in zip(y1, y2)]


nn.Sequential(
    BiDirectionRNN(),
    BiDirectionRNN(),
    BiDirectionRNN(),
    BiDirectionRNN(),
    BiDirectionRNN(),
)

nn.RNN()