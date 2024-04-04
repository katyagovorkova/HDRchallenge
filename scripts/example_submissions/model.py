import torch
import torch.nn as nn
import torch.nn.functional as F


class ADClassifier(nn.Module):

    def __init__(self):
        super(ADClassifier, self).__init__()

        self.n_head = 1
        self.flat1 = nn.Flatten()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=400, nhead=self.n_head)
        self.fc1 = nn.Linear(400*self.n_head, 400)
        self.flat2 = nn.Flatten()
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x):

        x = self.flat1(x)
        x = self.encoder_layer(x)
        x = F.relu(self.fc1(x))
        x = self.flat2(x)
        x = F.sigmoid(self.fc2(x))

        return x
