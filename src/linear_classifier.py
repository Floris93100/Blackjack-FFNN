import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes, lr):
        super(LinearClassifier, self).__init__()
        
        self.linear = nn.Linear(n_features, n_classes)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        return self.linear(x)