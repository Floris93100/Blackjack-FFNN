import torch.nn as nn
import torch

class BackpropNN(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, activation_fn):
        super().__init__()
        self.model = nn.Sequential(
            # sequential network with 1 input layer, 4 hidden layers, 1 output layer
            nn.Linear(input_size, hidden_size, bias=True),
            activation_fn,
            nn.Linear(hidden_size, hidden_size, bias=True),
            activation_fn,
            nn.Linear(hidden_size, hidden_size, bias=True),
            activation_fn,
            nn.Linear(hidden_size, hidden_size, bias=True),
            activation_fn,
            nn.Linear(hidden_size, output_size, bias=True)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def predict_action(self, x):
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(x)
            return predictions.argmax().item()