import torch
import torch.nn as nn
from tqdm import tqdm

class FFNN(nn.Module):
    
    def __init__(
        self, 
        layers, 
        bias=True, 
        threshold = 0.5, 
        epochs=60, 
        learning_rate=0.01,
        activation_fn = nn.ReLU(),
        labels=4
    ):
        super().__init__()
        
        self.labels = labels
        self.model = []
        for i in range(1, len(layers)):
            self.model.append(
                FFLayer(
                    layers[i-1], 
                    layers[i], 
                    bias, 
                    threshold, 
                    epochs, 
                    learning_rate, 
                    activation_fn,
                )
            )   
            
    def combine_input_and_label(self, x, y, n):
        y_one_hot = torch.eye(n)[y]
        return torch.concat((x, y_one_hot), 1)       

    def predict_accumulated_goodness(self, u):
        goodness_per_action = []
        for label in range(self.labels):
            x_test = self.combine_input_and_label(u, torch.full((u.size(0),),label), self.labels)
            accumulated_goodness = 0
            for layer in self.model:
                x_test = layer.forward(x_test)
                if layer != self.model[0]:
                    accumulated_goodness += layer.calculate_goodness(x_test)
            goodness_per_action.append(accumulated_goodness)
        predicted_label = torch.argmax(torch.stack(goodness_per_action), dim=0, keepdim=True)
        return predicted_label
        
    def train(self, u_pos, u_neg):
        x_pos, x_neg = u_pos, u_neg
        for layer in self.model:
            print(f'Training Layer: {self.model.index(layer) + 1}')
            x_pos, x_neg = layer.train(x_pos, x_neg)
                

class FFLayer(nn.Linear):
    
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias, 
        threshold, 
        epochs, 
        learning_rate,
        activation_fn,
    ):
        super().__init__(in_features, out_features, bias)
        
        self.activation_fn = activation_fn # vervangen door eigen versie?
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.threshold = threshold
        
    def layer_normalization(self, x):
        # x_i = x_i / (sqrt(sum(x_i^2)))
        return x / (torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))+ 1e-08) #  / N erbij?
    
    def forward(self, x):
        # normalize input vector
        x_norm = self.layer_normalization(x)
        # z = Wx + b
        z = super().forward(x_norm)
        
        # print(torch.mm(x_norm, self.weight.T) +
        #     self.bias.unsqueeze(0)) waarom niet dit? addmmbackward0 vs addbackward0
        
        # a = f(z)
        return self.activation_fn(z)
    
    
    def calculate_goodness(self, x):
        # goodness = sum of squared activations
        # print(x.pow(2).mean(1))
        return torch.sum(x**2, dim=1)

    def train(self, x_pos, x_neg):
        total_loss = 0
        for epoch in tqdm(range(self.epochs)):
            a_pos = self.forward(x_pos)
            a_neg = self.forward(x_neg)
            
            g_pos = self.calculate_goodness(a_pos)
            g_neg = self.calculate_goodness(a_neg)
            
            loss_pos = torch.log(1 + torch.exp(-(g_pos - self.threshold)))
            loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
            loss = (loss_pos + loss_neg).sum() # sum of mean reduction?
            
            self.optimizer.zero_grad()
            # calculate derivative
            loss.backward() 
            self.optimizer.step()
            
            total_loss += loss.item()
        print(f'Avg Loss: {total_loss/self.epochs}')
            
        return self.forward(x_pos).detach(), self.forward(x_neg).detach() # .detach()?
        
        