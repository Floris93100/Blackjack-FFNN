import torch
import torch.nn as nn
from tqdm import tqdm

from linear_classifier import LinearClassifier

class FFNN(nn.Module):
    
    def __init__(
        self, 
        device,
        layers, 
        bias=True, 
        threshold = 0.5, 
        learning_rate=0.01,
        epochs=60, 
        batch_size=32,
        activation_fn = nn.ReLU(),
        lr_decay=False,
        labels=4,
        classifier=False 
    ):
        super().__init__()
        
        self.device = device
        self.labels = labels
        self.batch_size = batch_size   
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
                    lr_decay,
                    device
                )
            )   
        self.to(self.device)
        
        if classifier:
            self.classifier = LinearClassifier(sum(layers[2:]),labels, 0.0001)
            
            
    def combine_input_and_label(self, x, y, n):
        y_one_hot = torch.eye(n)[y]
        return torch.concat((x, y_one_hot), 1)       

    def predict_accumulated_goodness(self, u, return_goodness=False):
        goodness_per_action = []
        for label in range(self.labels):
            x_test = self.combine_input_and_label(u, torch.full((u.size(0),),label), self.labels)
            accumulated_goodness = 0
            for layer in self.model:
                x_test = layer.forward(x_test.to(self.device))
                if layer != self.model[0]:
                    accumulated_goodness += layer.calculate_goodness(x_test)
            goodness_per_action.append(accumulated_goodness)
        predicted_label = torch.argmax(torch.stack(goodness_per_action), dim=0, keepdim=True)
        
        if return_goodness:
            return goodness_per_action
        
        return predicted_label
    
    def predict_classifier(self, u):
        # Combine input with neutral labels
        x = torch.concat((u, torch.full((u.size(0),self.labels), (1/self.labels))), dim=1)
        input = self.collect_hidden_layer_activations(x)

        self.classifier.eval()
        
        with torch.no_grad():
            predictions = self.classifier(input)
            return torch.argmax(predictions, dim=1)

        
    def train(self, u_pos, u_neg):
        x_pos, x_neg = u_pos.to(self.device), u_neg.to(self.device)
        for layer in self.model:
            #print(f'\nTraining Layer: {self.model.index(layer) + 1}')
            #print("-"*40)
            x_pos, x_neg, losses = layer.train(x_pos.to(self.device), x_neg.to(self.device), self.batch_size)
        
        return losses
          
    def save_model(self, path='../models/model.pth', save=True):
        state = {
            'model_state': [layer.state_dict() for layer in self.model],
            'optimizer_state': [layer.optimizer.state_dict() for layer in self.model],
        }
        if save:    
            torch.save(state, path)
        else:
            return state
        
    def load_model(self, path='../models/model.pth', from_file=True, state=None):
        if from_file:
            state = torch.load(path)
        else:
            state = state
        for layer, layer_state, optimizer_state in zip(self.model, state['model_state'], state['optimizer_state']):
            layer.load_state_dict(layer_state)
            layer.optimizer.load_state_dict(optimizer_state)

    def collect_hidden_layer_activations(self, u):
        input = torch.empty(0).to(self.device)
        for layer in self.model:
            u = layer.forward(u)
            # Collect normalized activities of all hidden layers except first
            if layer != self.model[0]:
                x_norm = layer.layer_normalization(u)
                input = torch.cat((input, x_norm), 1)
        
        return input.clone().detach()
            
    def train_classifier(self, x, y, epochs):
        # Forward pass FFNN
        input = self.collect_hidden_layer_activations(x)
        
        # Train Softmax classifier
        self.classifier.train()
        
        print("\nTraining Softmax")
        print("-"*40)
        for epoch in tqdm(range(epochs)):
            predictions = self.classifier(input)
            loss = self.classifier.loss(predictions, y)
            
            self.classifier.optimizer.zero_grad()
            loss.backward()
            self.classifier.optimizer.step()
                      
        print(f"Last epoch loss: {loss.item()}")
            
        return

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
        lr_decay,
        device
    ):
        super().__init__(in_features, out_features, bias)
        
        self.activation_fn = activation_fn # vervangen door eigen versie?
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.threshold = threshold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.to(device)
        
    def get_learning_rate(self, current_epoch):
        # lr(e) = 2LR/E * (1 + E - e)
        if self.lr_decay:
            if current_epoch > self.epochs/2:
                lr = (2 * self.learning_rate) / self.epochs * (1 + self.epochs - current_epoch)
            else:
                lr = self.learning_rate
            return lr
        else:
            return self.learning_rate
        
    def layer_normalization(self, x):
        # x_i = x_i / (sqrt(sum(x_i^2)))
        return x / (torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))+ 1e-08) #  / N erbij?
    
    def forward(self, x):
        # normalize input vector
        x_norm = self.layer_normalization(x)
        # z = Wx + b
        z = super().forward(x_norm)
        # a = f(z)
        return self.activation_fn(z)
    
    
    def calculate_goodness(self, x):
        # goodness = sum of squared activations
        return torch.sum(x**2, dim=1)

    def train(self, x_pos, x_neg, batch_size):
        losses = []
        num_batches = len(x_pos) // batch_size
        
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            
            if self.lr_decay:
                lr = self.get_learning_rate(epoch + 1)
                self.optimizer.param_groups[0]['lr'] = lr

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                x_pos_batch = x_pos[start_idx:end_idx]
                x_neg_batch = x_neg[start_idx:end_idx]
                
                a_pos = self.forward(x_pos_batch)
                a_neg = self.forward(x_neg_batch)
                
                g_pos = self.calculate_goodness(a_pos)
                g_neg = self.calculate_goodness(a_neg)
                
                loss_pos = torch.log(1 + torch.exp(-(g_pos - self.threshold)))
                loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
                loss = (loss_pos + loss_neg).sum()
                epoch_loss += loss.item()
                
                self.optimizer.zero_grad()
                # calculate derivative
                loss.backward() 
                self.optimizer.step()
                
            losses.append(epoch_loss / x_pos.size(0))    
            #print(f'epoch:{epoch+1}, avg loss: {epoch_loss / x_pos.size(0)}, lr: {self.get_learning_rate(epoch + 1)}')
            
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), losses 
        
        