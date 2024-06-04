from blackjack_agent import BlackjackAgent
from FFNN import FFNN
import torch

class FFNNAgent(BlackjackAgent):
    def __init__(self, env, model_path, device, layers, threshold, lr, filename=None):
        super().__init__(env, filename)
        self.model = FFNN(
            device,
            layers=layers,
            threshold=threshold,
            learning_rate=lr,
        )
        self.model.load_model(model_path)
        self.device = device
        
    def action_selector(self, observation):
        return self.model.predict_accumulated_goodness(torch.tensor([observation]).to(self.device)).item()