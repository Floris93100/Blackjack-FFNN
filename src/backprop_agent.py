from blackjack_agent import BlackjackAgent
from backpropNN import BackpropNN
import torch

class BackpropAgent(BlackjackAgent):
    def __init__(self, env, model, input_size, output_size, hidden_size, activation_fn,  filename=None):
        super().__init__(env, filename)
        self.model = BackpropNN(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            activation_fn=activation_fn
        )
        self.model.load_state_dict(model)
        
    def action_selector(self, observation):
        return self.model.predict_action(torch.Tensor(observation))
        
            
            