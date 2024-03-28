import random
from blackjack_agent import BlackjackAgent

class RandomAgent(BlackjackAgent):
    def __init__(self):
        super().__init__()
    
    def action_selector(self, observation):
        return random.choice(self.action_space)
        
        
