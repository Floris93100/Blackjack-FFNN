from blackjack_agent import BlackjackAgent

class RandomAgent(BlackjackAgent):
    def __init__(self, env, filename=None):
        super().__init__(env, filename)
    
    def action_selector(self, observation):
        return self.action_space.sample()
        
        
