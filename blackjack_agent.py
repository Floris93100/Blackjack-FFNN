import random

class BlackjackAgent():
    # Blackjack agent class, agents can implement action_selector depending on the strategy
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
    
    def get_action(self, observation):
        action = self.action_selector(observation)
        
        double_down_allowed = observation[3]
        split_allowed = not observation[4] and observation[5]
            
        # Choose different action while double down not allowed
        while (not double_down_allowed and action == 2) or (not split_allowed and action == 3):
            action = self.action_selector(observation)

        return action
        
    def action_selector(self, observation):
        return random.choice(self.action_space)
    
    def learn(self, state, action, reward, next_state):
        pass