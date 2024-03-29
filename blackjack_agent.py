import random

class BlackjackAgent():
    # Blackjack agent class, agents can implement action_selector depending on the strategy
    def __init__(self, env):
        self.action_space = env.action_space
    
    def get_action(self, observation):
        
        #no action required if natural blackjack
        if observation[0] == 21:
            return 0
        
        action = self.action_selector(observation)
        
        double_down_allowed = observation[3]
        split_allowed = not observation[4] and observation[5]
            
        # Choose different action while double down not allowed
        while (not double_down_allowed and action == 2) or (not split_allowed and action == 3):
            action = self.action_selector(observation)

        return action
        
    def action_selector(self, observation):
        return self.action_space.sample()
    
    def learn(self, state, action, reward, next_state):
        pass