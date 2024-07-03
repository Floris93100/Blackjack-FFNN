import csv

class BlackjackAgent():
    # Blackjack agent class, agents can implement action_selector depending on the strategy
    def __init__(self, env, filename=None):
        self.action_space = env.action_space
        self.filename = f"../data/{filename}.csv"
        
        self.data = []
    
    def get_action(self, observation):
        
        # No action required if natural blackjack
        if observation[0] == 21:
            return 0
        
        action = self.action_selector(observation)
        
        double_down_allowed = observation[3]
        split_allowed = observation[4]
            
        # Hit if double down or split not allowed
        while (not double_down_allowed and action == 2) or (not split_allowed and action == 3):
            action = 1

        return action
        
    def action_selector(self, observation):
        return self.action_space.sample()
    
    def collect_data(self, observation, action):
        self.data.append((observation[0], observation[1], observation[2], observation[3], observation[4], action))
        
    def save_data(self):
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(["Player sum", "Dealer card", "Usable ace", "Double down allowed", "Split allowed", "Action"])
            writer.writerows(self.data)
    
    def learn(self, state, action, reward, next_state, done):
        pass
    
    def decay_epsilon(self):
        pass