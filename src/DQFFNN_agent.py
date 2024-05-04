from blackjack_agent import BlackjackAgent
from FFNN import FFNN
import torch
import numpy as np
import random

class DQFFNNAgent(BlackjackAgent):
    def __init__(
        self, 
        env,
        device, 
        layers, 
        threshold, 
        lr,
        epsilon,
        epsilon_decay,
        discount_factor,
        batch_size=1,
        update_td_target=1000,
        filename=None
        ):
        super().__init__(env, filename)
        
        self.model = FFNN(
            device,
            layers=layers,
            threshold=threshold,
            learning_rate=lr,
            epochs=1,
            batch_size=batch_size
        )
        self.td_target = FFNN(
            device,
            layers=layers,
            threshold=threshold,
            learning_rate=lr,
            epochs=1,
            batch_size=batch_size
        )
        self.td_target.load_model(from_file=False, state=self.model.save_model(save=False))
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.train_iteration = 0
        self.update_td_target = update_td_target
        self.batch_size = batch_size
        
        #experience replay buffer
        self.D = []
    
    def generate_neg_labels(y, n):
        y_ = y.copy()
        for i in range(len(y)):
            negative_labels = [j for j in range(n) if j != y[i]]
            y_[i] = np.random.choice(negative_labels)
        return y_
    
    def obs_to_tensor(self, observation):
        return torch.tensor(observation, dtype=torch.float32)
    
    def update(self): 
        
        # sample minibatch from experience replay
        if len(self.D) < self.batch_size:
            return
        
        #batch = random.choice(self.D) # bigger batch?
        batch = random.sample(self.D, k=self.batch_size)
        
        #state, action, reward, next_state, done = batch
        
        states, actions, rewards, next_states, done = zip(*batch)
            
        states = self.obs_to_tensor(states)
        next_states = self.obs_to_tensor(next_states)
        rewards = self.obs_to_tensor(rewards)
        
        print(f'states: {states}, actions: {actions}, rewards: {rewards}, next_states: {next_states}, done: {done}')
        
        with torch.no_grad():
            # compute td target
            td_values = self.td_target.predict_accumulated_goodness(next_states, return_goodness=True)
            max_q = torch.max(td_values, dim=1).values
            max_q[done] = 0
            td_target = rewards + self.discount_factor * max_q
            print(f'td_target: {td_target} = {rewards} + {self.discount_factor} * {max_q}')
            
            # compute model output
            predicted_q = self.model.predict_accumulated_goodness(states, return_goodness=True)[actions].item()
        
        
        # compute td error
        td_error = td_target - predicted_q
        # positive sample is state with action
        #state = self.obs_to_tensor(state)    
        x_pos = self.model.combine_input_and_label(states, torch.full((states.size(0),),actions), self.model.labels)

        # td target - goodness of action in current model?
        if td_error >= 0:
            # positive action, negative input is state with random action
            y_neg = self.generate_neg_labels(actions, self.model.labels)
            # random_action = self.action_space.sample() 
            x_neg = self.model.combine_input_and_label(states, torch.full((states.size(0),), y_neg), self.model.labels)
        else:
            # negative action, negative input is equal to positive pass
            x_neg = x_pos.clone()
        
        print(f'x_pos: {x_pos}')
        print(f'x_neg: {x_neg}')
        
        # update model 
        self.model.train(x_pos, x_neg)
        
        self.train_iteration += 1
        
        # optionally update td target
        if self.train_iteration % self.update_td_target == 0:
            self.td_target.load_model(from_file=False, state=self.model.save_model(save=False))
            
        # if self.train_iteration % 100 == 0:
        #     print("train iteration: ", self.train_iteration, f'state: {state}, action: {action}, td_values: {td_values}')
    
        return
    
    def learn(self, state, action, reward, next_state, done):
        
        # add experience to replay buffer
        self.D.append((state, action, reward, next_state, done))
        
        self.update()
        
        return
    
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)
        #print(f'epsilon: {self.epsilon}')
    
    def action_selector(self, observation):
        # epsilon greedy action selection
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.model.predict_accumulated_goodness(self.obs_to_tensor(observation)).item()
            