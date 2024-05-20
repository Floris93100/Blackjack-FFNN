from FFNN import FFNN
import torch
import numpy as np
import random
from collections import deque

class DQFFNNAgent():
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
        buffer_size=100000,
        ):
        
        self.model = FFNN(
            device,
            layers=layers,
            threshold=threshold,
            learning_rate=lr,
            epochs=1,
            batch_size=batch_size,
            labels=env.action_space.n
        )
        self.td_target = FFNN(
            device,
            layers=layers,
            threshold=threshold,
            learning_rate=lr,
            epochs=1,
            batch_size=batch_size,
            labels=env.action_space.n
        )
        self.td_target.load_model(from_file=False, state=self.model.save_model(save=False))
        
        self.env = env
        self.threshold = threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.train_iteration = 0
        self.update_td_target = update_td_target
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        #experience replay buffer
        self.D = []
    
    def generate_neg_label(self, y, n):
        negative_labels = [i for i in range(n) if i != y]
        return np.random.choice(negative_labels)
    
    def obs_to_tensor(self, observation):
        return torch.tensor(np.array(observation), dtype=torch.float32).to(self.model.device)
    
    def update(self): 
        
        if len(self.D) < self.batch_size:
            return
        
        # sample minibatch from experience replay
        batch = random.sample(self.D, k=self.batch_size)
        
        states, actions, rewards, next_states, done = zip(*batch)
  
        next_states = self.obs_to_tensor(next_states)
        rewards = self.obs_to_tensor(rewards)
                
        with torch.no_grad():
            # compute td target
            td_values = self.td_target.predict_accumulated_goodness(next_states, return_goodness=True)
            max_q = torch.max(torch.stack(td_values), dim=0).values # niet max maar huidige actie goodness?
            max_q[done] = 0
            td_target = rewards + self.discount_factor * max_q

            # compute model output
            predicted_q = self.model.predict_accumulated_goodness(self.obs_to_tensor(states), return_goodness=True)
            predicted_q = torch.transpose(torch.stack(predicted_q), 0, 1)[range(len(actions)), actions]
        
        # compute td error
        td_error = td_target - predicted_q
        
        # positive sample is state with action
        x_pos = self.model.combine_input_and_label(
            self.obs_to_tensor(states), 
            torch.tensor(actions, dtype=torch.int64), 
            self.model.labels
        )
        x_neg = torch.zeros_like(x_pos)
        
        for i in range(len(td_target)):
            if td_error[i] >= 0:
                # positive action, negative input is state with random action
                y_neg = self.generate_neg_label(actions[i], self.model.labels)                
                state = self.obs_to_tensor([states[i]])
                x_neg[i] = self.model.combine_input_and_label(state, torch.full((state.size(0),), y_neg), self.model.labels)
            else:
                # negative action, negative input is equal to positive pass
                x_neg[i] = x_pos[i].clone()
        
        # update model 
        self.model.train(x_pos, x_neg)
        
        self.train_iteration += 1
        
        # optionally set td target parameters to model parameters
        if self.train_iteration % self.update_td_target == 0:
            self.td_target.load_model(from_file=False, state=self.model.save_model(save=False))
            
        return
    
    def learn(self, state, action, reward, next_state, done):
        
        if len(self.D) > self.buffer_size:
            deque(self.D).popleft()
        
        # add experience to replay buffer
        self.D.append((state, action, reward, next_state, done))
        
        self.update()
        
        return
    
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)
        #print(f'epsilon: {self.epsilon}')
    
    def get_action(self, observation):
        # epsilon greedy action selection
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.model.predict_accumulated_goodness(self.obs_to_tensor([observation])).item()
            