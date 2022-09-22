import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random

from collections import namedtuple, deque


np.random.seed(1)

class DeepQNetwork(nn.Module):
    
    def __init__(self, n_actions, n_features):
        
        super(DeepQNetwork, self).__init__()
        self.l1 = nn.Linear(n_features, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, n_actions)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class DeepQLearning():
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            discount_factor=0.9,
            e_greedy=0.05,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32
    ):
    
    # Initialzie variables here 
        self.discount_factor = discount_factor
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        
        
        
        self.model = DeepQNetwork(n_actions, n_features)
        self.model_target = DeepQNetwork(n_actions, n_features)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.replaybuffer = namedtuple('replay', field_names=['s', 'a', 'r', 'next_s'])
        self.memory = deque(maxlen= memory_size) ##fifo

    def store_transition(self, s, a, r, next_s):
        t = self.replaybuffer(s, a, r, next_s)
        self.memory.append(t)

    def choose_action(self, state):        
        ##epsilon-greedy
        p = np.random.random()
        if p < self.e_greedy: ##return action randomly
            return np.random.randint(low = 0, high = 4)
        else: ##return action with highest probability
            s = torch.from_numpy(state).float().unsqueeze(0) 
            with torch.no_grad():
                a = self.model(s)
            return np.argmax(a.detach().cpu().numpy())

    def learn(self):
        ##sample a minibatch from replay buffer and learn
        mini_batch = random.sample(self.memory, k = self.batch_size)
        
        states = torch.from_numpy(np.vstack([t.s for t in mini_batch if t is not None])).float()
        actions = torch.from_numpy(np.vstack([t.a for t in mini_batch if t is not None])).long()
        rewards = torch.from_numpy(np.vstack([t.r for t in mini_batch if t is not None])).float()
        next_states = torch.from_numpy(np.vstack([t.next_s for t in mini_batch if t is not None])).float()
        
        self.model.train()
        self.model_target.eval()
        q = self.model(states).gather(1, actions)
        with torch.no_grad():
            maxq = self.model_target(next_states).max(1)[0].detach().unsqueeze(1)
        loss = self.loss(rewards + self.discount_factor * maxq, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        ##update target param
        for target_param, local_param in zip(self.model_target.parameters(),
                                           self.model.parameters()):
            target_param.data.copy_(local_param.data)  
        
