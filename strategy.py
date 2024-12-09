import math
from random import random, choice

class EpsilonGreedyDecay:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, rate=0.99, type="exp", steps=1000):
        self.epsilon = epsilon_start
        self.start = epsilon_start
        self.end = epsilon_end
        self.decay = rate
        self.type = type
        self.steps = steps
        self.step = 0

    def update_epsilon(self):
        if self.type == "exp":
            self.epsilon = self.end + (self.start - self.end) * math.exp(-1. * self.step * (1- self.decay))
        elif self.type == "linear":
            self.epsilon = self.start + self.step*(self.end - self.start) / self.decay
        
        self.step += 1
            
        if self.epsilon < self.end:
            self.epsilon = self.end
        return self.epsilon
    
    def get_action(self, q_values):
        self.update_epsilon()
        if random() < self.epsilon:
            return choice(list(q_values.keys()))
        else:
            return max(q_values, key=q_values.get)
        
    def update_q_values(self, q_values, action, reward, alpha):
        q_values[action] += alpha * (reward-q_values[action])
        return q_values