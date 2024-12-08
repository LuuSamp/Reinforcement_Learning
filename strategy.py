import math

class EpsilonGreedyDecay:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, rate=0.99, type="exp", steps=None):
        self.epsilon = epsilon_start
        self.start = epsilon_start
        self.end = epsilon_end
        self.decay = rate
        self.type = type
        self.steps = steps

    def get_epsilon(self, step):
        if self.type == "exp":
            self.epsilon = self.end + (self.start - self.end) * math.exp(-1. * step / self.decay)
        elif self.type == "linear":
            self.epsilon = self.start + step * (self.end - self.start) / self.decay
        return self.epsilon
    