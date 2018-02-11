from scipy.stats import beta
import numpy as np
import random

class Bandit(object):
    def __init__(self, num_options):
        self.num_options = num_options
        self.trials = np.zeros(shape=(num_options,), dtype=int)
        self.successes = np.zeros(shape=(num_options,), dtype=int)

    def add_result(self, bandit_ids, success):
        self.trials[bandit_ids] = self.trials[bandit_ids] + 1
        self.successes[bandit_ids] = self.successes[bandit_ids] + success

    def get_recommendation(self):
        raise NotImplementedError()

class BayesianBandit(Bandit):
    def __init__(self, num_options=3, prior=None):
        super().__init__(num_options)

        if prior is None:
            prior = [(1, 1)] * num_options
        self.prior = prior

    def add_result(self, bandit_ids, success):
        self.trials[bandit_ids] = self.trials[bandit_ids] + 1
        self.successes[bandit_ids] = self.successes[bandit_ids] + success

    def get_recommendation(self):
        sampled_theta = []
        for i in range(self.num_options):
            dist = beta(self.prior[i][0]+self.successes[i],
                        self.prior[i][1] + self.trials[i] - self.successes[i])
            sampled_theta += [ dist.rvs() ]
        return sampled_theta.index( max(sampled_theta) )

class EpsilonGreedyBandit(Bandit):
    def __init__(self, num_options=3, epsilon=0.1):
        super().__init__(num_options)
        self.epsilon = epsilon

    def get_recommendation(self):
        if (random.random() > self.epsilon) and sum(self.successes) > 0:
            return np.argmax(np.divide(self.successes, self.trials))
        else:
            return random.randrange(len(self.successes))
