import pandas as pd
import numpy as np
from pylab import loglog, semilogx
import matplotlib.pyplot as plt
from bandits import EpsilonGreedyBandit, BayesianBandit

# np.random.seed(42)

def test_bandit(bandit, num_trials=3000):
    trials = np.zeros(shape=(num_trials, 3))
    successes = np.zeros(shape=(num_trials, 3))

    for i in range(num_trials):
        s = data.sample().values
        choice = bandit.get_recommendation()
        conv = s[0][choice]
        bandit.add_result(choice, conv)

        trials[i] = bandit.trials
        successes[i] = bandit.successes
    return trials, successes

data = pd.read_csv('Bandits income data.csv')
for column in data.columns:
    print('Frequentist success rates for {} : {}'.format(column, data[column].sum()/len(data)))

bandit = BayesianBandit(num_options=3)
# bandit = EpsilonGreedyBandit(num_options=3, epsilon=0.1)
data = data.sample(n=len(data))

# for row in data_sample.sample(10).values:
#     bandit.add_result([0,1,2], row)

num_trials = 5000
trials, successes = test_bandit(bandit, num_trials=num_trials)

plt.subplot(211)
n = np.arange(num_trials)+1
loglog(n, trials[:, 0], label="bandit 1")
loglog(n, trials[:, 1], label="bandit 2")
loglog(n, trials[:, 2], label="bandit 3")

plt.legend()
plt.xlabel("Number of trials")
plt.ylabel("Number of trials per bandit")

plt.subplot(212)
semilogx(n, np.sum(successes, axis=1)/n, label="Successes")
plt.xlabel("Number of trials")
plt.ylabel("Successes")
plt.legend()
plt.show()


