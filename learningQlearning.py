# Imports
import random
import numpy as np
from matplotlib import pyplot as plt

# Define Reward/Environment matrix, R
R = np.array([
        [-1,  -1,  -1,  -1,   0,  -1],
        [-1,  -1,  -1,   0,  -1,   0],
        [-1,  -1,  -1,   0,  -1,  -1],
        [-1,   0,   0,  -1,   0,  -1],
        [ 0,  -1,  -1,   0,  -1,   0],
        [-1,   0,  -1,  -1,   0,   0]])
# Double checked, correct!

TARGET = 5
argarr = np.argwhere(np.equal(R[:, TARGET], 0)).ravel()
R[argarr, TARGET] = 100


def hardmax(Z):
    sZ = sum(Z)
    return [z / sZ for z in Z]


def softmax(Z):
    eZ = np.exp(Z)
    return eZ / np.sum(eZ)


class Agent:
    """Representation of an agent, acting in an environment (R)"""

    def __init__(self, gamma=0.8):
        self.Q = np.zeros_like(R) + 0.5  # Memory matrix
        self.gamma = gamma  # learning hyperparameter

    def reward(self, direct_reward, expected_reward):
        return direct_reward + self.gamma * expected_reward

    def fit(self, steps, environment, display=None, master=None):
        state = random.randrange(len(environment))

        for step in range(1, steps + 1):
            # print("Episode: {:>3}, state: {}".format(episode, state))
            valid = [i for i, p in enumerate(environment[state]) if p >= 0]
            action = random.choice(valid)
            self.Q[state, action] = self.reward(direct_reward=environment[state, action],
                                                expected_reward=max(self.Q[action]))
            np.clip(self.Q, -1, 100, out=self.Q)
            if display is not None:
                display.set_data(self.Q)
                master.pause(0.1)
            state = action

    def run(self, environment, runs=100):
        steps_required = []
        for _ in range(runs):
            state = random.randrange(len(environment))
            steps = 0
            while state != 5:
                arg, prob = zip(*[(i, prb) for i, prb in enumerate(self.Q[state]) if prb >= 0.])
                prob = softmax(prob) if sum(prob) > 0. else np.ones_like(prob) / len(prob)
                state = np.random.choice(arg, p=prob)
                steps += 1
            steps_required.append(steps)
        print("Average from {} runs: {} steps".format(runs, sum(steps_required) / len(steps_required)))

if __name__ == '__main__':
    agent = Agent()
    # agent.run(R)
    agent.run(R)

    plt.ion()
    mat = plt.imshow(np.zeros_like(agent.Q), vmin=0., vmax=100., interpolation="none")

    agent.fit(100, R, display=mat, master=plt)

    plt.close()

    agent.run(R)

    # THIS IS AWESOME!!!!!!!!
