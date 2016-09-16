#!/usr/bin/env python3

# Leslie Rice
# Programming Assignment 3
# Windy Gridworld: Sarsa, Q-Learning, Expected Sarsa

from matplotlib import pyplot as plt
import random
import numpy

class Gridworld:
    def __init__(self, epsilon, alpha, gamma, version, stochastic):
        self.epsilon = epsilon                                      # epsilon for e-greedy policy
        self.alpha = alpha                                          # step size parameter
        self.gamma = gamma                                          # discount factor
        self.stochastic = stochastic                                # wind stochastic

        self.grid = [[0 for i in range(10)] for i in range(7)]      # gridworld
        for i in range(0, 7):                                       # set crosswind
            self.grid[i][3] = 1
            self.grid[i][4] = 1
            self.grid[i][5] = 1
            self.grid[i][6] = 2
            self.grid[i][7] = 2
            self.grid[i][8] = 1

        self.A = [[None for i in range(10)] for i in range(7)]      # actions across grid

        if version == 1:
            for row in range(1, 6):                                 # initial four moves: up, down, left, right
                for col in range(1, 9):
                    self.A[row][col] = ["up", "down", "left", "right"]
            for col in range(1, 9):
                    self.A[0][col] = ["down", "left", "right"]
            for col in range(1, 9):
                    self.A[6][col] = ["up", "left", "right"]
            self.A[0][0] = ["down", "right"]
            self.A[0][9] = ["down", "left"]
            self.A[6][0] = ["up", "right"]
            self.A[6][9] = ["up", "left"]
            for row in range(1, 6):
                    self.A[row][0] = ["up", "down", "right"]
            for row in range(1, 6):
                    self.A[row][9] = ["up", "down", "left"]

        if version == 2:
            for row in range(1, 6):                                    # initial four moves + king's moves (diagonals)
                for col in range(1, 9):
                    self.A[row][col] = ["up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right"]
            for col in range(1, 9):
                    self.A[0][col] = ["down", "left", "right", "down-left", "down-right"]
            for col in range(1, 9):
                    self.A[6][col] = ["up", "left", "right", "up-left", "up-right"]
            self.A[0][0] = ["down", "right", "down-right"]
            self.A[0][9] = ["down", "left", "down-left"]
            self.A[6][0] = ["up", "right", "up-right"]
            self.A[6][9] = ["up", "left", "up-left"]
            for row in range(1, 6):
                    self.A[row][0] = ["up", "down", "right", "up-right", "down-right"]
            for row in range(1, 6):
                    self.A[row][9] = ["up", "down", "left", "up-left", "down-left"]


        self.Q = {}                                                 # initialize Q(s,a) = 0 for all s,a
        for row in range(7):
            for col in range(10):
                for a in self.A[row][col]:
                    self.Q["{0},{1}: {2}".format(row, col, a)] = 0

    def choose_a(self, row, col, time_steps):                                   # choose action based on e-greedy policy
        if random.random() < self.epsilon:
            return random.choice(self.A[row][col])
        else:
            return self.max_a(row, col)
        # if not time_steps:
        #     time_steps = 1
        # if random.random() < 1 / time_steps:
        #     return random.choice(self.A[row][col])
        # else:
        #     return self.max_a(row, col)

    def max_a(self, row, col):                                      # choose action a with max Q(s, a) for given s
        max_val = -99999
        a_choice = None
        for a in self.A[row][col]:
            if self.Q["{0},{1}: {2}".format(row, col, a)] > max_val:
                max_val = self.Q["{0},{1}: {2}".format(row, col, a)]
                a_choice = a
        return a_choice

    def max_q(self, row, col):                                      # max Q(s, a) for given s
        max_val = -99999
        for a in self.A[row][col]:
            if self.Q["{0},{1}: {2}".format(row, col, a)] > max_val:
                max_val = self.Q["{0},{1}: {2}".format(row, col, a)]
        return max_val

    def expected_q(self, row, col):                                 # expected Q(s, a) for given s
        max_a = self.max_a(row, col)
        sum_val = 0
        for a in self.A[row][col]:
            if a is max_a:
                sum_val += (1-self.epsilon) * self.Q["{0},{1}: {2}".format(row, col, a)]
            else:
                sum_val += self.epsilon/(len(self.A[row][col])-1) * self.Q["{0},{1}: {2}".format(row, col, a)]
        return sum_val

    def take_a(self, row, col, a):                                  # take action a, return next state
        fell = False                                                # detect whether wind causes agent to fall off grid
        # if self.stochastic:                                         # force of wind is stochastic
        #     row += int(numpy.random.normal(self.grid[row][col], 3))
        # else:
        row -= self.grid[row][col]
        if a == "up":
            row -= 1
        elif a == "down":
            row += 1
        elif a == "left":
            col -= 1
        elif a == "right":
            col += 1
        elif a == "up-left":
            row -= 1
            col -= 1
        elif a == "up-right":
            row -= 1
            col += 1
        elif a == "down-left":
            row += 1
            col -= 1
        else:
            row += 1
            col += 1
        if row < 0:                             # check if wind caused to fall off grid
            row = 0
            fell = True
        elif row > 6:
            row = 6
            fell = True
        return row, col, fell

    def sarsa_episode(self, total_time):
        time_steps = 0
        row = 3                                 # start from row 3, column 0 every time
        col = 0
        a = self.choose_a(row, col, total_time)
        while not(row == 3 and col == 7):       # repeat until s is terminal
            time_steps += 1
            row_p, col_p, fell = self.take_a(row, col, a)
            a_p = self.choose_a(row_p, col_p, total_time + time_steps)
            # print("going {0} at {1},{2} and move to {3},{4} then going {5}".format(a, row, col, row_p, col_p, a_p))
            if row_p == 3 and col_p == 7:
                reward = 0
            elif fell:
                reward = -100
            else:
                reward = -1
            self.Q["{0},{1}: {2}".format(row, col, a)] += self.alpha * (reward + self.gamma * self.Q["{0},{1}: {2}".format(row_p, col_p, a_p)] - self.Q["{0},{1}: {2}".format(row, col, a)])
            row = row_p
            col = col_p
            a = a_p
        return time_steps

    def q_learning_episode(self, total_time):
        time_steps = 0
        row = 3                                 # start from row 3, column 0 every time
        col = 0
        while not(row == 3 and col == 7):       # repeat until s is terminal
            time_steps += 1
            a = self.choose_a(row, col, total_time + time_steps)
            row_p, col_p, fell = self.take_a(row, col, a)
            # print("going {0} at {1},{2} and move to {3},{4}".format(a, row, col, row_p, col_p))
            if row_p == 3 and col_p == 7:
                reward = 0
            elif fell:
                reward = -100
            else:
                reward = -1
            self.Q["{0},{1}: {2}".format(row, col, a)] += self.alpha * (reward + self.gamma * self.max_q(row_p, col_p) - self.Q["{0},{1}: {2}".format(row, col, a)])
            row = row_p
            col = col_p
        return time_steps

    def expected_sarsa_episode(self, total_time):
        time_steps = 0
        row = 3                                 # start from row 3, column 0 every time
        col = 0
        while not(row == 3 and col == 7):       # repeat until s is terminal
            time_steps += 1
            a = self.choose_a(row, col, total_time + time_steps)
            row_p, col_p, fell = self.take_a(row, col, a)
            # print("going {0} at {1},{2} and move to {3},{4}".format(a, row, col, row_p, col_p))
            if row_p == 3 and col_p == 7:
                reward = 0
            elif fell:
                reward = -100
            else:
                reward = -1
            self.Q["{0},{1}: {2}".format(row, col, a)] += self.alpha * (reward + self.gamma * self.expected_q(row_p, col_p) - self.Q["{0},{1}: {2}".format(row, col, a)])
            row = row_p
            col = col_p
        return time_steps

def run(method, version, stochastic):
    g = Gridworld(0.1, 0.5, 0.75, version, stochastic)
    total_results = [0 for i in range(8001)]
    num_episodes = 0
    time_steps = 0
    results = [0 for i in range(8001)]
    while time_steps < 8000:
        if method == "sarsa":
            ep_time_steps = g.sarsa_episode(time_steps)
        elif method == "q-learning":
            ep_time_steps = g.q_learning_episode(time_steps)
        else:
            ep_time_steps = g.expected_sarsa_episode(time_steps)
        if time_steps + ep_time_steps > 8000:
            break
        for i in range(time_steps, time_steps + ep_time_steps):
            results[i] = num_episodes
        # print(ep_time_steps)
        num_episodes += 1
        time_steps += ep_time_steps
    for i in range(time_steps, 8001):
        results[i] = num_episodes + 1
    return results

def main():
    # total_sarsa_results = [0 for i in range(8001)]
    # total_q_learning_results = [0 for i in range(8001)]
    # total_expected_sarsa_results = [0 for i in range(8001)]
    # for i in range(2000):
    #     sarsa_results = run("sarsa", 1, 0)
    #     q_learning_results = run("q-learning", 1, 0)
    #     expected_sarsa_results = run("expected_sarsa", 1, 0)
    #     for i in range(8001):
    #         total_sarsa_results[i] += sarsa_results[i]
    #         total_q_learning_results[i] += q_learning_results[i]
    #         total_expected_sarsa_results[i] += expected_sarsa_results[i]
    # for i in range(8001):
    #     total_sarsa_results[i] /= 2000
    #     total_q_learning_results[i] /= 2000
    #     total_expected_sarsa_results[i] /= 2000
    # plt.plot(range(0, 8001), total_sarsa_results, label="Sarsa")
    # plt.plot(range(0, 8001), total_q_learning_results, label="Q-Learning")
    # plt.plot(range(0, 8001), total_expected_sarsa_results, label="Expected Sarsa")
    # plt.legend(loc="lower right")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Episodes")
    # plt.title("Four moves, epsilon=0.1, alpha=0.5, gamma=0.75")
    # plt.show()

    total_sarsa_results = [0 for i in range(8001)]
    total_q_learning_results = [0 for i in range(8001)]
    total_expected_sarsa_results = [0 for i in range(8001)]
    for i in range(2000):
        sarsa_results = run("sarsa", 2, 1)
        q_learning_results = run("q-learning", 2, 1)
        expected_sarsa_results = run("expected_sarsa", 2, 1)
        for i in range(8001):
            total_sarsa_results[i] += sarsa_results[i]
            total_q_learning_results[i] += q_learning_results[i]
            total_expected_sarsa_results[i] += expected_sarsa_results[i]
    for i in range(8001):
        total_sarsa_results[i] /= 2000
        total_q_learning_results[i] /= 2000
        total_expected_sarsa_results[i] /= 2000
    plt.plot(range(0, 8001), total_sarsa_results, label="Sarsa")
    plt.plot(range(0, 8001), total_q_learning_results, label="Q-Learning")
    plt.plot(range(0, 8001), total_expected_sarsa_results, label="Expected Sarsa")
    plt.legend(loc="lower right")
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.title("King's moves, epsilon=0.1, alpha=0.5, gamma=0.75")
    plt.show()


    # total_sarsa_results = [0 for i in range(8001)]
    # total_q_learning_results = [0 for i in range(8001)]
    # total_expected_sarsa_results = [0 for i in range(8001)]
    # for i in range(2000):
    #     sarsa_results = run("sarsa", 2, 1)
    #     q_learning_results = run("q-learning", 2, 1)
    #     expected_sarsa_results = run("expected_sarsa", 2, 1)
    #     for i in range(8001):
    #         total_sarsa_results[i] += sarsa_results[i]
    #         total_q_learning_results[i] += q_learning_results[i]
    #         total_expected_sarsa_results[i] += expected_sarsa_results[i]
    # for i in range(8001):
    #     total_sarsa_results[i] /= 2000
    #     total_q_learning_results[i] /= 2000
    #     total_expected_sarsa_results[i] /= 2000
    # plt.plot(range(0, 8001), total_sarsa_results, label="Sarsa")
    # plt.plot(range(0, 8001), total_q_learning_results, label="Q-Learning")
    # plt.plot(range(0, 8001), total_expected_sarsa_results, label="Expected Sarsa")
    # plt.legend(loc="lower right")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Episodes")
    # plt.title("King's moves + stochastic wind, epsilon=0.1, alpha=0.5, gamma=0.75")
    # plt.show()




if __name__ == "__main__" :
    main()
