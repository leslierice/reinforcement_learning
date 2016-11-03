#!/usr/bin/env python3

# Leslie Rice
# Programming Assignment 4
# Soccer: WoLF / Q-Learning

from matplotlib import pyplot as plt
import numpy as np
import random

DEBUG = False

class WAgent:                               # agent who uses WoLF Policy Hill Climbing
    def __init__(self):
        self.A = ["N", "S", "E", "W", "X"]
        self.pos = [0, 0]
        self.cur_state = "0, 0, 0, 0"
        self.prev_state = "0, 0, 0, 0"
        self.Q = {}
        self.pi = {}
        self.pi_avg = {}
        self.C = {}
        for row_a in range(4):
            for col_a in range(5):
                for row_b in range(4):
                    for col_b in range(5):
                        self.C[str(row_a) + ", " + str(col_a) + ", " + str(row_b) + ", " + str(col_b)] = 0
                        for a in self.A:
                            self.Q[str(row_a) + ", " + str(col_a) + ", " + str(row_b) + ", " + str(col_b) + ", " + a] = 0
                            self.pi[str(row_a) + ", " + str(col_a) + ", " + str(row_b) + ", " + str(col_b) + ", " + a] = 1/len(self.A)
                            self.pi_avg[str(row_a) + ", " + str(col_a) + ", " + str(row_b) + ", " + str(col_b) + ", " + a] = 0

    def set_position(self, pos):
        self.pos = pos

    def set_cur_state(self, cur_state):
        self.cur_state = ", ".join([str(s) for s in cur_state])

    def set_prev_state(self, prev_state):
        self.prev_state = ", ".join([str(s) for s in prev_state])

    def choose_a(self, epsilon):
        if random.random() < epsilon:
            return random.choice(self.A)
        else:
            select_map = {}
            start = 0
            for a in self.A:
                p = self.pi[self.cur_state + ", " + a]
                k = (start, p + start)
                select_map[k] = a
                start += p
            r = random.random()
            for k in select_map.keys():
                if r >= k[0] and r < k[1]:
                    return select_map[k]

    def max_a(self):
        max_val = -float("inf")
        a_choice = None
        for a in self.A:
            cur_val = self.Q[self.prev_state + ", " + a]
            if cur_val > max_val:
                max_val = cur_val
                a_choice = a
        return a_choice

    def max_q(self):
        return max(self.Q[self.cur_state + ", " + a] for a in self.A)

    def update_values(self, a_action, alpha, reward, gamma, delta_w, delta_l):
        self.Q[self.prev_state + ", " + a_action] = (1-alpha) * self.Q[self.prev_state + ", " + a_action] + alpha * (reward + gamma * self.max_q())
        self.C[self.prev_state] += 1
        self.update_pi_avg()
        self.update_pi(delta_w, delta_l, a_action)

    def update_pi_avg(self):
        for a in self.A:
            self.pi_avg[self.prev_state + ", " + a] += 1/self.C[self.prev_state] * (self.pi[self.prev_state + ", " + a] - self.pi_avg[self.prev_state + ", " + a])

    def update_pi(self, delta_w, delta_l, a_action):
        if sum(self.pi[self.prev_state + ", " + a] * self.Q[self.prev_state + ", " + a] for a in self.A) > sum(self.pi_avg[self.prev_state + ", " + a] * self.Q[self.prev_state + ", " + a] for a in self.A):
            delta = delta_w
        else:
            delta = delta_l
        sum_val = 0
        for a in self.A:
            if a == self.max_a():
                self.pi[self.prev_state + ", " + a] += delta
                if self.pi[self.prev_state + ", " + a] > 1:
                    self.pi[self.prev_state + ", " + a] = 1
            else:
                self.pi[self.prev_state + ", " + a] -= delta / (len(self.A)-1)
                if self.pi[self.prev_state + ", " + a] < 0:
                    self.pi[self.prev_state + ", " + a] = 0

class QAgent:                               # agent who uses Q-Learning
    def __init__(self):
        self.A = ["N", "S", "E", "W", "X"]
        self.cur_state = "0, 0, 0, 0"
        self.prev_state = "0, 0, 0, 0"
        self.Q = {}
        self.pi = {}
        for row_a in range(4):
            for col_a in range(5):
                for row_b in range(4):
                    for col_b in range(5):
                        for a in self.A:
                            self.Q[str(row_a) + ", " + str(col_a) + ", " + str(row_b) + ", " + str(col_b) + ", " + a] = 1
                            self.pi[str(row_a) + ", " + str(col_a) + ", " + str(row_b) + ", " + str(col_b), a] = 1/len(self.A)

    def set_position(self, pos):
        self.pos = pos

    def set_cur_state(self, cur_state):
        self.cur_state = ", ".join([str(s) for s in cur_state])

    def set_prev_state(self, prev_state):
        self.prev_state = ", ".join([str(s) for s in prev_state])

    def choose_a(self, epsilon):
        if random.random() < epsilon:
            return random.choice(self.A)
        else:
            return self.max_a()

    def max_a(self):
        max_val = -float("inf")
        a_choice = None
        for a in self.A:
            cur_val = self.Q[self.cur_state + ", " + a]
            if cur_val > max_val:
                max_val = cur_val
                a_choice = a
        return a_choice

    def max_q(self):
        return max(self.Q[self.cur_state + ", " + a] for a in self.A)

    def update_values(self, a_action, alpha, reward, gamma, delta_l, delta_w):
        self.Q[self.prev_state + ", " + a_action] += alpha * (reward + gamma * self.max_q() - self.Q[self.prev_state + ", " + a_action])

class RAgent:                               # agent who uses random policy
    def __init__(self):
        self.A = ["N", "S", "E", "W", "X"]

    def set_position(self, pos):
        self.pos = pos

    def set_cur_state(self, cur_state):
        self.cur_state = cur_state

    def set_prev_state(self, prev_state):
        self.prev_state = prev_state

    def choose_a(self, epsilon):
        return random.choice(self.A)

class Soccer:
    def __init__(self, epsilon, alpha, decay, gamma, version, delta_l=None, delta_w=None):
        self.epsilon = epsilon                  # exploration parameter
        self.alpha = alpha                      # learning rate agent B
        self.decay = decay
        self.gamma = gamma                      # discount factor
        self.version = version                  # versions of learning methods used
        self.delta_l = delta_l
        self.delta_w = delta_w

        if version == "WW":
            self.agentA = WAgent()
            self.agentB = WAgent()
        elif version == "WR":
            self.agentA = WAgent()
            self.agentB = RAgent()
        elif version == "QQ":
            self.agentA = QAgent()
            self.agentB = QAgent()
        else:
            self.agentA = QAgent()
            self.agentB = RAgent()

        self.reset_position()

    def reset_position(self):
        self.agentA.set_position([2,3])
        self.agentB.set_position([1,1])
        self.p = random.choice(["a", "b"])      # possession of ball
        cur_state = self.agentA.pos + self.agentB.pos
        self.agentA.set_cur_state(cur_state)
        self.agentB.set_cur_state(cur_state)

    def action(self, a):                        # give numeric move for action letter
        if a == "N":
            return [1, 0]
        elif a == "S":
            return [-1, 0]
        elif a == "E":
            return [0, 1]
        elif a == "W":
            return [0, -1]
        else:
            return [0, 0]

    def move_a(self, a_action):                   # move A according to action given
        reward = 0
        new_a_pos = list(map(sum, zip(self.agentA.pos, self.action(a_action))))
        if new_a_pos == self.agentB.pos:
            self.p = "b"
        else:
            if (new_a_pos == [1, -1] or new_a_pos == [2, -1]) and self.p == "a":                      # A scores
                reward = 1
            if new_a_pos[0] >= 0 and new_a_pos[0] <= 3 and new_a_pos[1] >= 0 and new_a_pos[1] <= 4:   # move if action keeps agent in grid range
                self.agentA.set_position(new_a_pos)
        return reward

    def move_b(self, b_action):                   # move B according to action given
        reward = 0
        new_b_pos = list(map(sum, zip(self.agentB.pos, self.action(b_action))))
        if new_b_pos == self.agentA.pos:
            self.p = "a"
        else:
            if (new_b_pos == [1, 5] or new_b_pos == [2, 5]) and self.p == "b":                       # B scores
                reward = 1
            if new_b_pos[0] >= 0 and new_b_pos[0] <= 3 and new_b_pos[1] >= 0 and new_b_pos[1] <= 4:  # move if action keeps agent in grid range
                self.agentB.set_position(new_b_pos)
        return reward

    def execute_moves(self, a_action, b_action):      # execute moves in random order and corresponding return rewards
        first = random.choice(["a", "b"])
        if first == "a":
            reward_a = self.move_a(a_action)
            if reward_a == 0:
                reward_b = self.move_b(b_action)
                reward_a = -reward_b
            else:
                reward_b = -1
        else:
            reward_b = self.move_b(b_action)
            if reward_b == 0:
                reward_a = self.move_a(a_action)
                reward_b = -reward_a
            else:
                reward_a = -1
        return reward_a, reward_b

    def game(self, train=True, a_fixed=False, b_fixed=False):
        self.reset_position()
        reward_a = 0
        reward_b = 0
        t = 0
        while reward_a == 0 and reward_b == 0:        # continue game until someone scores
            t += 1
            if random.random() >= 0.1 or train:
                a_action = self.agentA.choose_a(self.epsilon)
                b_action = self.agentB.choose_a(self.epsilon)

                if DEBUG:
                    print("a prev: ", self.agentA.pos)
                    print("b prev: ", self.agentB.pos)
                    print("a action: ", a_action)
                    print("b action: ", b_action)

                prev_state = self.agentA.pos + self.agentB.pos
                self.agentA.set_prev_state(prev_state)
                self.agentB.set_prev_state(prev_state)

                reward_a, reward_b = self.execute_moves(a_action, b_action)

                if DEBUG:
                    print("a new: ", self.agentA.pos)
                    print("b new: ", self.agentB.pos)
                    print("reward a: ", reward_a)
                    print("reward b: ", reward_b)
                    print("")

                cur_state = self.agentA.pos + self.agentB.pos
                self.agentA.set_cur_state(cur_state)
                self.agentB.set_cur_state(cur_state)
                if not a_fixed:
                    self.agentA.update_values(a_action, self.alpha, reward_a, self.gamma, self.delta_l, self.delta_w)
                if (type(self.agentB) is WAgent or type(self.agentB) is QAgent) and not b_fixed:
                    self.agentB.update_values(b_action, self.alpha, reward_b, self.gamma, self.delta_l, self.delta_w)
                self.alpha *= self.decay
                if (self.delta_l):
                    self.delta_l *= self.decay
                    self.delta_w *= self.decay
        if DEBUG:
            print("t steps: ", t)
            print("")
            print("")
        if reward_a == -1:
            reward_a = 0
        return t, reward_a

def main():
    steps_trained = 1250000
    steps_tested = 100000

    # WW
    # Train AgentA
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "WW", 0.8, 0.2)
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game()
        steps += cur_steps
    trained_agentA = s.agentA

    # Train Competitor
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    s.agentB = trained_agentA
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game(b_fixed=True)
        steps += cur_steps
    trained_agentB = s.agentA

    # Test
    # vs. Random
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "WR")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_WWR = games_played
    percent_won_WWR = won / games_played * 100
    print("num games: ", games_played)
    print("WW vs. random percentage games won: ", won/games_played * 100, "%")

    # vs. Competitor
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    s.agentB = trained_agentB
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_WWC = games_played
    percent_won_WWC = won / games_played * 100
    print("num games: ", games_played)
    print("WW vs. competitor percentage games won: ", won/games_played * 100, "%")

    # WR
    # Train AgentA
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "WR", 0.8, 0.2)
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game()
        steps += cur_steps
    trained_agentA = s.agentA

    # Train Competitor
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    s.agentB = trained_agentA
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game(b_fixed=True)
        steps += cur_steps
    trained_agentB = s.agentA

    # Test
    # vs. random
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "WR")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_WRR = games_played
    percent_won_WRR = won / games_played * 100
    print("num games: ", games_played)
    print("WR vs. random percentage games won: ", won/games_played * 100, "%")

    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    s.agentB = trained_agentB
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_WRC = games_played
    percent_won_WRC = won / games_played * 100
    print("num games: ", games_played)
    print("WR vs. competitor percentage games won: ", won/games_played * 100, "%")

    # QQ
    # Train AgentA
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game()
        steps += cur_steps
    trained_agentA = s.agentA

    # Train Competitor
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    s.agentB = trained_agentA
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game(b_fixed=True)
        steps += cur_steps
    trained_agentB = s.agentA

    # Test
    # vs. Random
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QR")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_QQR = games_played
    percent_won_QQR = won / games_played * 100
    print("num games: ", games_played)
    print("QQ vs. random percentage games won: ", won/games_played * 100, "%")

    # vs. Competitor
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    s.agentB = trained_agentB
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_QQC = games_played
    percent_won_QQC = won / games_played * 100
    print("num games: ", games_played)
    print("QQ vs. competitor percentage games won: ", won/games_played * 100, "%")

    # QR
    # Train AgentA
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QR")
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game()
        steps += cur_steps
    trained_agentA = s.agentA

    # Train Competitor
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    s.agentB = trained_agentA
    steps = 0
    while steps < steps_trained:
        cur_steps, won = s.game(b_fixed=True)
        steps += cur_steps
    trained_agentB = s.agentA

    # Test
    # vs. Random
    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QR")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    s.agentB = RAgent()
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_QRR = games_played
    percent_won_QRR = won / games_played * 100
    print("num games: ", games_played)
    print("QR vs. random percentage games won: ", won/games_played * 100, "%")


    s = Soccer(0.2, 1.0, 0.99999999, 0.9, "QQ")
    steps = 0
    won = 0
    s.agentA = trained_agentA
    s.agentB = trained_agentB
    games_played = 0
    while steps < steps_tested:
        cur_steps, cur_won = s.game(train=False, a_fixed=True, b_fixed=True)
        steps += cur_steps
        won += cur_won
        games_played += 1
    num_games_QRC = games_played
    percent_won_QRC = won / games_played * 100
    print("num games: ", games_played)
    print("QR vs. competitor percentage games won: ", won/games_played * 100, "%")


    num_games = [num_games_WWR, num_games_WWC, num_games_WRR, num_games_WRC, num_games_QQR, num_games_QQC, num_games_QRR, num_games_QRC]
    labels = ["WWR", "WWC", "WRR", "WRC", "QQR", "QQC", "QRR", "QRC"]

    fig = plt.figure()

    width = .35
    ind = np.arange(len(num_games))
    plt.bar(ind, num_games, width=width)
    plt.xticks(ind + width / 2, labels)
    plt.title("Number of Games Played")
    fig.savefig("5.png")
    plt.close(fig)

    percent_won = [percent_won_WWR, percent_won_WWC, percent_won_WRR, percent_won_WRC, percent_won_QQR, percent_won_QQC, percent_won_QRR, percent_won_QRC]
    labels = ["WWR", "WWC", "WRR", "WRC", "QQR", "QQC", "QRR", "QRC"]

    fig = plt.figure()

    width = .35
    ind = np.arange(len(percent_won))
    plt.bar(ind, percent_won, width=width)
    plt.xticks(ind + width / 2, labels)
    plt.title("Percent Won")
    fig.savefig("6.png")
    plt.close(fig)

main()
