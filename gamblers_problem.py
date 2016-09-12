#!/usr/bin/env python3

# Leslie Rice
# Programming Assignment 2
# Gambler's Problem: Value Iteration

from matplotlib import pyplot as plt

class Game:
    def __init__(self, ph, theta, epsilon):
        self.V = {}
        self.ph = ph                    # probability of coin coming up heads
        self.theta = theta
        self.epsilon = epsilon
        self.all_states = [i for i in range(0, 101)]
        self.states = [i for i in range(1, 100)]
        self.Pi = {}

        for s in self.all_states:       # initialize V(s) to 0 for s != 100, V(100) = 1
            self.V[s] = 0
        self.V[100] = 0

    def max_val(self, s):
        max_val = -1
        for a in range(1, min(s, 100 - s) + 1):
            sum_val = 0
            win = s + a
            lose = s - a
            if win >= 100:
                win = 100
                sum_val += self.ph * (1 + self.epsilon * self.V[win])
            else:
                sum_val += self.ph * (0 + self.epsilon * self.V[win])
            if lose < 0:
                lose = 0
            sum_val += (1-self.ph) * (0 + self.epsilon * self.V[lose])
            if sum_val > max_val:
                max_val = sum_val
                self.Pi[s] = a
        return max_val

    def value_iterate(self):
        delta = 0
        for s in self.states:
            v = self.V[s]
            self.V[s] = self.max_val(s)
            print(self.V[s])
            delta = max(delta, abs(v - self.V[s]))
        return delta

    def run(self):
        delta = self.value_iterate()
        num_iterations = 1
        while num_iterations < 1000:
            delta = self.value_iterate()
            num_iterations += 1
        return num_iterations


def main():
    ph = 0.7
    theta = 1e-5
    epsilon = 1
    game = Game(ph, theta, epsilon)
    num_iterations = game.run()
    print(num_iterations)
    final_policy = list(game.Pi.values())
    value_estimates = list(game.V.values())
    plt.plot(range(1, 100), value_estimates[1:len(value_estimates)-1])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    title = "ph = {0}".format(ph)
    plt.title(title)
    plt.show()
    plt.plot(range(1, 100), final_policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.ylim(0, 50)
    plt.title(title)
    plt.show()


if __name__ == "__main__" :
    main()
