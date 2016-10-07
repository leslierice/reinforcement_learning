#!/usr/bin/env python3

# Leslie Rice
# Programming Assignment 4
# Baird's Counterexample, Semi-Gradient TD(0)

from matplotlib import pyplot as plt
import random

def main():
    final_theta_vals = [[0 for j in range(0, 100)] for i in range(0, 8)]
    for i in range(0, 100):
        theta_vals = [[] for i in range(0, 8)]
        theta = [1, 1, 1, 1, 1, 1, 10, 1]
        alpha = 0.001
        for e in range(0, 100):
            states = [[2, 0, 0, 0, 0, 0, 0, 1], [0, 2, 0, 0, 0, 0, 0, 1], [0, 0, 2, 0, 0, 0, 0, 1], \
                 [0, 0, 0, 2, 0, 0, 0, 1], [0, 0, 0, 0, 2, 0, 0, 1], [0, 0, 0, 0, 0, 2, 0, 1], \
                 [0, 0, 0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0]]
            cur_state = 0
            terminated = False
            while not terminated:
                if random.random() < 1/7:   # solid action
                    if random.random() < 0.01:
                        terminated = True
                        next_state = 7
                    else:
                        next_state = 6
                    action = "solid"
                else:                       # dotted action
                    if random.random() < 0.01:
                        terminated = True
                        next_state = 7
                    else:
                        val = random.random()
                        if val < 1/6:
                            next_state = 0
                        elif val < 2/6:
                            next_state = 1
                        elif val < 3/6:
                            next_state = 2
                        elif val < 4/6:
                            next_state = 3
                        elif val < 5/6:
                            next_state = 4
                        else:
                            next_state = 5
                    action = "dotted"
                if action == "solid":
                    p = 1 / (1/7)
                else:
                    p = 0
                v_next = 0
                v_cur = 0
                for i in range(0, 8):
                    v_next += theta[i] * states[next_state][i]
                    v_cur += theta[i] * states[cur_state][i]
                d = v_next - v_cur
                if not terminated:
                    for index in range(0, len(theta)):
                        theta[index] += alpha * p * d * states[cur_state][index]
                    cur_state = next_state
            for i in range(0, 8):
                theta_vals[i].append(theta[i])
        for i in range(0, 8):
            for j in range(0, 100):
                final_theta_vals[i][j] += theta_vals[i][j]
    for i in range(0, 8):
        for j in range(0, 100):
            final_theta_vals[i][j] /= 100

    colors = ["#FF3333", "#FF8233", "#FFF233", "#49FF33", "#33FFE1", "#3350FF", "#CB33FF", "#FF33C3"]
    for i in range(0, 8):
        plt.plot(range(0, 100), final_theta_vals[i], label="theta_{0}".format(i+1), color=colors[i])

    plt.legend(loc="upper left")
    plt.xlabel("Episodes")
    plt.ylabel("Components of the parameter vector at the end of the episode")
    plt.show()
if __name__ == "__main__" :
    main()
