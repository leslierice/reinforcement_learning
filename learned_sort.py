#!/usr/bin/env python3

# Algorithm Selection using Reinforcement Learning
# Sorting: QuickSort vs. InsertionSort

from matplotlib import pyplot as plt
from numpy.linalg import inv
import numpy as np
import random
import time
import math

# QuickSort

def quicksort(l):
    quicksort_helper(l, 0, len(l) - 1)

def quicksort_helper(l, first, last):
    if first < last:
        split = quicksort_partition(l, first, last)
        quicksort_helper(l, first, split - 1)
        quicksort_helper(l, split + 1, last)

def quicksort_partition(l, first, last):
    pivot = l[first]
    left = first + 1
    right = last
    done = False
    while not done:
        while left <= right and l[left] <= pivot:
            left = left + 1
        while l[right] >= pivot and right >= left:
            right = right - 1
        if right < left:
            done = True
        else:
            temp = l[left]
            l[left] = l[right]
            l[right] = temp
    temp = l[first]
    l[first] = l[right]
    l[right] = temp
    return right

# InsertionSort

def insertion_sort(l, first, last):
    for index in range(first+1, last+1):
        currentvalue = l[index]
        position = index
        while position > first and l[position-1] > currentvalue:
            l[position] = l[position-1]
            position = position-1
        l[position] = currentvalue
    return None


class Action:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.w = np.matrix('0.0 ; 0.0 ; 0.0')
        self.A = np.matrix('1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0')
        self.b = np.matrix('0.0 ; 0.0 ; 0.0')

    def calc_q(self, s):
        return np.transpose(self.w) * self.phi(s)

    def phi(self, s):
        return np.matrix([[pow(s, 2)], [s * math.log(s, 2)], [s]])

    def update_weights(self):
        self.w = inv(self.A) * self.b

    def update_A(self, s):
        self.A += self.phi(s) * np.transpose(self.phi(s))

    def update_b(self, s, c):
        self.b += self.phi(s) * c


class LearnedSort:
    def __init__(self, epsilon, alpha, gamma, algorithms):
        self.epsilon = epsilon      # epsilon for e-greedy policy
        self.alpha = alpha          # step size parameter
        self.gamma = gamma          # discount factor

        self.actions = []
        for a in algorithms:
            self.actions.append(Action(a))

        self.cutoff_point = 0

    def compute_policy(self, s):     # after training, pre-compute learned policy (cutoff point)
        for i in range(1, s+1):
            if self.min_a(i).algorithm == insertion_sort and self.min_a(i+1).algorithm == quicksort_partition:
                self.cutoff_point = i+1
                break

    def choose_a(self, s):              # choose action according to e-greedy policy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.min_a(s)

    def min_a(self, s):                 # return action with min Q val for given state
        min_val = float("inf")
        a_choice = None
        for a in self.actions:
            val = a.calc_q(s)
            if val < min_val:
                min_val = val
                a_choice = a
        return a_choice

    def min_q(self, s):                 # return min Q value for given state
        return min([a.calc_q(s) for a in self.actions])

    def monte_carlo(self, l, first, last, total_cost):     # monte-carlo return
        total_cost = 0
        if (first < last):
            s = last - first + 1
            a = self.min_a(s)

            start = time.clock()
            p = a.algorithm(l, first, last)
            end = time.clock()
            total_cost += end - start

            if p is not None:
                self.monte_carlo(l, first, p-1, total_cost)
                self.monte_carlo(l, p+1, last, total_cost)
        return total_cost

    def learning_episode(self, l):       # given a list to sort, sort based on Q-learning method
        done = False

        first = 0
        last = len(l) - 1
        s = last - first + 1             # current size of partition

        while not done and first < last:
            a = self.choose_a(s)                # choose algorithm to execute

            start = time.clock()
            p = a.algorithm(l, first, last)     # execute algorithm
            end = time.clock()
            c = end - start

            if p is None:                       # insertion sort was used, list is now sorted
                m_val = 0
                s_new = 1
                done = True
            else:                               # run monte carlo algorithm on smaller partition
                if (p - first < last - p):
                    m_val = self.monte_carlo(l, first, p-1, 0)        # monte-carlo return
                    first = p + 1
                else:
                    m_val = self.monte_carlo(l, p+1, last, 0)        # monte-carlo return
                    last = p - 1
                s_new = last - first + 1

            q_val = (1 - self.alpha) * a.calc_q(s) + self.alpha * (c + m_val + self.min_q(s_new))
            a.update_A(s)                 # update matrix A
            a.update_b(s, q_val)          # update vector b
            a.update_weights()            # update weights

            s = s_new

    def episode(self, l, first, last):
        if (first < last):
            if last - first + 1 < self.cutoff_point:
                insertion_sort(l, first, last)
            else:
                p = quicksort_partition(l, first, last)
                self.episode(l, first, p-1)
                self.episode(l, p+1, last)

def main():
    learn_max_size = 50
    exec_max_size = 1000
    val_max = 100
    num_training_eps = 1000
    num_exec_eps = 1000

    algorithms = [quicksort_partition, insertion_sort]

    g = LearnedSort(0.6, 1.0, 0.75, algorithms)

    training_sizes = sorted([random.randint(1, learn_max_size) for i in range(0, num_training_eps)])       # train 1000 randomly generated instances of sizes 1 - 100, smallest to largest
    for size in training_sizes:
        l = [random.randint(0, val_max) for i in range(0, size)]
        ep_t = g.learning_episode(l)

    g.compute_policy(learn_max_size)       # pre-compute optimal policy

    learned_results = [0 for i in range(exec_max_size)]
    ins_results = [0 for i in range(exec_max_size)]
    q_results = [0 for i in range(exec_max_size)]

    for i in range(0, num_exec_eps):
        for size in range(1, exec_max_size+1):
            l = [random.randint(0, val_max) for j in range(0, size)]

            ins_l = list(l)
            orig_l = list(ins_l)
            start = time.clock()
            insertion_sort(ins_l, 0, size-1)
            end = time.clock()
            ins_results[size-1] += end - start
            if sorted(orig_l) != ins_l:
                assert False

            q_l = list(l)
            orig_l = list(q_l)
            start = time.clock()
            quicksort(q_l)
            end = time.clock()
            q_results[size-1] += end - start
            if sorted(orig_l) != q_l:
                assert False

            learned_l = list(l)
            orig_l = list(learned_l)
            start = time.clock()
            g.episode(learned_l, 0, size-1)
            end = time.clock()
            learned_results[size-1] += end - start
            if sorted(orig_l) != learned_l:
                assert False

    for i in range(0, 101):
        ins_results[i] /= 100
        q_results[i] /= 100
        learned_results[i] /= 100

    plt.plot(range(1, 101), ins_results, label="insertion")
    plt.plot(range(1, 101), q_results, label="quick")
    plt.plot(range(1, 101), learned_results, label="learned")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__" :
    main()
