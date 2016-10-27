#!/usr/bin/env python3

from matplotlib import pyplot as plt
import random
import numpy as np
from numpy.linalg import inv
import time
import math

class LearnedSort:
    def __init__(self, epsilon, alpha, gamma):
        self.epsilon = epsilon                                      # epsilon for e-greedy policy
        self.alpha = alpha                                          # step size parameter
        self.gamma = gamma                                          # discount factor
        self.alist = []                                             # list to sort
        self.w_q = np.matrix('0.0 ; 0.0 ; 0.0')                     # weight vector for quicksort
        self.w_i = np.matrix('0.0 ; 0.0 ; 0.0')                     # weight vector for insertionsort
        self.A_mat_q = np.matrix('1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0')     # A matrix for quicksort
        self.b_vec_q = np.matrix('0.0 ; 0.0 ; 0.0')                             # b vector for quicksort
        self.A_mat_i = np.matrix('1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0')     # A matrix for insertionsort
        self.b_vec_i = np.matrix('0.0 ; 0.0 ; 0.0')                             # b matrix for insertionsort
        self.A = ["quick", "insertion"]

    def phi(self, size):
        return np.matrix([[pow(size, 2)], [size * math.log(size, 2)], [size]])

    def get_q(self, size, a):               # Q(n, a) = w_1(a) * n^2 + w_2(a) * n * log_2(n) + w_3(a) * n
        if a == "quick":
            return np.transpose(self.w_q) * self.phi(size)
        else:
            return np.transpose(self.w_q) * self.phi(size)

    def update_weights(self, a):       # w(t) = (A(t))^(-1) * b(t)
        if a == "quick":
            self.w_q = inv(self.A_mat_q) * self.b_vec_q
        else:
            self.w_i = inv(self.A_mat_i) * self.b_vec_i

    def update_A(self, size, a):       # A(t+1) = A(t) + phi(s(t+1), a(t+1)) * transpose(phi(s(t+1), a(t+1)))
        if a == "quick":
            self.A_mat_q += self.phi(size) * np.transpose(self.phi(size))
        else:
            self.A_mat_i += self.phi(size) * np.transpose(self.phi(size))

    def update_b(self, size, a, r):     # b(t+1) = b(t) + phi(s(t+1), a(t+1)) * Q(t+2)(s(t+1), a(t+1))
        if a == "quick":
            self.b_vec_q += self.phi(size) * r
        else:
            self.b_vec_i += self.phi(size) * r

    def quick_sort(self, first, last):  # one iteration of recursive quicksort algorithm
        start = time.clock()
        pivotvalue = self.alist[first]
        leftmark = first + 1
        rightmark = last
        done = False
        while not done:
            while leftmark <= rightmark and self.alist[leftmark] <= pivotvalue:
                leftmark = leftmark + 1
            while self.alist[rightmark] >= pivotvalue and rightmark >= leftmark:
                rightmark = rightmark - 1
            if rightmark < leftmark:
                done = True
            else:
                temp = self.alist[leftmark]
                self.alist[leftmark] = self.alist[rightmark]
                self.alist[rightmark] = temp
        temp = self.alist[first]
        self.alist[first] = self.alist[rightmark]
        self.alist[rightmark] = temp
        end = time.clock()
        time_elapsed = end - start
        return rightmark, time_elapsed

    def insertion_sort(self, first, last):
        start = time.clock()
        for index in range(first+1, last+1):
            currentvalue = self.alist[index]
            position = index
            while position > first and self.alist[position-1] > currentvalue:
                self.alist[position] = self.alist[position-1]
                position = position-1
            self.alist[position] = currentvalue
        end = time.clock()
        time_elapsed = end - start
        return None, time_elapsed

    def choose_a(self, s):                    # choose action according to e-greedy policy               # choose action based on e-greedy policy
        if random.random() < self.epsilon:
            return random.choice(self.A)
        else:
            return self.min_a(s)

    def min_a(self, s):                     # return action value with min corresponding Q val for a given state
        min_val = 99999
        a_choice = None
        for a in self.A:
            val = self.get_q(s, a)
            if val < min_val:
                min_val = val
                a_choice = a
        return a_choice

    def min_q(self, s):                    # return min Q value for a given state
        min_val = 99999
        for a in self.A:
            val = self.get_q(s, a)
            if val < min_val:
                min_val = val
        return min_val

    def take_a(self, s, a):               # take action a and return new partition location and cost
        if a == "quick":
            p, r = self.quick_sort(s[0], s[1])
        else:
            p, r = self.insertion_sort(s[0], s[1])
        return p, r

    def monte_carlo(self, s, total_cost):     # monte-carlo return,  sum of all individual costs when starting with a subproblem corresponding to state s and following greedy policy until the subproblem has been fully solved
        total_cost = 0
        if (s[0] < s[1]):
            cur_size = s[1] - s[0] + 1
            a = self.min_a(cur_size)
            p, r = self.take_a(s, a)
            total_cost += r
            if p is None:                    # then insertionsort was used, and it is fully sorted
                return total_cost
            else:                            # then quicksort was used, and need to keep sorting 2 new subproblems
                self.monte_carlo([s[0], p-1], total_cost)
                self.monte_carlo([p+1, s[1]], total_cost)
        return total_cost

    def q_learning_episode(self, alist):    # given a list to sort, sort based on Q-learning method
        self.alist = alist
        t = 0                               # time to complete sorting
        finished = False
        s = [0, len(self.alist)-1]          # current partition
        while not finished:
            cur_size = s[1] - s[0] + 1      # current size of partition
            if (s[0] < s[1]):
                a = self.choose_a(cur_size)
                p, r = self.take_a(s, a)
                t += r
                if p is None:               # insertion sort was used, we are done sorting
                    q_val = (1 - self.alpha) * self.get_q(cur_size, a) + self.alpha * r
                    finished = True
                else:                        # run monte carlo algorithm on smaller partition
                    if (p - s[0] < s[1] - p):
                        m_val = self.monte_carlo([s[0], p-1], 0)        # monte-carlo return
                        s[0] = p + 1
                    else:
                        m_val = self.monte_carlo([p+1, s[1]], 0)        # monte-carlo return
                        s[1] = p-1
                    t += m_val
                    new_size = s[1] - s[0] + 1
                    q_val = (1 - self.alpha) * self.get_q(cur_size, a) + self.alpha * (r + m_val + self.min_q(new_size))
                self.update_A(cur_size, a)
                self.update_b(cur_size, a, q_val)
                self.update_weights(a)
            else:
                finished = True
        if sorted(self.alist) != self.alist:
            assert False
        return t

def insertion_sort(alist):
    start = time.clock()
    for index in range(1, len(alist)):
        currentvalue = alist[index]
        position = index
        while position > 0 and alist[position - 1] > currentvalue:
            alist[position] = alist[position - 1]
            position = position - 1
        alist[position] = currentvalue
    end = time.clock()
    time_elapsed = end - start
    if sorted(alist) != alist:
        assert False
    return time_elapsed

def quicksort(alist):
    start = time.clock()
    quicksort_helper(alist, 0, len(alist) - 1)
    end = time.clock()
    time_elapsed = end - start
    if sorted(alist) != alist:
        assert False
    return time_elapsed

def quicksort_helper(alist,first,last):
    if first < last:
        splitpoint = partition(alist, first, last)
        quicksort_helper(alist, first, splitpoint - 1)
        quicksort_helper(alist, splitpoint + 1, last)


def partition(alist,first,last):
    pivotvalue = alist[first]
    leftmark = first+1
    rightmark = last
    done = False
    while not done:
        while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
            leftmark = leftmark + 1
        while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
            rightmark = rightmark -1
        if rightmark < leftmark:
            done = True
        else:
            temp = alist[leftmark]
            alist[leftmark] = alist[rightmark]
            alist[rightmark] = temp
    temp = alist[first]
    alist[first] = alist[rightmark]
    alist[rightmark] = temp
    return rightmark

def main():
    g = LearnedSort(0.6, 1.0, 0.75)
    training_sizes = [random.randint(1, 10) for i in range(0, 20)]       # train 20 randomly generated instances of sizes 1 - 10, smaller first
    training_sizes = sorted(training_sizes)
    for item in training_sizes:
        alist = [random.randint(0, 100) for i in range(0, item)]
        ep_t = g.q_learning_episode(alist)
    training_sizes = [random.randint(10, 100) for i in range(0, 20)]    # train 20 randomly generated instances of sizes 10 - 100, smaller first
    training_sizes = sorted(training_sizes)
    for item in training_sizes:
        alist = [random.randint(0, 100) for i in range(0, item)]
        ep_t = g.q_learning_episode(alist)
    learned_results = [0 for i in range(101)]
    ins_results = [0 for i in range(101)]
    q_results = [0 for i in range(101)]
    g.epsilon = 0.0                                                     # follow greedy policy after training
    for i in range(0, 101):
        alist = [random.randint(0, 100) for i in range(0, i)]           # test for sizes 1 - 100
        learned_alist = list(alist)
        ins_alist = list(alist)
        q_alist = list(alist)
        learned_results[i] = g.q_learning_episode(learned_alist)
        ins_results[i] = insertion_sort(ins_alist)
        q_results[i] = quicksort(q_alist)
    plt.plot(range(0, 101), ins_results, label="insertion")
    plt.plot(range(0, 101), q_results, label="quick")
    plt.plot(range(0, 101), learned_results, label="learned")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__" :
    main()
