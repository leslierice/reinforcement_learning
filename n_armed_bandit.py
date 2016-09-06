#!/usr/bin/env python3

# Leslie Rice
# Programming Assignment 1
# Multi-arm Bandits

from operator import attrgetter
from matplotlib import pyplot as plt
from enum import Enum
import random
import numpy
import math


class Version(Enum):
    egreedy, ucb, gradient = range(3)


class Action:
    def __init__(self):
        self.value = numpy.random.normal(0, 1)          # select q*(a) according to normal distribution with mean 0, variance 1
        self.num_selections = 0                         # number of times this action has been selected
        self.estimate = 0                               # estimated action value Qn
        self.preference = 0                             # preference Ht(a)
        self.direction = random.randint(0, 1)

    def get_reward(self):
        return numpy.random.normal(self.value, 1)       # reward selected from normal distribution with mean q*(a), variance 1

    def play_stationary(self):                          # use non-constant step-size parameter for stationary environment
        reward_received = self.get_reward()
        self.num_selections += 1
        if self.num_selections == 1:
            self.estimate = reward_received
        else:
            self.estimate = self.estimate + (1 / self.num_selections) * (reward_received - self.estimate)    # Q(n+1) = Qn + (1 / n) [Rn - Qn]
        return reward_received

    def play_nonstationary(self, alpha):                # use constant step-size parameter for non-stationary environment
        reward_received = self.get_reward()
        self.num_selections += 1
        self.estimate = self.estimate + alpha * (reward_received - self.estimate)
        return reward_received

    def update_preference(self, p, reward, baseline):
        self.preference = self.preference + 0.1 * (reward - baseline) * p    # alpha = 0.1

    def change_action_value(self):
        self.value = numpy.random.normal(0, 1)

    def reset_initial(self, estimate):
        self.estimate = estimate

    def reset(self):
        self.num_selections = 0
        self.estimate = 0
        self.preference = 0


class Bandit:
    def __init__(self, num_arms):
        self.actions = [Action() for i in range(num_arms)]
        self.baseline = 0
        self.num_rewards = 0

    def set_initial(self, estimate):
        for a in self.actions:
            a.estimate = estimate

    def choose_greedy(self, alpha=None):
        action_taken = max(self.actions, key=attrgetter('estimate'))    # select action with highest estimated action value
        if alpha:
            reward = action_taken.play_nonstationary(alpha)
        else:
            reward = action_taken.play_stationary()
        optimal = self.determine_optimality(action_taken, reward)
        return reward, optimal

    def choose_random(self, alpha):
        action_taken = random.choice(self.actions)          # select randomly from all actions
        if alpha:
            reward = action_taken.play_nonstationary(alpha)
        else:
            reward = action_taken.play_stationary()
        optimal = self.determine_optimality(action_taken, reward)

        return reward, optimal

    def choose_ucb(self, alpha):
        self.num_rewards += 1                              # increment total number of rewards
        argmax = -100
        action_taken = None
        for a in self.actions:
            arg = a.estimate + 2 * math.sqrt(math.log(self.num_rewards) / a.num_selections) if a.num_selections else float("inf")     # c = 2
            if arg > argmax:
                argmax = arg
                action_taken = a
        if not action_taken:
            print("Error.")
        if alpha:
            reward = action_taken.play_nonstationary(alpha)
        else:
            reward = action_taken.play_stationary()
        optimal = self.determine_optimality(action_taken, reward)
        return reward, optimal

    def choose_gradient(self, alpha):
        self.num_rewards += 1                               # increment total number of rewards
        grad_sum = 0
        for a in self.actions:
            grad_sum += math.exp(a.preference)
        select_map = {}
        start = 0
        for a in self.actions:
            p = (math.exp(a.preference)) / grad_sum         # probability of taking action a
            k = (start, p + start)
            select_map[k] = a
            start += p
        r = random.random()
        for k in select_map.keys():
            if r >= k[0] and r < k[1]:
                action_taken = select_map[k]
                if alpha:
                    reward = action_taken.play_nonstationary(alpha)
                else:
                    reward = action_taken.play_stationary()
                break
        if reward is None:
            print("Error.")
        optimal = self.determine_optimality(action_taken, reward)
        self.baseline += (1 / self.num_rewards) * (reward - self.baseline)      # compute baseline incrementally
        for a in self.actions:                                                  # update preferences
            if a is action_taken:
                a.update_preference(1-(math.exp(a.preference)/grad_sum), reward, self.baseline)
            else:
                a.update_preference(math.exp(a.preference)/grad_sum, reward, self.baseline)
        return reward, int(optimal)

    def determine_optimality(self, action_taken, reward_received):              # determine if the action taken was the optimal action
        optimal = True
        for a in self.actions:
            if a is not action_taken:
                reward = a.get_reward()
                if reward > reward_received:
                    optimal = False
                    break
        return optimal

    def reset(self):
        for a in self.actions:
            a.reset()
        self.baseline = 0
        self.num_rewards = 0


class Game:
    def __init__(self, num_arms, num_tasks):
        self.bandits = [Bandit(num_arms) for i in range(num_tasks)]
        self.num_tasks = num_tasks

    def play(self, num_plays, initial_estimate, version, e=None, alpha=None):
        for b in self.bandits:                          # set initial action values
            b.set_initial(initial_estimate)

        average_rewards = []
        percents_optimal = []

        for i in range(num_plays):                      # play game num_plays times
            sum_rewards = 0
            sum_opt_actions = 0
            for b in self.bandits:                      # play with given method
                if version is Version.egreedy:
                    if random.random() < e:             # choose randomly with probability e
                        reward, optimal = b.choose_random(alpha)
                    else:
                        reward, optimal = b.choose_greedy(alpha)
                elif version is Version.ucb:
                    reward, optimal = b.choose_ucb(alpha)
                elif version is Version.gradient:
                    reward, optimal = b.choose_gradient(alpha)
                else:
                    print("Version not implemented.")

                sum_rewards += reward
                sum_opt_actions += int(optimal)

                if (alpha):                              # if non-stationary, q*(a) take independent random walks
                    for a in b.actions:
                        if (random.random() < 0.1):     # probability of walk = 0.1
                            if a.direction == 1:
                                if a.value + 0.25 < 2.5:
                                    a.value += 0.25
                            else:
                                if a.value - 0.25 >= -2.5:
                                    a.value -= 0.25
                            # a.change_action_value()

            avg_reward = sum_rewards / self.num_tasks               # average reward for bandit
            average_rewards.append(avg_reward)
            percent_optimal = sum_opt_actions / self.num_tasks      # % optimal actions for bandit
            percents_optimal.append(percent_optimal)

        for b in self.bandits:
            b.reset()

        return average_rewards, percents_optimal


def main():
    game = Game(10, 2000)

    #stationary tests

    greedy0r, greedy0o = game.play(500, 0, Version.egreedy, e=0)          # greedy, Q1(a) = 0, stationary
    greedy1r, greedy1o = game.play(500, 1, Version.egreedy, e=0)          # greedy, Q1(a) = 1, stationary
    greedy5r, greedy5o = game.play(500, 5, Version.egreedy, e=0)          # greedy, Q1(a) = 5, stationary
    greedy10r, greedy10o = game.play(500, 10, Version.egreedy, e=0)       # greedy, Q1(a) = 10, stationary

    plt.plot(range(1, 501), greedy0r, '#00688B', label='Q1(a)=0')
    plt.plot(range(1, 501), greedy1r, '#4B0082', label='Q1(a)=1')
    plt.plot(range(1, 501), greedy5r, '#EE7600', label='Q1(a)=5')
    plt.plot(range(1, 501), greedy10r, '#458B00', label='Q1(a)=10')
    plt.xticks([0, 250, 500])
    plt.yticks([0, 0.5, 1, 1.5])
    plt.legend(loc='lower right')
    plt.xlabel('Plays')
    plt.ylabel('Average reward')
    plt.title('Greedy, stationary')
    plt.show()

    egreedy0r, egreedy0o = game.play(500, 0, Version.egreedy, e=0.1)          # egreedy, e = 0.1, Q1(a) = 0, stationary
    egreedy1r, egreedy1o = game.play(500, 1, Version.egreedy, e=0.1)          # egreedy, e = 0.1, Q1(a) = 1, stationary
    egreedy5r, egreedy5o = game.play(500, 5, Version.egreedy, e=0.1)          # egreedy, e = 0.1, Q1(a) = 5, stationary
    egreedy10r, egreedy10o = game.play(500, 10, Version.egreedy, e=0.1)       # egreedy, e = 0.1, Q1(a) = 10, stationary

    plt.plot(range(1, 501), egreedy0r, '#00688B', label='Q1(a)=0')
    plt.plot(range(1, 501), egreedy1r, '#4B0082', label='Q1(a)=1')
    plt.plot(range(1, 501), egreedy5r, '#EE7600', label='Q1(a)=5')
    plt.plot(range(1, 501), egreedy10r, '#458B00', label='Q1(a)=10')
    plt.xticks([0, 250, 500])
    plt.yticks([0, 0.5, 1, 1.5])
    plt.legend(loc='lower right')
    plt.xlabel('Plays')
    plt.ylabel('Average reward')
    plt.title('E-greedy, e = 0.1, stationary')
    plt.show()


    # non-stationary tests

    greedy5r, greedy5o = game.play(1000, 5, Version.egreedy, e=0, alpha=0.1)       # greedy, Q1(a) = 5
    egreedy0r, egreedy0o = game.play(1000, 5, Version.egreedy, e=0.1, alpha=0.1)   # e-greedy with e = 0.1, Q1(a) = 5
    ucb0r, ucb0o = game.play(1000, 0, Version.ucb, alpha=0.1)                      # UCB with c = 2, Q1(a) = 0
    gradient0r, gradient0o = game.play(1000, 0, Version.gradient, alpha=0.1)       # gradient with alpha = 0.1, Q1(a) = 0

    plt.plot(range(1, 1001), greedy5r, '#0099CC', label='Greedy, Q1(a)=5')
    plt.plot(range(1, 1001), egreedy0r, '#458B00', label='E-greedy, Q1(a)=5')
    plt.plot(range(1, 1001), ucb0r, '#4B0082', label='UCB, Q1(a)=0')
    plt.plot(range(1, 1001), gradient0r, '#EE7600', label='Gradient, Q1(a)=0')
    plt.xticks([0, 250, 500, 750, 1000])
    plt.yticks([0, 0.5, 1, 1.5])
    plt.legend(loc='lower right')
    plt.xlabel('Plays')
    plt.ylabel('Average reward')
    # plt.title('Non-stationary, step-size parameter = 0.1, random walks')
    plt.title('Non-stationary, step-size parameter = 0.1, directed walks')
    plt.show()


if __name__ == "__main__" :
    main()
