import logging, random
import gym
import configparser
import os
import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym import spaces
from gym.envs.toy_text import discrete
import numpy as np

############################################################
# Costants of the game
############################################################
NUM_ACTIONS = 4
WEEKLY_COST = -5
NO_REWARD = 0
WIN_REWARD = 1
LOSE_REWARD = -1

class Village2(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    initialValues = {}

    def __init__(self):
        num_states = 222768
        num_weeks = 13
        num_plantedcorn = 4
        num_harvestedcorn = 4
        initial_money = 51
        initial_labordays = 21
        initial_state_distrib = np.zeros(num_states)
        num_actions = NUM_ACTIONS
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for week in range(num_weeks):
            for plantedcorn in range(num_plantedcorn):
                for harvestedcorn in range(num_harvestedcorn):
                    for money in range(initial_money):
                        for labordays in range(initial_labordays):
                            state = self.encode(week, plantedcorn, harvestedcorn, money, labordays)
                            # Test end state
                            if money > 0 and week != 12:
                                initial_state_distrib[state] += 1
                            for action in range(num_actions):
                                # defaults
                                new_week = week + 1
                                new_plantedcorn = plantedcorn
                                new_harvestedcorn = harvestedcorn
                                new_money = money + WEEKLY_COST
                                new_labordays = labordays
                                reward = NO_REWARD
                                done = False
                                
                                # Action 0 - idle (do nothing)
                                # Action 1 - plant corn
                                if action == 1:
                                    # It is only possible in the first 3 weeks
                                    # And if there are available laborDays
                                    if new_week >= 0 and new_week <= 3 and new_labordays >= 5:
                                        new_labordays -= 5
                                        new_plantedcorn += 1
                                # Action 2 - harvest corn
                                elif action == 2:
                                    # It is only possible in [6,8] week interval
                                    #And if there are available laborDays
                                    # And if there is planted corn
                                    if new_week >= 6 and new_week <= 8 and new_plantedcorn > 0 and new_labordays >= 1:
                                        new_labordays -= 1
                                        new_plantedcorn -= 1
                                        new_harvestedcorn += 1
                                elif action == 3:
                                    # It is only possible in [8,12] week interval
                                    # And if there is harvested corn
                                    if new_week >= 8 and new_week <= 12 and new_harvestedcorn > 1:
                                        new_harvestedcorn -= 1
                                        priceCorn = self._sellCorn(new_week)
                                        new_money += priceCorn
                                
                                if new_money <= 0:
                                    new_money = 0
                                    done = True
                                elif new_week == 12:
                                    done = True
                                
                                if new_money <= 0:
                                    reward = LOSE_REWARD
                                elif new_week == 12:
                                    reward = new_money
                                new_state = self.encode(
                                    new_week, new_plantedcorn, new_harvestedcorn, new_money, new_labordays)
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)
    
    def reset(self):
        self.s = self.encode(0,0,0,50,20)
        self.lastaction = None
        return self.s
    
    def encode(self, week, plantedcorn, harvestedcorn, money, labordays):
        # (13), 4, 4, 51, 15
        i = week
        i *= 4
        i += plantedcorn
        i *= 4
        i += harvestedcorn
        i *= 51
        i += money
        i *= 21
        i += labordays
        return i

    def decode(self, i):
        out = []
        out.append(i % 21)
        i = i // 21
        out.append(i % 51)
        i = i // 51
        out.append(i % 4)
        i = i // 4
        out.append(i % 4)
        i = i // 4
        out.append(i)
        assert 0 <= i < 13
        return reversed(out)
    
    def _sellCorn(self, week):
        values = []
        prob = []
        if week == 8:
            values = [10,20]
            prob = [0.5,0.5]
            return random.choices(values, prob)[0]
        elif week == 9:
            values = [12,20]
            prob = [0.4,0.6]
            return random.choices(values, prob)[0]
        elif week == 10:
            values = [14,20]
            prob = [0.3,0.7]
            return random.choices(values, prob)[0]
        elif week == 11:
            values = [16,20]
            prob = [0.2,0.8]
            return random.choices(values, prob)[0]
        elif week == 12:
            values = [18,20]
            prob = [0.05,0.95]
            return random.choices(values, prob)[0]
        return 20
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        
        week, plantedcorn, harvestedcorn, money, labordays = self.decode(self.s)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Idle", "Plant Corn", "Harvest Corn", "Sell Corn"][self.lastaction]))
        outfile.write('Week: {}. Planted Corn: {}, Harvested Corn: {}, Money: {}, LaborDays: {}' \
            .format(week, plantedcorn, harvestedcorn, money, labordays))
        outfile.write("\n")
        
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
    