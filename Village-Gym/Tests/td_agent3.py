#!/usr/bin/env python
import os
import sys
import random
import time
import logging
import json
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from copy import deepcopy

import pandas as pd
import click
from tqdm import tqdm as _tqdm
tqdm = _tqdm

import gym
import gym_village
#from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
#    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD

WEEKLY_COST = -25
NO_REWARD = 0
WIN_REWARD = 1
LOSE_REWARD = -1
DEFAULT_VALUE = 0
EPISODE_CNT = 10000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = 'best_td_agent3.dat'
EPSILON = 0.025
ALPHA = 0.03
CWD = os.path.dirname(os.path.abspath(__file__))
LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')

ft_values = defaultdict(float)
ft_visits = defaultdict(lambda: 0)

def reset_feature_values():
    global ft_values, ft_visits
    ft_values = defaultdict(float)
    ft_visits = defaultdict(lambda: 0)

def set_feature_value(feature, value):
    ft_visits[tuple(feature)] += 1
    ft_values[tuple(feature)] = value

def best_val_indices(values, fn):
    best = fn(values)
    return [i for i, v in enumerate(values) if v == best]

############################################################
# Set log level by verbosity level.
############################################################
def set_log_level_by(verbosity):
    """Set log level by verbosity level.
    verbosity vs log level:
        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG
    Args:
        verbosity (int): Verbosity level given by CLI option.
    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level
############################################################
# Execute an action and returns resulted state
############################################################
def after_action_state(state, info, action):
    """Execute an action and returns resulted state.
    Args:
        state (tuple): Board status + mark
        action (int): Action to run
    Returns:
        tuple: New state
    """

    nstate = list(state[:])
    ninfo = list(info[:])
    
    nstate[0] += 1
    ninfo[0] += WEEKLY_COST
    # Action 1: Plant Corn
    if action == 1:
        # It is only possible in the first 3 weeks
        # And if there are available laborDays
        if nstate[0] <= 3 and ninfo[1] >= 25:
            ninfo[1] -= 25
            nstate[1] += 1
    elif action == 2:
        # It is only possible in [6,8] week interval
        # And if there are available laborDays
        # And if there is planted corn
        if nstate[0] >= 6 and nstate[0] <= 8 and nstate[1] > 0 and ninfo[1] >= 5:
            ninfo[1] -= 5
            nstate[1] -= 1
            nstate[2] += 1
    elif action == 3:
        # It is only possible in [9,13] week interval
        # And if there is harvested corn
        if nstate[0] >= 9 and nstate[0] <= 13 and nstate[2] > 0:
            nstate[2] -= 1
            priceCorn = _sellCorn(nstate[0])
            ninfo[0] += priceCorn
    # Money Value
    if ninfo[0] >= 100:
        nstate[3] = 3
    elif ninfo[0] > 0 and ninfo[0] < 100:
        nstate[3] = 2
    elif ninfo[0] <= 0:
        nstate[3] = 1
    # Labor Days Value
    if ninfo[1] >= 100:
        nstate[4] = 3
    elif ninfo[1] > 0 and ninfo[1] < 100:
        nstate[4] = 2
    elif ninfo[1] <= 0:
        nstate[4] = 1
    
    nstate = tuple(nstate)
    ninfo = tuple(ninfo)
    return nstate, ninfo
############################################################
# Defines the price of Corn
############################################################
def _sellCorn(week):
    values = []
    prob = []
    if week == 9:
        values = [50,100]
        prob = [0.5,0.5]
        return random.choices(values, prob)[0]
    elif week == 10:
        values = [60,100]
        prob = [0.4,0.6]
        return random.choices(values, prob)[0]
    elif week == 11:
        values = [70,100]
        prob = [0.3,0.7]
        return random.choices(values, prob)[0]
    elif week == 12:
        values = [80,100]
        prob = [0.2,0.8]
        return random.choices(values, prob)[0]
    elif week == 13:
        values = [90,100]
        prob = [0.05,0.95]
        return random.choices(values, prob)[0]
    return 100
    
class TDAgent(object):
    def __init__(self, epsilon, alpha):
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_rate = 1.0
        self.weights = defaultdict(float)

    def act(self, state, info, ava_actions):
        return self.egreedy_policy(state, info, ava_actions)

    def egreedy_policy(self, state, info, ava_actions):
        """Returns action by Epsilon greedy policy.

        Return random action with epsilon probability or best action.

        Args:
            state: state
            ava_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        logging.debug("egreedy_policy")
        e = random.random()
        #logging.debug("e {}, rate: {}, compare: {}".format(e,self.episode_rate, self.epsilon * ( 1 - self.episode_rate)))
        #print(self.epsilon / ( self.episode_rate))
        if e < self.epsilon / ( 0.1 *  self.episode_rate ):
            #print('explore')
            logging.debug("Explore with eps {}".format(self.epsilon))
            action = self.random_action(ava_actions)
        else:
            logging.debug("Exploit with eps {}".format(self.epsilon))
            action = self.greedy_action(state, info, ava_actions)
        return action

    def random_action(self, ava_actions):
        return random.choice(ava_actions)

    def greedy_action(self, state, info, ava_actions):
        """Return best action by current state value.
        
        Evaluate each action, select best one. Tie-breaking is random.
        
        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions
        
        Returns:
            int: Selected action
        """
        assert len(ava_actions) > 0
        
        ava_values = []
        for action in ava_actions:
            nstate, ninfo = after_action_state(state, info, action)
            nval = self.getQ(nstate, ninfo, action)
            ava_values.append(nval)
            vcnt = ft_visits[(nstate,action)]
            logging.debug("  nstate {} val {:0.2f} visits {}".
                          format(nstate, nval, vcnt))
        
        indices = best_val_indices(ava_values, max)
        
        # tie breaking by random choice
        aidx = random.choice(indices)
        logging.debug("greedy_action ava_values {} indices {} aidx {}".
                      format(ava_values, indices, aidx))

        action = ava_actions[aidx]

        return action
    
    def identityFeatureExtractor(self, state, info, action):
        featureKey = (tuple(state), action)
        featureValue = 1
        return [(featureKey, featureValue)]
    
    def getQ(self, state, info, action):
        """Return the Q function associated with the weights and features.
        Args:
            state, action (tuple): State, Action.
        Returns:
            float: Value of a state.
        """
        score = 0.0
        
        for f, v in self.identityFeatureExtractor(state, info, action):
            score += self.weights[f] * v
        return score

    def backup(self, state, info, nstate, ninfo, action, reward):
        """Backup value by difference and step size.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        logging.debug("backup state {} nstate {} reward {}".
                      format(state, nstate, reward))

        prediction = self.getQ(state, info, action)
        QoptSprime = self.getQ(nstate, ninfo, action)
        target = reward + 0.95*QoptSprime
        scalar = self.alpha*(prediction - target)
        #print(scalar)
        for f, v in list(self.identityFeatureExtractor(state, info, action)):
            self.weights[f] = self.weights[f] - v * scalar
            set_feature_value(f, self.weights[f])
        logging.debug("  value from {:0.2f} to {:0.2f}".format(prediction, target))

@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):
    global tqdm

    set_log_level_by(verbose)
    if verbose > 0:
        tqdm = lambda x: x  # NOQA

@cli.command(help="Learn and save the model.")
@click.option('-p', '--episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--save-file', default=MODEL_FILE, show_default=True,
              help="Save model data as file name.")
def learn(max_episode, epsilon, alpha, save_file):
    _learn(max_episode, epsilon, alpha, save_file)

def _learn(max_episode, epsilon, alpha, save_file):
    """Learn by episodes.

    Make two TD agent, and repeat self play for given episode count.
    Update state values as reward coming from the environment.

    Args:
        max_episode (int): Episode count.
        epsilon (float): Probability of exploration.
        alpha (float): Step size.
        save_file: File name to save result.
    """
    reset_feature_values()

    env = gym.make('village-v0')
    agent = TDAgent(epsilon, alpha)
    values = {}
    sum = 0
    for i in tqdm(range(max_episode)):
        episode = i + 1
        env.show_episode(False, episode)

        # reset agent for new episode
        agent.episode_rate = episode / float(max_episode)

        state, info = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.act(state, info, ava_actions)

            # update (no rendering)
            nstate, reward, done, ninfo = env.step(action)
            agent.backup(state, info, nstate, ninfo, action, reward)

            if done:
                money, ld = env._get_obs();
                sum += money
                if episode%1 == 0:
                    values[episode] = sum/1
                    sum = 0
                env.show_result(False, episode)
                # set terminal state value
                #set_feature_value(state, reward)

            state = nstate
            info = ninfo

    # save states
    with open('tempdata_3_learn.dat', 'wt') as f:
        for k,v in values.items():
            print('episode: {}, value: {}'.format(k,v))
            f.write('episode: {}, value: {}\n'.format(k,v))
     
    ft_values = deepcopy(agent.weights)
    save_model(save_file, max_episode, epsilon, alpha)
    
def save_model(save_file, max_episode, epsilon, alpha):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="td", max_episode=max_episode, epsilon=epsilon,
                    alpha=alpha)
        # write feature values
        f.write('{}\n'.format(json.dumps(info)))
        for feature, value in ft_values.items():
            vcnt = ft_visits[feature]
            f.write('{}\t{:0.3f}\t{}\n'.format(feature, value, vcnt))

def load_model(filename):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            vcnt = eval(elms[2])
            ft_values[state] = val
            ft_visits[state] = vcnt
    return info

@cli.command(help="Play alone.")
@click.option('-f', '--load-file', default=MODEL_FILE, show_default=True,
              help="Load file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def play(load_file, show_number):
    _play(load_file, show_number)

def _play(load_file, show_number):
    """Play with learned model.

    Make TD agent and adversarial agnet to play with.
    Play and switch starting mark when the game finished.
    TD agent behave no exploring action while in play mode.

    Args:
        load_file (str):
        vs_agent (object): Enemy agent of TD agent.
        show_number (bool): Whether show grid number for visual hint.
    """
    load_model(load_file)
    env = gym.make('village-v0')
    td_agent = TDAgent(0, 0)  # prevent exploring
    agent = td_agent
    agent.weights = deepcopy(ft_values)
    reset_feature_values()
    # print(agent.weights)
    for total_episodes in range(1000,1001):
        positive = False
        sum_total = 0
        values = {}
        sum = 0
        for i_episode in range(total_episodes):
            state, info = env.reset()
            done = False
            while not done:
                ava_actions = env.available_actions()
                action = agent.act(state, info, ava_actions)
                #print(action)
                state, reward, done, info = env.step(action)
                #env.render(mode='human')
                if done:
                    money, ld = env._get_obs();
                    sum += money
                    sum_total += money
                    if i_episode%1 == 0:
                        values[i_episode] = sum/1
                        sum = 0
                    env.show_result(True, i_episode)
                    if state[0] == 13:
                        positive = True
                        #env.show_result(True)
                    break
        average = sum_total/total_episodes
        with open('tempdata_3_play.dat', 'wt') as f:
            for k,v in values.items():
                #print('episode: {}, value: {}'.format(k,v))
                f.write('episode: {}, value: {}\n'.format(k,v))
        print('episodes: {}, average: {}'.format(total_episodes, average))
if __name__ == '__main__':
    cli()
