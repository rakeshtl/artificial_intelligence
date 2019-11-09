#!/usr/bin/env python
import random

import gym_tictac4

CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
#from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
#    after_action_state, tomark, next_mark

############################################################
# Return agent by correspondent mark
############################################################
def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent
############################################################
# Return game status by current board status.
############################################################
def check_game_status(board):
    """Return game status by current board status.
    Args:
        board (list): Current board state
    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0
############################################################
# Execute an action and returns resulted state
############################################################
def after_action_state(state, action):
    """Execute an action and returns resulted state.
    Args:
        state (tuple): Board status + mark
        action (int): Action to run
    Returns:
        tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)
############################################################
# Converts a code to mark -> 0:' ', 1:'O', 2:'X'
############################################################
def tomark(code):
    return CODE_MARK_MAP[code]
############################################################
# Return next _mark (to play)
############################################################
def next_mark(mark):
    return 'X' if mark == 'O' else 'O'
############################################################
# Converts a a mark to code -> 'O':1, 'X': 2
############################################################
def tocode(mark):
    return 1 if mark == 'O' else 2

class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)

def play(max_episode=10):
    episode = 0
    start_mark = 'O'
    env = gym.make('tictac4-v0')
    agents = [BaseAgent('O'),
              BaseAgent('X')]

    while episode < max_episode:
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            env.show_turn(True, mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        # rotate start
        start_mark = next_mark(start_mark)
        episode += 1

if __name__ == '__main__':
    play()
