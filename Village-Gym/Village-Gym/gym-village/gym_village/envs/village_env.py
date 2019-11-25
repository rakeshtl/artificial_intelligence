import logging, random
import gym
import configparser
import os
from gym import spaces

############################################################
# Costants of the game
############################################################
NUM_ACTIONS = 4
WEEKLY_COST = -25
NO_REWARD = 0
WIN_REWARD = 1
LOSE_REWARD = -1

LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')
############################################################
# Auxiliary function
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
    
class Village(gym.Env):
    metadata = {'render.modes': ['human']}
    initialValues = {}

    def __init__(self, alpha = 0.02):
        super(Village, self).__init__()
        self.setInitialValues()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Discrete(2)
        self.alpha = alpha
        self.state = []
        self.seed()
        self.reset()
    
    def reset(self):
        self.done = False
        self.money = 250
        self.laborDays = 100
        self.state = self.startState()
        return self.state, self._get_obs()
    
    def setInitialValues(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), './', 'village.ini'))
        self.initialValues = self.configSectionMap(config, "InitialValues")
        print(self.initialValues)
        
    def configSectionMap(self, config, section):
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = config.get(section, option)
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1
        
    def startState(self):
        """Return game status.
        Returns:
            state[0]: week
            state[1]: planted corn
            state[2]: harvested corn
            state[3]: money value
            state[4]: labor days value
        """
        return [0, 0, 0, 3, 3]
    
    def _get_obs(self):
        return self.money, self.laborDays
    
    def isEnd(self, state):
        """Return game status.
        Returns:
            bool:
                False: game in progress
                True: no money left or total weeks == 13
        """
        status = False
        if state[3] <= 0:
            status = True
        elif state[0] == 13:
            status = True
        return status
    
    def step(self, action):
        """Step environment by action.
        Args:
            action (int): 0: Idle, 1: Plant, 2: Harvest, 3: Sell
        Returns:
            state: State
            int: Reward
            bool: Done
            dict: Money, LaborDays
        """
        assert self.action_space.contains(action)
        
        # udpate week - in state
        self.state[0] += 1
        reward = NO_REWARD
        self.money += WEEKLY_COST
        # Action 1: Plant Corn
        if action == 1:
            # It is only possible in the first 3 weeks
            # And if there are available laborDays
            if self.state[0] >= 0 and self.state[0] <= 3 and self.laborDays >= 25:
                self.laborDays -= 25
                self.state[1] += 1
        elif action == 2:
            # It is only possible in [6,8] week interval
            # And if there are available laborDays
            # And if there is planted corn
            if self.state[0] >= 6 and self.state[0] <= 8 and self.state[1] > 0 and self.laborDays >= 5:
                self.laborDays -= 5
                self.state[1] -= 1
                self.state[2] += 1
        elif action == 3:
            # It is only possible in [9,13] week interval
            # And if there is harvested corn
            if self.state[0] >= 9 and self.state[0] <= 13 and self.state[2] > 0:
                self.state[2] -= 1
                priceCorn = self._sellCorn(self.state[0])
                self.money += priceCorn
        # Money Value
        if self.money >= 100:
            self.state[3] = 3
        elif self.money > 0 and self.money < 100:
            self.state[3] = 2
        elif self.money <= 0:
            self.state[3] = 1
        # Labor Days Value
        if self.laborDays >= 100:
            self.state[4] = 3
        elif self.laborDays > 0 and self.laborDays < 100:
            self.state[4] = 2
        elif self.laborDays <= 0:
            self.state[4] = 1
        logging.debug("check_game_status state {} status {}".format(self.state, self.isEnd(self.state)))
        
        if self.isEnd(self.state):
            self.done = True
        
        if self.money <= 0:
            reward = LOSE_REWARD
        elif self.state == 13:
            reward = WIN_REWARD
        
        return tuple(self.state), reward, self.done, self._get_obs()
        
    def _sellCorn(self, week):
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
    
    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_data(print)  # NOQA
            print('')
        else:
            self._show_board(logging.info)
            logging.info('')
    
    def _show_data(self, showfn):
        """Return important info."""
        showfn('Week: {}. Planted Corn: {}, Harvested Corn: {}, Money: {}, LaborDays: {}' \
        .format(self.state[0], self.state[1], self.state[2], self.money, self.laborDays))
    
    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)
    
    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))
    
    def show_result(self, human, episode):
        self._show_result(print if human else logging.info, episode)
    
    def _show_result(self, showfn, episode):
        status = self.isEnd(self.state)
        msg = "Week: {}, Total Money: {}, epsiode: {}".format(self.state[0], self.money, episode)
        #showfn("==== Finished: {} ====".format(msg))
        showfn("{}".format(msg))
        #showfn('')
    
    def available_actions(self):
        """Step environment by action.
        Returns:
            0: Idle
            1: Plant Corn
            2: Harvest Corn
            3: Sell Corn
        """
        return [0, 1, 2, 3]
    