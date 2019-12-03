
# coding: utf-8

# **DQN_Village_Agent_Game_Single_Crop**

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import numpy as np
import pandas as pd
from random import random as rand

import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


# In[3]:


# use variables CORN , BEAN , COTTON 
def get_data(period):
     # returns a T x 3 list of market prices
     # each row is a different step 
     # 0 = Hold
     # 1 = Plant
     # 2 = Sell
    df=pd.DataFrame(columns=["CORN","BEAN","COTTON"])

    corn_base_price = 20.0
    bean_base_price = 25.0
    cotton_base_price = 30.0
    df["CORN"] =np.random.normal(60.0, 20.0, period)
    df["BEAN"] =np.random.normal(70.0, 25.0, period)
    df["COTTON"] =np.random.normal(100.0, 30.0, period)    
    df["CORN"].where(df["CORN"]<=corn_base_price, corn_base_price)
    df['BEAN'].where(df['BEAN']<=bean_base_price, bean_base_price)
    df['COTTON'].where(df['COTTON']<=cotton_base_price, cotton_base_price)   
    return df.values


# In[4]:


def get_scaler(env):
# return scikit-learn scaler object to scale the states
# Note: you could also populate the replay buffer here
    states = []
    keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    values = [1,1,1,0,0,2,2,2,3,3,3,3,3,1,1,1,0,0,2,2,2,3,3,3,3,3]
    time_action = dict(zip(keys, values))    
    for step in range(env.n_step):  
        action = time_action[step]            
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
#     print("===============states (statem reward, done, info)=========================")
#     for c in states:
#         print(c)    
    scaler = StandardScaler()
    scaler.fit(states)     
    return scaler 


# In[5]:


# to store trained model with result
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[6]:


class DQLModel:
    """ Neural Network Model with Stochastic Gradient Descent """
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []
    
    #Predict function to predict the state Q value
    def predict(self, X):
        # make sure X is N x D
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b
    
    #Stochastic Gradient Descent
    def sgd(self, X, Y, learning_rate=0.0001, momentum=0.99):
        # make sure dimension of X is N x D
        assert(len(X.shape) == 2)

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K
        num_values = np.prod(Y.shape)    
        
        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values          
        
        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb       
        

        # update params
        self.W += self.vW
        self.b += self.vb

#         print("weight : ", self.W)       
        
        mse = np.mean((Yhat - Y)**2)        
        self.losses.append(mse)
        
        return mse
    
    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


# In[7]:


action_list = list(map(list, itertools.product([0, 1, 2, 3], repeat = 1)))
action_list


# In[8]:


class VillageEnv:
    """
    A 3-item village gaming environment.
    𝑠𝑡𝑎𝑡𝑒  = [𝑚𝑜𝑛𝑒𝑦 𝑣𝑎𝑙𝑢𝑒, 𝑙𝑎𝑏𝑜𝑟 𝑢𝑛𝑖𝑡𝑠, 𝑤𝑒𝑒𝑘, 𝑐𝑜𝑟𝑛 𝑝𝑟𝑖𝑐𝑒, 𝑝𝑙𝑎𝑛𝑡𝑒𝑑 𝑐𝑜𝑟𝑛, ℎ𝑎𝑟𝑣𝑒𝑠𝑡𝑒𝑑 𝑐𝑜𝑟𝑛, 𝑠𝑡𝑜𝑟𝑒𝑑 𝑐𝑜𝑟𝑛]

    State: vector of size 7 (1 * 5 + 2)
        - # 𝑝𝑙𝑎𝑛𝑡𝑒𝑑_c𝑜𝑟𝑛 - planted corn
        - # harvested_𝑐𝑜𝑟n - harvested corn      
        - 𝑐𝑜𝑟𝑛_𝑝𝑟𝑖𝑐𝑒: market price of the corn         
        - Money Value (can be used to purchase more labor units and pay the family expenses)
        - 𝑙𝑎𝑏𝑜𝑟_𝑢𝑛𝑖𝑡𝑠 (can be used to plant & harves crops)
    Action: categorical variable with 4 possibilites
        - for each stock, you can:
        - 0 = Hold  (do nothing)
        - 1 = Plant
        - 2 = Harvest
        - 3 = Sell       
    """
    def __init__(self, number_of_weeks, number_of_crops, initial_investment=1000, initial_labor_unit=52, initial_price=0):
        # data
#         self.market_price_time_series = data
        self.n_step, self.n_crop = number_of_weeks, number_of_crops

        # instance attributes
        self.initial_investment = initial_investment
        self.initial_labor_unit = initial_labor_unit
        self.market_price = initial_price  
        self.cur_step = None
        self.crop_in_barn = None
        self.crop_planted = None
#         self.crop_harvested = None             
        self.cash_in_hand = None
        self.labor_in_hand = None

        self.action_space = np.arange(3**number_of_crops +1)
        # action permutations
        # returns a nested list with elements like:
        # [0]        
        # 0 = hold
        # 1 = plant
        # 2 = harvest
        # 3 = sell
        self.action_list = list(map(list, itertools.product([0, 1, 2, 3], repeat = self.n_crop)))       
        # calculate size of state
        self.state_dim = self.n_crop * 5 + 2
        self.reset()
    
    def reset(self):
        self.cur_step = 0
        self.crop_planted = np.zeros(self.n_crop)
#         self.crop_harvested = np.zeros(self.n_crop)          
        self.crop_in_barn = np.zeros(self.n_crop)        
        self.cash_in_hand = self.initial_investment
        self.labor_in_hand = self.initial_labor_unit
        self.week          = self.cur_step 
        
        return self._get_obs()
    
    def _get_obs(self):        
        obs                              = np.empty(self.state_dim)        
        obs[:self.n_crop]                = self.crop_planted        
        obs[self.n_crop:2*self.n_crop]   = self.crop_in_barn       
        obs[2*self.n_crop:3*self.n_crop] = self.market_price
#         obs[3*self.n_crop:4*self.n_crop] = self.market_price
         
        obs[-3] = self.week
        obs[-2] = self.labor_in_hand
        obs[-1] = self.cash_in_hand

        return obs
    
    
    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()
#         print("Action...:", action,"    prev_val...", prev_val)

        # update price, i.e. go to the next day
        self.cur_step += 1  
        self._get_market_price()        
        
        # perform the action
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val
        

        # done if we have reached out the end of the time series step or money value is zero
        done = (self.cur_step == self.n_step)|(self.cash_in_hand < 0)
        
#         print("number of step:", self.n_step,"current step:", self.cur_step,"cash_in_hand:",self.cash_in_hand, " done", done )

        # store the current value of the portfolio here
        info = {'cur_val': cur_val, 'cash_in_hand': self.cash_in_hand, "cur_step":self.cur_step}

        # conform to the Gym API        
        return self._get_obs(), reward, done, info
    
    # get market price of the crop
    def _get_market_price(self):
        r = rand()
        if self.cur_step == 9:             
            self.market_price =  50 if r < 0.5 else 100            
        elif self.cur_step == 10: 
            self.market_price =  60 if r < 0.4 else 100 
        elif self.cur_step  == 11:
            self.market_price =  70 if r < 0.3 else 100 
        elif self.cur_step  == 12:
            self.market_price =  80 if r < 0.2 else 100 
        elif self.cur_step  == 13:
            self.market_price =  90 if r < 0.1 else 100 
        else:
            self.market_price = 0      
 
    
    #Get the q value of the state
    def _get_val(self):
        self.planted_crop_value = [50] 
        self.barn_crop_value = [80] 
        wealth = (self.crop_planted.dot(self.planted_crop_value) +                  
                 self.crop_in_barn.dot(self.barn_crop_value) + self.cash_in_hand)        
        return wealth
    
    # perform the action
    def _trade(self, action):
        # index the action we want to perform
        # 0 = hold
        # 1 = plant
        # 2 = harvest
        # 3 = sell       
        self.money_val_to_plant = 25
        self.money_val_to_harvest = 5
        action_vec = self.action_list[action]
       
        # determine which stocks to buy or sell
        sell_index = [] # stores index of crops we want to sell
        plant_index = [] # stores index of crops we want to plant
        
        self.cash_in_hand -= 25    #Weekly expense for family   
 
        # Take action PLANT
        if action_vec[0] == 1:            
            # PLANT ACTION: Choose number of crops to plant            
            unit_to_plant = np.random.choice(3)+1             
           
            for i in range(unit_to_plant):                    
                    if self.labor_in_hand > 40:
                        self.crop_planted[0] += 1
                        self.labor_in_hand -= 25
                        unit_to_plant -=1                                          
                    else:
                        break    
        
        # Take action HARVEST    
        if action_vec[0] == 2:            
            unit_to_harvest = int(self.crop_planted[0])
            # PLANT ACTION: Choose number of crops to plant            
            for i in range(unit_to_harvest): 
                if self.labor_in_hand > 5:                   
                    self.crop_in_barn[0] += 1
                    self.labor_in_hand -= 5
                    self.crop_planted[0] -= 1                
                else:
                    break               

        # Take action SELL
        if action_vec[0] == 3:            
            if self.crop_in_barn[0] > 0.0:
                if int(self.cur_step) == 9 or int(self.cur_step) == 22: 
#                     print("sell at step",self.cur_step,"....")                    
                    self.crop_in_barn -= 1
                    self.cash_in_hand += self.market_price
#                     print("step+++++: ", self.cur_step, "market_price: ", self.market_price, "cash in hand: ",self.cash_in_hand)
                else:
                    r = rand()
#                     print("sell at step",self.cur_step,"===")
                    threshold = (self.cur_step/12.0) if (self.cur_step < 13 and self.cur_step > 9) else (self.cur_step/25.0)
                    if r < threshold:                        
                        self.crop_in_barn -= 1
                        self.cash_in_hand += self.market_price
#             print("current step: ", self.cur_step,"  crop in barn:", self.crop_in_barn[0], "  cash in hand:", self.cash_in_hand)     
            
            


# In[9]:


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQLModel(state_size, action_size)
    
    def act(self, state,step):        
        if np.random.rand() <= self.epsilon:
            keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
            values = [1,1,1,0,0,2,2,2,3,3,3,3,3,1,1,1,0,0,2,2,2,3,3,3,3,3]
            time_action = dict(zip(keys, values))           
            action = time_action[step]
#             print("step",step,"action@@@@@@: ",action, "rand_action:", np.random.choice(self.action_size-2) )
#             return np.random.choice(self.action_size-2)
            return action
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])  # returns action
    
    #train the network using back probagation & SGD optimizer with momentum term 
    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.argmax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target
        
        
#       print("target_full:", target_full)
        # Run one training step
        error = self.model.sgd(state, target_full)       
        
        
#         print("  action: ",action, "  reward:", reward,  " error: ", error)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    
    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)


# In[10]:


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False
    step = 0
    while not done:        
        action = agent.act(state,step)
        next_state, reward, done, info = env.step(action)
        step = info["cur_step"]+1
#         print("step:",info["cur_step"], "done:", done, "cash in hand",info['cash_in_hand'])
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val'], info['cash_in_hand']


# In[11]:


# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(x, xlabel, ylabel, title):
    # matplotlib histogram
    plt.hist(x, color = 'blue', edgecolor = 'black',
             bins = 10)

    # seaborn histogram
    sns.distplot(x, hist=True, kde=False, 
                 bins=10, color = 'blue',
                 hist_kws={'edgecolor':'black'})
    # Add labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# In[12]:


if __name__ == '__main__':
    
    # configuration
    np.random.seed(seed=12)
    models_folder = 'DQN_Village_Agent_models'
    rewards_folder = 'DQN_Village_Agent_rewards'
    num_episodes = 1000
    # num_episodes = 20000
    batch_size = 32
    initial_investment = 225
    initial_labor_unit = 100
    mode = 'test'
    number_of_weeks = 13
    number_of_crops = 1

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    
    n_timesteps, n_stocks = number_of_weeks, number_of_crops
    
    
    # Create train and test data of market prices
    n_train = n_timesteps
#     n_train = n_timesteps // 2
#     train_data = data[:n_train]
#     test_data = data[n_train:]
    
    #create environment object
    env = VillageEnv(number_of_weeks, number_of_crops, initial_investment,initial_labor_unit)    
    state_size = env.state_dim
    action_size = len(env.action_space)
    print("***********State size {}, Action Size {}".format(state_size,action_size))


    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    money_value = []
    episode = []
    money_50 = []
    episode_50 = []

    if mode == 'train':
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = VillageEnv(number_of_weeks, number_of_crops, initial_investment,initial_labor_unit)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val, cash = play_one_episode(agent, env, mode)
        dt = datetime.now() - t0
        if (e+1)%50 == 0:
#           print(f"episode: {e + 1}/{num_episodes}, value: {val:.2f},cash: {cash:.2f},max_cash: {np.max(money_value):.2f}")
            print(f"episode: {e + 1}/{num_episodes}, value: {val:.2f}, cash: {cash:.2f}, avg_cash: {np.mean(cash):.2f} duration: {dt}")
            money_50.append(np.mean(cash))
            episode_50.append(e+1)
        money_value.append(cash) # append episode end portfolio value
        episode.append(e+1)


    # save the weights when we are done
    if mode == 'train':
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # plot losses
    #     plt.plot(agent.model.losses)
        xlabel = "Losses"
        ylabel = "Frequency count"
        title = "Distribution of Model Losses"
        plot_metrics(agent.model.losses, xlabel, ylabel, title)
    #     plt.show()

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{mode}.npy', money_value)


# In[13]:


if mode == 'train':
    print("average train loss: {}".format(round(np.mean(agent.model.losses),3)))
    print("min train loss: {}, max train loss: {}".format(round(np.min(agent.model.losses),3),
                                                          round(np.max(agent.model.losses),3)))


# In[14]:


# agent.model.losses


# In[15]:


x = money_value
title = 'Cash in Hand Distribution'
xlabel = 'Money Vaue'
ylabel = 'Frequency Count'

plot_metrics(x, xlabel, ylabel, title)


# In[16]:


max(money_value)


# In[17]:



res = pd.DataFrame(list(zip(episode, money_value)))
res.columns= ['Episode','Value']
res['Episode']=50*(res['Episode']//50+1)
res['model']='DQN-1(State)'


# In[18]:


res.to_csv("play.csv",index=False)


# In[19]:


res1 = res.copy()
res1['Value'] = 225
res1['model'] = "ORACLE"


# In[20]:


res2 = res.copy()
res2['model']='DQN-2(State)'


# In[21]:


test = pd.concat([res,res1,res2])


# In[22]:


result =pd.concat([res,res1], axis=0)


# In[24]:


result.groupby('model')['Value'].min()


# In[25]:


result.groupby('model')['Value'].max()


# In[26]:


result.groupby('model')['Value'].mean()


# In[27]:


from matplotlib import pyplot
import seaborn

import seaborn as sns
sns.set_style("white")
# sns.set(style="darkgrid")


# In[30]:


# Plot the responses for different events and regions
fig, ax = pyplot.subplots(figsize=(9,4))
pyplot.title('MODEL PLAYING ', size=15)
pyplot.xlabel('Episode', size=15)
plt.ylim(0, 300)
plt.xlim(0,1000)
pyplot.ylabel('Cash', size=15)
sns.lineplot(x="Episode", y="Value",hue='model',style="model",markers=False, dashes=False,
             data=result)
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# In[31]:


fig, ax = pyplot.subplots(figsize=(5,5))
pyplot.title('MODELS COMPARISON (PLAY) ', size=15)
ax = pyplot.xlabel('MODEL', size=15)
ax = pyplot.ylabel('Value',size = 15)
ax= sns.boxplot( y="Value",palette="Set3", x="model", dodge=True, data = result)


# In[32]:


result.groupby('model')['Value'].mean()

