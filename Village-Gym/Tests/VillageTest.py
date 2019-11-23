import gym
import gym_village
env = gym.make('village-v0')
sum = 0
for i_episode in range(1000):
    observation = env.reset()
    #actions = [1,1,1,0,0,2,2,2,0,3,0,3,3]
    for t in range(13):
        #print(observation)
        #print(env.action_space)
        action = env.action_space.sample()
        #action = actions[t]
        #env.render()
        observation, reward, done, info = env.step(action)
        if done:
            #env.render()
            money, laborDays = env._get_obs();
            #if money > 0:
            sum += money
            env.show_result(True, i_episode)
            #print("Episode finished after {} timesteps".format(t+1))
            break
    #average = sum/i_episode
    #print('episodes: {}, average: {}'.format(i_episode, average))
env.close()