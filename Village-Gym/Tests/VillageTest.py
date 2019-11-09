import gym
import gym_village
env = gym.make('village-v0')
for i_episode in range(1):
    observation = env.reset()
    actions = [1,1,1,0,0,2,2,2,0,3,0,3,3]
    for t in range(13):
        #print(observation)
        #print(env.action_space)
        action = env.action_space.sample()
        #action = actions[t]
        env.render()
        observation, reward, done, info = env.step(action)
        if done:
            env.render()
            #env.show_result_2()
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
