import gym
import gym_tictac4
env = gym.make('tictac4-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        #print(observation)
        #print(env.action_space)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.render()
            env.show_result_2()
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()