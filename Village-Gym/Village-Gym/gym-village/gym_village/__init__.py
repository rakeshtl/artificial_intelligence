from gym.envs.registration import register

register(
    id='village-v0',
    entry_point='gym_village.envs:Village',
)