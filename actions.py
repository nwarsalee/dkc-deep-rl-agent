"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
"""

import gym
import numpy as np
import retro

class Discretizer(gym.ActionWrapper):
    """
    General Discretizer that works on all gym environments
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    From: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py#L9
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class DkcDiscretizer(Discretizer):
    """
    Use Donkey Kong-specific discrete actions
    based on https://strategywiki.org/wiki/Donkey_Kong_Country/Controls
    """
    def __init__(self, env):
        # NOTE: RIGHT, Y is running right (same with LEFT, Y)
        super().__init__(env=env, combos=[['LEFT'], ['LEFT', 'B'], ['RIGHT'], ['RIGHT', 'B'], ['DOWN', 'Y'], ['B'], ['Y']])