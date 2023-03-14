# Custom Observation Wrapper code for Gym environment

from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
import cv2

class ColourModifier(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, obs):
        # Make image have red hue
        # red = obs.copy()
        # red[:, :, (1,2)] = 0

        # Invert image colours
        # invert = obs.copy()
        # invert = 255 - invert

        image = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV )

        return image
        # return obs