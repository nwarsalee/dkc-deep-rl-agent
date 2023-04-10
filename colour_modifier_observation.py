# Custom Observation Wrapper code for Gym environment

from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
import cv2

class ColourModifierObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        def mask_colours(img):
            # Convert BGR to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # define range of brown and green in HSV
            lower_brown = np.array([95,0,0])
            upper_brown = np.array([180,255,255])

            lower_green = np.array([24,0,0])
            upper_green = np.array([96,202,255])

            # Create a mask. Threshold the HSV image to get only yellow colors
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # Add masks together
            mask = mask_brown + mask_green

            # Bitwise-AND mask and original image
            result = cv2.bitwise_and(img,img, mask=mask)

            return result
        
        # Apply colour masking to attempt to remove background
        image = mask_colours(obs)

        return image