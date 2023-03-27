# Custom Observation Wrapper code for Gym environment

from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

class ColourModifierObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, obs):
        def colour_quantization(obs):
            # --- COLOUR QUANTIZATION ---

            # Turn to LAB to be able to use Euclidean distance
            (h, w) = obs.shape[:2]
            image = cv2.cvtColor(obs, cv2.COLOR_BGR2LAB)

            # Reshape into feature vector
            img_vec = image.reshape((image.shape[0] * image.shape[1], 3))

            # Apply Kmeans
            clusters=8
            clt = MiniBatchKMeans(n_clusters = clusters, n_init="auto")
            labels = clt.fit_predict(img_vec)
            quant = clt.cluster_centers_.astype("uint8")[labels]

            # reshape the feature vectors to images
            quant = quant.reshape((h, w, 3))

            # convert from L*a*b* to RGB
            quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

            return quant

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
            # mask = mask_brown

            # Bitwise-AND mask and original image
            result = cv2.bitwise_and(img,img, mask=mask)

            return result
        

        # Make image have red hue
        # red = obs.copy()
        # red[:, :, (1,2)] = 0

        # Invert image colours
        # invert = obs.copy()
        # invert = 255 - invert

        # --- BASIC COLOUR MODIFICATION ---

        # image = cv2.cvtColor(obs, cv2.COLOR_BGR2LAB)

        # image = colour_quantization(obs)

        image = mask_colours(obs)

        return image
        # return obs