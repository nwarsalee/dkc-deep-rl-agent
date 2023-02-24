# Main python file
# For testing and getting main stuff setup

import retro
import cv2
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam

# Function to build a keras model for DeepQLearning
def build_model(height, width, channels, actions):
    model = Sequential()
    # Setup convolutional layers
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation="relu", input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation="relu"))
    model.add(Convolution2D(64, (3,3), activation="relu"))
    model.add(Flatten())

    # Set up dense layers
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))

    # Set up output layer
    model.add(Dense(actions, activation='linear'))

    return model

# Create new env with gym retro
env = retro.make("DonkeyKongCountry-Snes", '1Player.CongoJungle.JungleHijinks.Level1')

# Iterate over 5 episodes
for i in range(5):
    state = env.reset()
    score = 0
    done = False

    # Grab resolution of game image
    inx, iny, inc = env.observation_space.shape
    pos = np.array([int(inx/8), int(iny/8)])

    print(pos)

    # Declare vars for x coord
    xpos = 0
    xpos_max = 0

    # Game Render loop
    while not done:
        # Display what is happening
        env.render()
        
        # Specify action/buttons randomly
        action = env.action_space.sample()

        # Turn image to grayscale
        state = cv2.resize(state, (pos[0], pos[1]))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = np.reshape(state, (pos[0], pos[1]))

        imgArray = state.flatten()

        # Update next frame with current actions
        state, reward, done, info = env.step(action)
        
        # Update score
        score += reward

        print("Ep#", i, " Action:", action, " | Reward:", reward)

        imgArray = []

env.close()