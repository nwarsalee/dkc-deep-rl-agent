# Main python file
# For testing and getting main stuff setup

import os
import retro
import cv2
import numpy as np
# Tensorflow/Keras Imports
from keras import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
# Reinforcement learning imports
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# Function to build a keras NeuralNetwork model for DeepQLearning
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

# Function to build a DeepQ Learning agent based on the keras AI model
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000)

    return dqn

# Function to test the SNES gym-retro environment
def test_gymretro(env):
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

# Function creates a gym environment using the integration located in
def create_gym_environment():
    gameName = "DonkeyKongCountrySNES"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    retro.data.Integrations.add_custom_path(os.path.join(script_dir, "custom_integrations"))
    assert(gameName in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    return retro.make(gameName, state='1Player.CongoJungle.JungleHijinks.Level1', inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.DISCRETE)
print("Finished loading in packages...")

# Create new env with gym retro
env = create_gym_environment()

test = False

if test:
    test_gymretro(env)
else:
    # Get screen and move information from the environment
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    # Build the model and the agent
    model = build_model(height, width, channels, actions)
    agent = build_agent(model, actions)

    # Compile model and run training
    agent.compile(Adam(lr=1e-4))
    agent.fit(env, nb_steps=1000, visualize=True, verbose=2)

    # TODO:
    #   - Understand the nb_steps better (seems to refer to # of frames, we want to set episodes/epochs of mulitple tries)
    #   - What to do with agent after being trained
    #   - Figure out how to edit hyperparams