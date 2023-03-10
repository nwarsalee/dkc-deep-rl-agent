# Main python file
# For testing and getting main stuff setup

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import retro
from gym.wrappers import GrayScaleObservation
from actions import DkcDiscretizer
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Stable baselines imports
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from helpers import TrainingCallback

print("Finished loading in packages...")

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
    # Add the game to the retro data
    game_name = "DonkeyKongCountrySNES"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    retro.data.Integrations.add_custom_path(os.path.join(script_dir, "custom_integrations"))
    assert(game_name in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    # Create the gym environment from the custom integration
    return retro.make(game_name, state='1Player.CongoJungle.JungleHijinks.Level1', inttype=retro.data.Integrations.ALL) # , use_restricted_actions=retro.Actions.DISCRETE

def showimg(state):
    plt.imshow(state)
    plt.show()

def show_framestack(state):
    plt.figure(figsize=(10,8))
    for i in range(state.shape[3]):
        plt.subplot(1,4, i+1)
        plt.imshow(state[0][:,:,i])
    plt.show()


# Function runs the model given an environment and path to the PPO model
def test_model(env, model_file_path):
    # Load weights and environment
    model = PPO.load(model_file_path)
    state = env.reset()
    
    # Run the model on the environment visually
    while True:
        action, _ = model.predict(state)
        state, _, _, _ = env.step(action)
        env.render()


# Hyperparameters to tune the model
hyper_totaltimesteps = 10000
hyper_numOfFrameStack = 4

# Folder saving
LOG_DIR = './logs/' # Where to save the logs
SAVE_DIR = './train/' # Where to save the model weights training increments

# Create new env with gym retro
env = create_gym_environment()

test = True

if test:
    env = DkcDiscretizer(env)
    env = GrayScaleObservation(env, True)

    # Vectorize image
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, hyper_numOfFrameStack, channels_order='last')

    test_model(env, "./latest_model_10000.zip")
    #test_gymretro(env)
else:
    # Get screen and move information from the environment
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    # Simply movement controls
    # TODO: Figure out if this works???
    env = DkcDiscretizer(env)
    # env.unwrapped.get_action_meanings()

    # Preprocess environment before sending to train
    # TODO:
    # Turn image into grayscale
    env = GrayScaleObservation(env, True)

    # Vectorize image
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, hyper_numOfFrameStack, channels_order='last')

    env.reset()

    state, reward, done, info = env.step([env.action_space.sample()])

    # TODO: IMPLEMENT LOGGING AND SAVING FOR MODEL
    
    training_callback = TrainingCallback(frequency=5000, dir_path=SAVE_DIR)
    # Instantiate model that uses PPO
    # TODO: Use custom cnn
    model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=LOG_DIR, learning_rate=1e-5, n_steps=512, device="cuda") 

    # TODO: Implement use of episodes
    model.learn(total_timesteps=10000, callback=training_callback)
    
    model.save(f"latest_model_{hyper_totaltimesteps}")
    
    # TODO:
    #   - Understand the nb_steps better (seems to refer to # of frames, we want to set episodes/epochs of mulitple tries)
    #   - What to do with agent after being trained
    #   - Figure out how to edit hyperparams