# Main python file
# For testing and getting main stuff setup

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import retro
from gym.wrappers import GrayScaleObservation
from actions import DkcDiscretizer
from observations import ColourModifier
import cv2
import numpy as np
import time
import glob
import argparse
from matplotlib import pyplot as plt
# Stable baselines imports
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
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
        # inx, iny, inc = env.observation_space.shape
        # pos = np.array([int(inx/8), int(iny/8)])

        # print(pos)

        # Declare vars for x coord
        xpos = 0
        xpos_max = 0

        counter = 0

        # Game Render loop
        while not done:
            # Display what is happening
            env.render()
            
            # Specify action/buttons randomly
            # action = env.action_space.sample()

            # Spam right and jump alternating
            if counter % 5 == 0:
                # JUMP
                action = 3
            else:
                # RIGHT
                action = 1

            # Turn image to grayscale
            # state = cv2.resize(state, (pos[0], pos[1]))
            # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            # state = np.reshape(state, (pos[0], pos[1]))

            imgArray = state.flatten()

            # Update next frame with current actions
            state, reward, done, info = env.step(action)
            
            # update max x
            xpos = info['x']
            if xpos > xpos_max:
                xpos_max = xpos

            # Print img of current frame
            if counter % 450 == 0:
                showimg(state)

            # Update score
            score += reward

            # print("Ep#", i, " Action:", action, " | Reward:", reward)

            imgArray = []
            counter +=1

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

def preprocess_env(env, hyper):
    """
    Function to preprocess game environment before training
    """
    # Discretize controls
    env = DkcDiscretizer(env)

    # Apply custom colour modifications to image
    # env = ColourModifier(env)

    # Turn image to grayscale
    env = GrayScaleObservation(env, True)

    # Vectorize image
    env = DummyVecEnv([lambda: env])

    # Stack frames of environment
    env = VecFrameStack(env, hyper['frame_stacks'], channels_order='last')

    # Reset env to reflect new preprocessing
    env.reset()

    return env

def time_convert(text, sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60

    print("{} - {:02d}:{:02d}:{:02d}".format(text, int(hours), int(mins), int(sec)))

def init_argparse() -> argparse.ArgumentParser:
    """
    Define argument parser to allow for command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train an RL agent to play Donkey Kong Country."
    )

    parser.add_argument(
        "-n", "--name", type=str, required=True, help="Name to store/load model."
    )

    parser.add_argument(
        "-t", "--test", action="store_true", help="Test pre-saved model on new game instance. If not specified, will default to training."
    )

    parser.add_argument(
        "-e", "--experiment", action="store_true", help="Whether to run experiments in the sandbox. Used for testing new functionality and hacking around."
    )

    parser.add_argument(
        "-s", "--steps", type=int, help="Number of timesteps to train the model for. Default is set to 10 000 timesteps."
    )

    return parser

def clear_past_train_progress(save_dir):
    """
    Function to delete all models saved under the train_progress folder
    """
    files = glob.glob(f'{save_dir}/*')

    for f in files:
        os.remove(f)


# Function runs the model given an environment and path to the PPO model
def test_model(env, model_file_path):
    # Load weights and environment
    model = PPO.load(model_file_path)
    state = env.reset()
    
    counter = 0

    action_counter = 7*[0]

    # Run the model on the environment visually
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)

        # Save what action was done
        print_heatmap = True
        action_counter[action[0]] += 1

        if reward[0] > 0:
            print("x: {}, y:{}".format(info[0]['x'], info[0]['y']))
            print("reward: {}".format(reward[0]))
            print("action:", action_map[action[0]])
        
        # if print_heatmap:
            # for i, map in enumerate(action_map):
                # print("{} - {} presses".format(map, action_counter[i]))

        env.render()
        counter += 1

# Function for testing wrappers on Gym environments
def test_wrappers(env):
    # Apply discretizer wrapper
    env = DkcDiscretizer(env)
    # Apply colour modifier on env
    env = ColourModifier(env)
    # Apply grayscale
    env = GrayScaleObservation(env)

    # Reset and step env to view new observation after wrapping
    # env.reset()

    # a = env.action_space.sample()
    # state, reward, done, info = env.step(a)

    # obs_space = env.observation_space
    # # print("Obs space:", obs_space)

    # showimg(state)

    test_gymretro(env)

# Parse incoming arguments
parser = init_argparse()
args = parser.parse_args()

# Hyperparameters to tune the model
hyper = {
    "timesteps" : 10000,
    "frame_stacks" : 4,
    "learn_rate" : 1e-6,
    "n_steps" : 512
}

# Set specified number of timesteps based on args
if args.steps:
    hyper['timesteps'] = args.steps

# Folder saving
LOG_DIR = './logs/' # Where to save the logs that will be used by tensorboard
SAVE_DIR = './train_progress/' # Where to save model progress during training (deletes after every new run)
MODEL_DIR = './models/' # Where to save final models after training

# Create new env with gym retro
env = create_gym_environment()

# Allowable actions
# TODO: Move this somewhere else
action_map = [['LEFT'], ['RIGHT'], ['DOWN', 'Y'], ['B'], ['Y']]

# Model name to test/train via args
model_name = args.name

# Flag for whether to train or test
test = args.test
experiment = args.experiment

if experiment:
    # Place for experimenting

    # test_wrappers(env)
    
    # Exit out
    exit(0)


# TEST MODEL
if test:
    # Preprocess environment
    env = preprocess_env(env, hyper)

    model_file = f"{model_name}.zip"

    print("Testing model named '{}'".format(model_name))

    test_model(env, model_file)

else:
    # Remove any previous training progress files before new training run
    clear_past_train_progress(SAVE_DIR)

    # Preprocess environment before training
    env = preprocess_env(env, hyper)

    # Create custom callback for logging progress
    training_callback = TrainingCallback(frequency=hyper['timesteps']/4, dir_path=SAVE_DIR)

    # Instantiate model that uses PPO
    policy_kwargs = dict(share_features_extractor=False)
    # TODO: Use custom cnn
    model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=LOG_DIR, learning_rate=hyper["learn_rate"], n_steps=hyper['n_steps'], device="cuda")

    print("Training with {} timesteps...".format(hyper['timesteps']))

    # Start timer
    start = time.time()

    # Train model
    model.learn(total_timesteps=hyper['timesteps'], callback=training_callback)

    # End timer
    end = time.time()
    total_time = end-start
    time_convert('Training Time', total_time)

    # Create directories to save model in
    model_path = f"{MODEL_DIR}/{model_name}"
    os.makedirs(model_path)

    # Save model to load it in later for testing
    model.save(f"{model_path}/{model_name}.zip")