# Main python file
# For testing and getting main stuff setup

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Local imports
from dkc_discretizer import DkcDiscretizer
from colour_modifier_observation import ColourModifierObservation
from helpers import TrainingCallback
from file_operations import clear_past_train_progress, save_model
from testing import test_gymretro, test_model, test_wrappers

import retro
from gym.wrappers import GrayScaleObservation
import numpy as np
import time
import argparse

# Stable baselines imports
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO

print("Finished loading in packages...")

# Function creates a gym environment using the integration located in
def create_gym_environment():
    # Add the game to the retro data
    game_name = "DonkeyKongCountrySNES"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    retro.data.Integrations.add_custom_path(os.path.join(script_dir, "custom_integrations"))
    assert(game_name in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    # Create the gym environment from the custom integration
    return retro.make(game_name, state='1Player.CongoJungle.JungleHijinks.Level1', inttype=retro.data.Integrations.ALL) # , use_restricted_actions=retro.Actions.DISCRETE

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

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60

    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(mins), int(sec))

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
    model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=f"{LOG_DIR}/{model_name}", learning_rate=hyper["learn_rate"], n_steps=hyper['n_steps'], device="cuda")

    print("Training with {} timesteps...".format(hyper['timesteps']))

    # Start timer
    start = time.time()

    # Train model
    model.learn(total_timesteps=hyper['timesteps'], callback=training_callback)

    # End timer
    end = time.time()
    total_time = end-start
    total_time = time_convert(total_time)
    print("Training time - {}".format(total_time))

    # Path to save model in
    model_path = f"{MODEL_DIR}/{model_name}"

    # Save model after completing training
    save_model(model, model_path, model_name, hyper, total_time)