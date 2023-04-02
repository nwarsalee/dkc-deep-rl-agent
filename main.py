# Main python file
# For testing and getting main stuff setup

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Local imports
from dkc_discretizer import DkcDiscretizer
from colour_modifier_observation import ColourModifierObservation
from train_progress import TrainingCallback
from reward_scale import RewardScaler
from file_operations import clear_past_train_progress, save_model
from testing import test_gymretro, test_model, test_wrappers

import retro
from gym.wrappers import GrayScaleObservation
import numpy as np
import time
import argparse
from typing import Callable

# Stable baselines imports
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO

print("Finished loading in packages...")

# Function creates a gym environment using the integration located in
def create_gym_environment(state='1Player.CongoJungle.JungleHijinks.Level1', record=False, record_path=""):
    # Add the game to the retro data
    game_name = "DonkeyKongCountrySNES"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    retro.data.Integrations.add_custom_path(os.path.join(script_dir, "custom_integrations"))
    assert(game_name in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    # Create the gym environment from the custom integration
    if record:
        os.makedirs(record_path, exist_ok=True)
        return retro.make(game_name, state=state, inttype=retro.data.Integrations.ALL, record=record_path)
    else:
        return retro.make(game_name, state=state, inttype=retro.data.Integrations.ALL)

def preprocess_env(env, hyper, preprocessing):
    """
    Function to preprocess game environment before training
    """
    # Discretize controls
    if preprocessing['discretize_actions']:
        env = DkcDiscretizer(env)

    # Add reward scaling to environment
    if preprocessing['reward_scale']:
        env = RewardScaler(env)

    # Apply custom colour modifications to image
    if preprocessing['colour_modifier']:
        env = ColourModifierObservation(env)

    # Turn image to grayscale
    if preprocessing['grayscale']:
        env = GrayScaleObservation(env, True)

    # Vectorize image
    if preprocessing['vectorize']:
        env = DummyVecEnv([lambda: env])

    # Stack frames of environment
    if preprocessing['frame_stack']:
        env = VecFrameStack(env, hyper['frame_stacks'], channels_order='last')

    return env

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

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

    parser.add_argument(
        "-r", "--record", action="store_true", help="Whether to record the experiment to a .bk2 file."
    )
    
    parser.add_argument(
        "-c", "--continuetrain", action="store_true", help="To continue training a prexisting model using the hyperparameters/preprocessing/rewards specified in this program."
    )
    
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="In continuing training, force the save to perform in the same directory."
    )

    return parser

# Parse incoming arguments
parser = init_argparse()
args = parser.parse_args()

# Hyperparameters to tune the model
hyper = {
    "timesteps" : 10000,
    "frame_stacks" : 4,
    "adaptive_alpha": False,
    "learn_rate" : 3e-4,
    "n_steps" : 512,
    "gamma" : 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.0075
}

# Preprocessing steps to use when training the model
preprocessing = {
    "discretize_actions" : True,
    "reward_scale": False,
    "colour_modifier" : True,
    "grayscale" : True,
    "vectorize" : True,
    "frame_stack" : True
}

# Set specified number of timesteps based on args
if args.steps:
    hyper['timesteps'] = args.steps

# Set adaptive learning rate...
if hyper['adaptive_alpha']:
    hyper["init_learn_rate"] = hyper['learn_rate']
    hyper['learn_rate'] = linear_schedule(hyper['learn_rate'])

# Folder saving
LOG_DIR = './logs/' # Where to save the logs that will be used by tensorboard
SAVE_DIR = './train_progress/' # Where to save model progress during training (deletes after every new run)
MODEL_DIR = './models/' # Where to save final models after training
RECORDING_DIR = "./recordings/" # Where to save the recordings 

# Model name to test/train via args
model_name = args.name
# Path to save/load model in
model_path = f"{MODEL_DIR}/{model_name}"
# Direct path to the zip of the model if it exists
model_file = f"{model_path}/{model_name}.zip"

# Flag for whether to train or test
test = args.test
experiment = args.experiment
continue_training = args.continuetrain

# Flag for whether to record the training
record = args.record
record_path = f"{RECORDING_DIR}/{model_name}"

# Other flags
overwrite_model = args.overwrite

# Create new env with gym retro
env = create_gym_environment(record=record, record_path=record_path)

if experiment:
    # Place for experimenting

    # test_wrappers(env)
    env = preprocess_env(env, hyper, preprocessing)
    # env = DkcDiscretizer(env)
    test_gymretro(env, showplot=True)
    
    # Exit out
    exit(0)


# TEST MODEL
if test:
    # Preprocess environment
    env = preprocess_env(env, hyper, preprocessing)

    print("Testing model named '{}'".format(model_name))

    test_model(env, model_file)

else:
    # Remove any previous training progress files before new training run
    clear_past_train_progress(SAVE_DIR)

    # Preprocess environment before training
    env = preprocess_env(env, hyper, preprocessing)

    # Create custom callback for logging progress
    training_callback = TrainingCallback(frequency=hyper['timesteps']/4, dir_path=SAVE_DIR)

    # Instantiate model that uses PPO
    policy_kwargs = dict(share_features_extractor=True)
    
    if continue_training:
        print("Continuing training from previously learned model...")
        model = PPO.load(model_file, tensorboard_log=f"{LOG_DIR}/{model_name}")
        model.set_env(env)
    else:
        model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=f"{LOG_DIR}/{model_name}", 
                    learning_rate=hyper["learn_rate"], n_steps=hyper['n_steps'],
                    device="cuda", gamma=hyper['gamma'], ent_coef=hyper['ent_coef'], clip_range=hyper['clip_range'],
                    policy_kwargs=policy_kwargs)

    print("Training with {} timesteps...".format(hyper['timesteps']))

    # Start timer
    start = time.time()

    # Train model
    model.learn(total_timesteps=hyper['timesteps'], callback=training_callback, progress_bar=True)

    # End timer
    end = time.time()
    total_time = end-start
    total_time = time_convert(total_time)
    print("Training time - {}".format(total_time))

    # Save model after completing training
    save_model(model, model_path, model_name, hyper, total_time, preprocessing, overwrite_model)
