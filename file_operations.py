# File to house all functions relating to file operations

import os
import glob
import json

def clear_past_train_progress(save_dir):
    """
    Function to delete all models saved under the train_progress folder
    """
    files = glob.glob(f'{save_dir}/*')

    for f in files:
        os.remove(f)

def save_model(model, path, name, hyper=None, time_elapsed=None, preprocessing=None):
    """
    Function to save model and save its parameters used for training
    """
    # Create directories to save model in
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("WARNING - Path {} already exists, saving model in current directory...".format(path))
        path = '.'

    # Save model to load it in later for testing
    model.save(f"{path}/{name}.zip")

    # Read in scenario.json to obtain reward function parameters
    with open('./custom_integrations/DonkeyKongCountrySNES/scenario.json', 'r') as f:
        reward_params = json.load(f)

    # Save all relevant info relating to training run
    config = {
        "training_time" : time_elapsed,
        "hyper_params" : hyper,
        "reward_params" : reward_params,
        "preprocessing" : preprocessing
    }
    with open(f"{path}/config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)