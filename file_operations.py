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

def get_config_info(file_path):
    """
    Function gets the config info file of a model to a variable
    Returns an empty array if config file wasn't found or retrivable
    """
    if not os.path.exists(file_path):
        return []
    
    config = []
    
    try:
        with open(file_path) as fin:
            old_config = json.load(fin)
            if isinstance(old_config, list):
                config = old_config
            elif isinstance(old_config, dict):
                 # support for older versions of config file
                config = [old_config]
    except Exception as e:
        print(f"get_config_info() failed with error {e}")
    finally:
        return config

def save_model(model, base_model_path, name, hyper=None, time_elapsed=None, preprocessing=None, overwrite_model=False):
    """
    Function to save model and save its parameters used for training
    """
    path = base_model_path
    # Create directories to save model in
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite_model:
        print(f"Overwritting model {name}")
    else:
        print("WARNING - Path {} already exists, saving model in current directory...".format(path))
        path = '.'

    # Save model to load it in later for testing
    model.save(f"{path}/{name}.zip")

    # Read in scenario.json to obtain reward function parameters
    with open('./custom_integrations/DonkeyKongCountrySNES/scenario.json', 'r') as f:
        reward_params = json.load(f)
    
    # Convert callable adaptive learning function to regular number
    if hyper and hyper["adaptive_alpha"]:
        hyper["learn_rate"] = hyper["init_learn_rate"]
        del hyper["init_learn_rate"]
    
    # Save all relevant info relating to training run
    config = get_config_info(f"{base_model_path}/config.json")
    config.append({
        "training_time" : time_elapsed,
        "hyper_params" : hyper,
        "reward_params" : reward_params,
        "preprocessing" : preprocessing
    })
    
    with open(f"{path}/config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)