# File to house all testing related functions

from output_frame import showimg, show_framestack
from gym.wrappers import GrayScaleObservation
from dkc_discretizer import DkcDiscretizer
from colour_modifier_observation import ColourModifierObservation

import time
import numpy as np

# Stable baselines imports
from stable_baselines3 import PPO

# Function to test the SNES gym-retro environment
def test_gymretro(env, showplot=False):
    # Iterate over 5 episodes
    for i in range(5):
        state = env.reset()

        # print(pos)

        # Declare vars for x coord
        xpos = 0
        xpos_max = 0

        counter = 0

        # Start timer
        start = time.time()

        # Game Render loop
        done = False
        while not done:
            # Display what is happening
            env.render()
            
            random_agent = False
            greedy_agent = True

            if random_agent:
                # Specify action/buttons randomly
                print("RANDOM AGENT")
                action = env.action_space.sample()

            # print(action)
            if greedy_agent:
                print("GREEDY AGENT")
                # Spam right and jump alternating
                if counter % 5 == 0:
                    # HIGH JUMP
                    # action = [3,3,3,3]
                    action = 3
                else:
                    # RIGHT
                    action = 1
                    # action = [1,1,1,1]

            # Update next frame with current actions
            state, reward, done, info = env.step(action)

            # update max x
            xpos = info['x']
            if xpos > xpos_max:
                xpos_max = xpos

            # Print img of current frame
            if showplot and counter % 450 == 0:
                showimg(state)
                # show_framestack(state)

            # Update score
            end = time.time()
            elapsed = end - start

            # If 60 seconds have passed, exit out
            if elapsed == 60:
                break

            counter += 1

    print("score:", info['score'])
    print("max xpos:", xpos_max)
    print("time:", elapsed)

    env.close()

# Function runs the model given an environment and path to the PPO model
def test_model(env, model_file_path, tries=10):
    # Load weights and environment
    model = PPO.load(model_file_path)

    # X max
    global_x_max = 0
    final_action_record = None

    # End of level conditions
    end_point = 5120
    finished = False

    for i in range(tries):
        print(f"Run #{i+1}")
        state = env.reset()
        done = False
        # Vars for keeping track of max x
        xpos = 0
        xpos_max = 0

        # List for storing actions of current run
        action_records = []

        # Run the model on the environment visually
        while not done:
            # Have model predict action and update state with that given action
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)

            # Add action to record list
            action_records.extend(action)

            # update max x
            xpos = info[0]['x']
            if xpos > xpos_max:
                xpos_max = xpos

            # Condition for if agent made it to the end
            if xpos >= end_point:
                print("!!!! MODEL REACHED THE END !!!!")
                finished = True
                break

            # Render environment
            env.render()
        
        # Print run summary
        print("> Furthest distance:", xpos_max)
        print("> Final score:", info[0]['score'])

        # Check if last run was the best, if so set the global x max and actions record
        if xpos_max > global_x_max:
            global_x_max = xpos_max
            final_action_record = action_records
        
        # Skip any subsequent runs if model made it to the end
        if finished:
            break

    print("Final actions:", final_action_record)
    print("Furthest distance:", global_x_max)

    # Save final action record to file
    model_name = model_file_path.split('/')[-1].rstrip('.zip')
    save_file = f'{model_name}_recorded.npy'
    with open(save_file, 'wb') as f:
        np.save(f, final_action_record)
    
    print(f"Saved best run in '{save_file}'...")


# Function for testing wrappers on Gym environments
def test_wrappers(env):
    # Apply discretizer wrapper
    env = DkcDiscretizer(env)
    # Apply colour modifier on env
    # env = ColourModifierObservation(env)
    # Apply grayscale
    # env = GrayScaleObservation(env)

    # Reset and step env to view new observation after wrapping
    # env.reset()

    # a = env.action_space.sample()
    # state, reward, done, info = env.step(a)

    # obs_space = env.observation_space
    # # print("Obs space:", obs_space)

    # showimg(state)

    test_gymretro(env, True)

# Function to play a pre-recorded set of moves from a model
def play_model(env, play):
    # Load in array from play file
    with open(f"{play}.npy", 'rb') as f:
        actions_list = np.load(f).tolist()

    # Run the model on the environment visually
    env.reset()
    for action in actions_list:
        # Have model predict action and update state with that given action
        state, reward, done, info = env.step(action)

        # Sleep for 1ms to slow down replay
        time.sleep(1/1000)

        if done:
            print("Scenario over...")

        # Render environment
        env.render()

    print("Finished replaying recording...")