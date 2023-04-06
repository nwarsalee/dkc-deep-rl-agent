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
    for i in range(1):
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
        # Declare vars for y coord
        ypos = 0
        ypos_max = 0

        counter = 0

        # Start timer
        start = time.time()

        # Game Render loop
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

            # print("info:", info)
            
            # update max x
            xpos = info['x']
            if xpos > xpos_max:
                xpos_max = xpos

            # # update max y
            # ypos = info['y']
            # if ypos > ypos_max:
            #     ypos_max = ypos
            #     # print("y:", ypos)

            # Print img of current frame
            if showplot and counter % 450 == 0:
                # showimg(state)
                show_framestack(state)

            # Update score
            # score += reward
            end = time.time()
            elapsed = end - start

            # If 60 seconds have passed, exit out
            if elapsed == 60:
                break

            # print("Ep#", i, " Action:", action, " | Reward:", reward)
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

            # print(action)

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
    with open('saved_actions.npy', 'wb') as f:
        np.save(f, final_action_record)
    
    print("Saved best run in 'saved_actions.npy'...")


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