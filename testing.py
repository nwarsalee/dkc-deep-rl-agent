# File to house all testing related functions

from output_frame import showimg
from gym.wrappers import GrayScaleObservation
from dkc_discretizer import DkcDiscretizer
from colour_modifier_observation import ColourModifierObservation

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

        # Game Render loop
        while not done:
            # Display what is happening
            env.render()
            
            # Specify action/buttons randomly
            action = env.action_space.sample()

            print(action)

            # Spam right and jump alternating
            if counter % 5 == 0:
                # HIGH JUMP
                action = [3,3,3,3]
            else:
                # RIGHT
                # action = 1

                # JUMP
                action = [1,1,1,1]

            # Turn image to grayscale
            # state = cv2.resize(state, (pos[0], pos[1]))
            # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            # state = np.reshape(state, (pos[0], pos[1]))

            imgArray = state.flatten()

            # Update next frame with current actions
            state, reward, done, info = env.step(action)

            print("info:", info)
            
            # update max x
            # xpos = info['x']
            # if xpos > xpos_max:
            #     xpos_max = xpos

            # # update max y
            # ypos = info['y']
            # if ypos > ypos_max:
            #     ypos_max = ypos
            #     # print("y:", ypos)

            # Print img of current frame
            if showplot and counter % 450 == 0:
                showimg(state)

            # Update score
            score += reward

            # print("Ep#", i, " Action:", action, " | Reward:", reward)

            imgArray = []
            counter +=1

    env.close()

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

        print(action)

        # Save what action was done
        # print_heatmap = True
        # action_counter[action[0]] += 1

        # if reward[0] > 0:
        #     print("x: {}, y:{}".format(info[0]['x'], info[0]['y']))
        #     print("reward: {}".format(reward[0]))
        #     print("action:", action)
        
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
    env = ColourModifierObservation(env)
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