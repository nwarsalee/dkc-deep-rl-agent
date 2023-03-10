# Tensorflow/Keras Imports
from keras import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
# Reinforcement learning imports
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
Stable baselines imports

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
