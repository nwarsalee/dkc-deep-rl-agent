# Donkey Kong Country - Deep RL Agent
Repository for a deep reinforcement learning agent that can play the first level of Donkey Kong Country

## Info
This is a final project for COMP4107 Neural Networks course at Carleton University.

### Members
- Josh Challenger
- Hadi Cheaito
- Nabeel Warsalee

### Final Result
After lots of training and experimentation, we managed to find a successful model that could complete the first level.

The recording of the run can be seen [here](https://www.youtube.com/watch?v=zl7wgRfEKMA)

# Features
This program allows for the training, testing and replay of models for Donkey Kong Country.

The implementation leverages the Proximal-Policy Optimization (PPO) RL algorithm for training our agent.

## Key Packages used
- OpenAI's Gym and Gym-Retro APIs
- Stable-Baselines3
## Training
Run a training episode by providing a name to save the model under and the number of training steps. Will produce a useable model stored in a zip file, with tensorboard logs produced in a separate log directory to monitor training performance.

To tune hyper-parameters, edit the values of the `hyper` dictionary in the `main.py`. For editing the reward weights, change the values of the `scenario.json` file in the `/custom_integrations` directory. For more information on *gym-retro* rewards and variables, see the [documentation](https://retro.readthedocs.io/en/latest/integration.html).

```
python main.py -n my_model -s 10000
```

## Testing
To test a recently trained model, simply provide the name of the trained model and use the testing flag. Testing mode will have the agent play 10 times, and will record the key presses of model that made it the furthest in those runs.
```
python main.py -n my_model -t
```

## Replay
Replay the best recorded run during the test mode by using the replay flag.
```
python main.py -n my_model -p
``` 