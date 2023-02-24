# Main python file
# For testing and getting main stuff setup

import retro

# Create new env with gym retro
env = retro.make("DonkeyKongCountry-Snes", '1Player.CongoJungle.JungleHijinks.Level1')

env.reset()

done = False
while not done:
    # Display what is happening
    env.render()
    
    # Specify action/buttons randomly
    action = env.action_space.sample()

    # Update next frame with current actions
    env.step(action)