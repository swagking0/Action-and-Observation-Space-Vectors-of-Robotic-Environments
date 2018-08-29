import gym
import os, datetime
from PIL import Image


env = gym.make('HandReach-v0')

directory_to_save_png = '../shadow_action_space/png/' + datetime.datetime.now().strftime("%m%d_%H%M%S") + os.sep

env.reset()


for i in range(50):
    rgb_array = env.render(mode='rgb_array')
    im = Image.fromarray(rgb_array)
    lov = im.crop((220,200,770,700)) # the crop setting needed to changed as per direction and the required parameters are present in resource file
    action_space = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # change the range between (-1 & 1) to observer the action
    env.step(action_space)
    if not os.path.exists(directory_to_save_png):
        os.makedirs(directory_to_save_png)
    lov.save(directory_to_save_png + "pic_{0:05d}.png".format(i))
