import gym
import os, datetime
from PIL import Image


env = gym.make('FetchPickAndPlace-v1')

directory_to_save_png = '../../two_finger_action_vector/png/' + datetime.datetime.now().strftime("%m%d_%H%M%S") + os.sep

env.reset()


for i in range(50):
    rgb_array = env.render(mode='rgb_array')
    im = Image.fromarray(rgb_array)
    lov = im.crop((300,200,1000,650))
    action_space = [0,0,0,0] # change the range between (-1 & 1) to observer the action
    env.step(action_space)
    if not os.path.exists(directory_to_save_png):
        os.makedirs(directory_to_save_png)
    lov.save(directory_to_save_png + "pic_{0:05d}.png".format(i))
