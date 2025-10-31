import numpy as np
from utils import *

num_states = 6
num_actions = 2

terminal_left_reward = 100
terminal_right_reward = 19
each_step_reward = 0

gamma = 0.5
misstep_prob = 0

generate_visualization()