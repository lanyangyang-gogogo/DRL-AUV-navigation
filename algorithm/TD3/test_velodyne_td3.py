import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 1024)
        self.layer_2 = nn.Linear(1024, 1024)
        self.layer_3 = nn.Linear(1024, 512)
        self.layer_4 = nn.Linear(512, 256)
        self.layer_5 = nn.Linear(256, action_dim)

        self.layer_norm1 = nn.LayerNorm(1024)
        self.layer_norm2 = nn.LayerNorm(1024)
        self.layer_norm3 = nn.LayerNorm(512)
        
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_norm1(self.layer_1(s)))
        s = F.relu(self.layer_norm2(self.layer_2(s)))
        s = F.relu(self.layer_norm3(self.layer_3(s)))
        s = F.relu(self.layer_4(s))
        s = self.dropout(s)
        a = self.tanh(self.layer_5(s))

        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 800  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 5
env = GazeboEnv("start_pid_demo_with_teleop.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 3

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "../pytorch_models")
    print("Model successfully loaded!")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

# --- Logging Metrics ---
episode_count = 1
success_count = 0
collision_count = 0
timeout_count = 0
episode_reward = 0.0

print("Starting testing loop...")

# Begin the testing loop
while True:
    action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1], action[2]/10]
    next_state, reward, done, target = env.step(a_in)
    
    episode_reward += reward

    # Check for max timesteps
    is_timeout = False
    if episode_timesteps + 1 == max_ep:
        is_timeout = True
        done = True
    
    done_bool = int(done)

    # On termination of episode
    if done_bool:
        # 1. Determine termination reason
        if target:
            success_count += 1
            termination_reason = "Success (Goal Reached)"
        elif is_timeout:
            timeout_count += 1
            termination_reason = "Timeout (Max steps reached)"
        else:
            collision_count += 1
            termination_reason = "Collision"

        # 2. Calculate Rates
        success_rate = (success_count / episode_count) * 100
        collision_rate = (collision_count / episode_count) * 100
        timeout_rate = (timeout_count / episode_count) * 100

        # 3. Print Detailed Log
        print(f"\n==================================================")
        print(f"Episode: {episode_count}")
        print(f"Termination Reason: {termination_reason}")
        print(f"Steps taken: {episode_timesteps + 1}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"--- Cumulative Statistics ---")
        print(f"Total Episodes: {episode_count}")
        print(f"Successes: {success_count} ({success_rate:.2f}%)")
        print(f"Collisions: {collision_count} ({collision_rate:.2f}%)")
        print(f"Timeouts: {timeout_count} ({timeout_rate:.2f}%)")
        print(f"==================================================")

        # 4. Reset environment and variables for the next episode
        state = env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0.0
        episode_count += 1
    else:
        state = next_state
        episode_timesteps += 1
