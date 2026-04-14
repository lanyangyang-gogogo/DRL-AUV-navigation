import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR


from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

# Episode usually represents a complete attempt at a task.
# Epoch describes a full forward and backward pass over the dataset.
# Iterations refer to the number of algorithm updates.
# Iteration usually refers to the training steps within each episode.

def evaluate(network, epoch, eval_episodes=2):
    avg_reward = 0.0
    col = 0   # Accumulate collision count
    suc = 0
    print(f"Evaluating at epoch {epoch}...")
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        print(f"evaluate的第{_}次测试:")
        done = False
        while not done and count < 801:   # Maximum steps per eval_episode cannot exceed 800
            action = network.get_action(np.array(state))
            
            # Transform the action output by the model. Assuming action[0] is in the range [-1, 1],
            # it is scaled to [0, 1] via (action[0] + 1) / 2. This transformation meets the environment's action range requirements.
            a_in = [(action[0] + 1)/2, action[1], action[2]/10] # 4 Degrees of Freedom (DoF)
            
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -240:  # Increment collision count if triggered
                col += 1
            # If reward is greater than 290, it is considered a success
            if reward > 290:
                suc += 1
                
    avg_reward /= eval_episodes   # Calculate average reward across all evaluation episodes
    avg_col = col / eval_episodes
    avg_suc = suc / eval_episodes 
    print("..............................................")
    print(
        "保存模型Average Reward over %i Evaluation Episodes, Epoch %i, Avg Reward:%f, Collisions: %f, Successes: %f"
        % (eval_episodes, epoch, avg_reward, avg_col, avg_suc)
    )
    print("..............................................") 
    return avg_reward

# Improved network architecture
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 1024)
        self.layer_2 = nn.Linear(1024, 1024)
        self.layer_3 = nn.Linear(1024, 512)
        self.layer_4 = nn.Linear(512, 256)
        self.layer_5 = nn.Linear(256, action_dim)

        # Use LayerNorm instead of BatchNorm
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

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 1024)
        self.layer_2_s = nn.Linear(1024, 512)
        self.layer_2_a = nn.Linear(action_dim, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(256, 1)

        # Use LayerNorm instead of BatchNorm
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)

        self.dropout = nn.Dropout(0.3)

    def forward(self, s, a):
        # State branch
        s1 = F.relu(self.layer_1(s))  # (batch_size, 1024)
        s1 = F.relu(self.layer_norm1(self.layer_2_s(s1)))  # (batch_size, 512)

        # Action branch
        a1 = F.relu(self.layer_norm1(self.layer_2_a(a)))  # (batch_size, 512)

        # Directly add state and action features, no matrix multiplication needed
        s1 = F.relu(s1 + a1) 

        s1 = F.relu(self.layer_3(s1))  # (batch_size, 256)
        s1 = self.dropout(s1)
        q1 = self.layer_4(s1)  # (batch_size, 1)

        # Second calculation block
        s2 = F.relu(self.layer_1(s))  # (batch_size, 1024)
        s2 = F.relu(self.layer_norm1(self.layer_2_s(s2)))  # (batch_size, 512)
        a2 = F.relu(self.layer_norm1(self.layer_2_a(a)))  # (batch_size, 512)

        # Directly add state and action features, no matrix multiplication needed
        s2 = F.relu(s2 + a2) 

        s2 = F.relu(self.layer_3(s2))  # (batch_size, 256)
        s2 = self.dropout(s2)
        q2 = self.layer_4(s2)  # (batch_size, 1)

        return q1, q2
    
# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)   # Target network
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.actor_lr = 0.001  # Initial learning rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.critic_lr = 0.001  # Initial learning rate
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Initialize Cosine Annealing scheduler (T_max is the decay period, dynamically adjusted later)
        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=10000  # Temporary T_max, will be adjusted based on remaining steps
        )
        self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=10000
        )

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=1,
        tau=0.005,    # Soft update coefficient for target network
        policy_noise=0.2,
        noise_clip=0.5,  # Ensure noise is clipped to a maximum value
        policy_freq=2,  # In TD3, the Actor network is updated less frequently than the Critic (typically every 2 Critic updates)
        current_episode=0  # Pass in current episode number
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        avg_reward = 0.0  # Initialize sum of rewards
        self.episode_num = current_episode  # Update current episode number

        # Dynamically adjust T_max (total remaining iterations after 800 episodes)
        if self.episode_num >= 800:
            remaining_episodes = max_episodes - 800  # Remaining episodes
            avg_iter_per_ep = 800  # Assumed average iterations per episode (adjust based on actual situation)
            self.scheduler_actor.T_max = remaining_episodes * avg_iter_per_ep
            self.scheduler_critic.T_max = remaining_episodes * avg_iter_per_ep

        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)   

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation 
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()    
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
           
            av_loss += loss
        # Accumulate the rewards to calculate the average reward
            # avg_reward += reward.mean().item()
            avg_reward += reward.sum().item()

        # Trigger the cosine annealing learning rate scheduler
        # if self.episode_num >= 3000:
        #     self.scheduler_actor.step()
        #     self.scheduler_critic.step()

        # Calculate the average reward for the entire batch of iterations
        avg_reward /= iterations

        self.iter_count += 1
        # avg_reward /= iterations  
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)
        self.writer.add_scalar("Reward", avg_reward, self.iter_count)
       
        self.writer.add_scalar("actor_learning_rate", self.actor_optimizer.param_groups[0]['lr'], self.iter_count)
        self.writer.add_scalar("critic_learning_rate", self.critic_optimizer.param_groups[0]['lr'], self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 6400  # After how many steps to perform the evaluation
max_ep = 800  # maximum number of steps per episode
eval_ep = 5  # number of episodes for evaluation
max_timesteps = 2.4e6  # Maximum number of steps to perform
max_episodes = 5000  # Set maximum number of episodes, can be adjusted as needed
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 128  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates   
buffer_size = 1.8e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Whether to save the model or not
load_model = False # Whether to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

# Create storage directories if they do not exist
if not os.path.exists("../results"):
    os.makedirs("../results")

if save_model and not os.path.exists("../pytorch_models"):
    os.makedirs("../pytorch_models")

# Create the training environment
environment_dim = 20
robot_dim = 5
env = GazeboEnv("start_pid_demo_with_teleop.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 3
max_action = 1

# Create the network
network = TD3(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "../pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while episode_num < max_episodes:

    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
                current_episode=episode_num  # Pass current episode number
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="../pytorch_models")
            np.save("../results/%s" % (file_name), evaluations)
            epoch += 1
            
        print(f"Episode {episode_num} has ended. Resetting environment...")
        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Add some exploration noise
    if expl_noise > expl_min:     # Gradually decrease exploration noise
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[0:-5]) < 5
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.concatenate((
                np.random.uniform(-1, 1, 2),     # The range of the first two values is (-1, 1)
                np.random.uniform(-0.1, 0.1, 1)  # The range of the last value is (-0.1, 0.1)
            ))

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1)/2, action[1], action[2]/10]
    
    next_state, reward, done, target = env.step(a_in)
    
    # Check if max timesteps reached and print once
    if episode_timesteps + 1 == max_ep:
        print("Reached the maximum number of time steps")
        
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))

if save_model:
    # Save model to new path
    network.save("%s" % file_name, directory="../pytorch_models")

# Save evaluation results to new path
np.save("../results/%s" % file_name, evaluations)
