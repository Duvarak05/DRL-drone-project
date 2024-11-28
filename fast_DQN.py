import airsim
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from PIL import Image
from gym import spaces


class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # 1 input channel for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjusted fully connected layer size
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Reshape the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class AirSimDroneEnv:
    def __init__(self, ip_address, step_length, image_shape, epsilon=1.0, alpha=0.1, gamma=0.9):
        self.step_length = step_length
        self.image_shape = image_shape
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)  # 7 possible actions
        self._setup_flight()

        # Save the starting position
        self.starting_position = self.drone.simGetVehiclePose().position

        # Image request for depth perspective
        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

        # Q-learning parameters
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        # Initialize DQN model
        self.model = DQN(action_space=self.action_space.n)
        self.target_model = DQN(action_space=self.action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32

    def transform_obs(self, responses):
        """ Transform the image response from AirSim into a processed observation """
        if not responses or len(responses) == 0:
            raise ValueError("No image response from AirSim.")

        # Convert depth image data to a numpy array
        img1d = np.array(responses[0].image_data_float, dtype=float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)  # Normalize depth values
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        # Resize and convert to grayscale
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _setup_flight(self):
        """ Set up the drone for flight """
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync().join()

        # Set an initial position close to the ground (e.g., z = -3)
        self.drone.moveToPositionAsync(0.1, 0.1, -5, 4).join()

        # Update starting position
        self.starting_position = self.drone.simGetVehiclePose().position

    def interpret_action(self, action):
        """ Map the action to specific movements """
        offsets = {
            0: (self.step_length, 0, 0),  # Move forward
            1: (-self.step_length, 0, 0),  # Move backward
            2: (0, self.step_length, 0),  # Move right
            3: (0, -self.step_length, 0),  # Move left
            4: (0, 0, 1),  # Move up slightly
            5: (0, 0, -1),  # Move down slightly
            6: (0, 0, 0),  # Hover
        }

        # Prevent drone from going too high or too low
        current_z = self.state["position"].z_val
        if action == 4 and current_z > -10:  # Limit maximum altitude
            return offsets[6]  # Hover instead
        elif action == 5 and current_z < -1:  # Limit minimum altitude
            return offsets[6]  # Hover instead

        return offsets[action]

    def _do_action(self, action):
        """ Execute the action with a cooldown """
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity

        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            3.0  # Increase duration for smoother movement
        ).join()

        time.sleep(0.1)  # Short cooldown between actions

    def _compute_reward(self, cumulative_reward):
      """ Compute the reward with collision handling and altitude restriction """
      reward = 0  # Small penalty for taking a step
      done = False
      termination_reason = None  # Variable to store the reason for termination

      position = self.state["position"]
      altitude = -position.z_val  # Altitude is negative in AirSim

      if altitude > 20:  # Penalize if the drone flies above 20 meters
        reward -= 2
        done = True
        termination_reason = "Drone flew above 20 meters."

      if self.state["collision"]:
        reward -= 10  # Deduct 10 points for a collision
        done = True  # End the episode on collision
        termination_reason = "Collision occurred."

      else:
        progress = np.linalg.norm(
            np.array([position.x_val, position.y_val, position.z_val]) -
            np.array([self.state["prev_position"].x_val, self.state["prev_position"].y_val, self.state["prev_position"].z_val])
        )
        reward += progress  # Reward based on progress made

      # Check cumulative reward threshold
      if cumulative_reward + reward > 200:
        reward = 200 - cumulative_reward  # Cap reward to ensure total is exactly 200
        done = True
        termination_reason = "Cumulative reward exceeded 200."

     # Print the reason for episode termination (if applicable)
      if done and termination_reason:
        print(f"Episode terminated. Reason: {termination_reason}")

      return reward, done, termination_reason
 
    def _get_obs(self):
        """ Get current observation from the drone """
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        # Update the state with current position, velocity, and collision info
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def step(self, action, cumulative_reward):
     """ Take one step in the environment """
     self._do_action(action)
     obs = self._get_obs()
     reward, done, termination_reason = self._compute_reward(cumulative_reward)

     # Print the reason for episode termination (if done)
     if done:
        print(f"Episode terminated. Reason: {termination_reason}")

     return obs, reward, done, self.state


    def reset(self):
        """ Reset the environment """
        self._setup_flight()

        # Set random orientation for the drone
        random_orientation = airsim.Quaternionr(
            random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        )
        self.drone.simSetVehiclePose(airsim.Pose(self.starting_position, random_orientation), True)

        return self._get_obs()

    def store_experience(self, experience):
        """ Store the experience in the replay buffer """
        self.replay_buffer.append(experience)

    def sample_experience(self):
        """ Sample a batch from the replay buffer """
        return random.sample(self.replay_buffer, self.batch_size)

    def train_dqn(self):
        """ Train the DQN model using experience replay """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        batch = self.sample_experience()
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to torch tensors
        states = np.array(states)
        next_states = np.array(next_states)

        # Reshape the states and next_states to have the correct shape
        states = torch.FloatTensor(states).permute(0, 3, 1, 2)  # From (batch, 84, 84, 1) to (batch, 1, 84, 84)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2)

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute Q-values
        q_values = self.model(states).gather(1, actions.view(-1, 1))  # Get Q-values for selected actions
        next_q_values = self.target_model(next_states).max(1)[0].detach()  # Max Q-value for next state

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss
        loss = self.loss_fn(q_values.view(-1), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target network
        if random.random() < 0.01:  # Update the target model every so often
            self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        """ Select an action based on the epsilon-greedy policy """
        if random.random() < self.epsilon:
            return random.choice(range(self.action_space.n))  # Exploration
        else:
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
            q_values = self.model(state)
            return torch.argmax(q_values).item()  # Exploitation

    def decay_epsilon(self):
        """ Decay epsilon over time """
        self.epsilon = max(self.epsilon * 0.995, 0.01)

import matplotlib.pyplot as plt
from collections import deque
import torch.optim as optim

# Initialize a list to store results for each tuning experiment
tuning_results = {}

# Define different values for the hyperparameters to test
learning_rates = [0.0001, 0.001]
discount_factors = [0.95, 0.99]
exploration_decay = [0.995, 0.99]
batch_sizes = [32, 64]
replay_buffer_sizes = [10, 5]

# Tuning for each hyperparameter
for lr in learning_rates:
    for gamma in discount_factors:
        for decay in exploration_decay:
            for batch_size in batch_sizes:
                for buffer_size in replay_buffer_sizes:
                    # Initialize the environment with the current hyperparameters
                    env = AirSimDroneEnv(
                        ip_address='127.0.0.1',
                        step_length=5.0,
                        image_shape=(84, 84, 1),
                        epsilon=1.0,
                        alpha=lr,
                        gamma=gamma,
                    )
                    env.batch_size = batch_size
                    env.replay_buffer = deque(maxlen=buffer_size)
                    env.optimizer = optim.Adam(env.model.parameters(), lr=lr)

                    # Initialize a list to store rewards
                    episode_rewards = []

                    # Training loop
                    for episode in range(10):  # Adjust range for more episodes
                        state = env.reset()
                        done = False
                        total_reward = 0

                        while not done:
                            action = env.select_action(state)
                            next_state, reward, done, _ = env.step(action, total_reward)
                            total_reward += reward

                            # Store experience and train the DQN
                            env.store_experience((state, action, reward, next_state, done))
                            env.train_dqn()

                            state = next_state

                            # End episode on collision
                            if done and reward == -10:
                                break

                        # Append the total reward of the episode
                        episode_rewards.append(total_reward)

                        # Decay epsilon after each episode
                        env.decay_epsilon()

                        # Print 'done' after each episode is completed
                        print(f"Episode {episode+1} completed with total reward {total_reward}")
                        print("done")

                    # Store results for the current set of hyperparameters
                    tuning_key = f"LR={lr}, γ={gamma}, Decay={decay}, Batch={batch_size}, Buffer={buffer_size}"
                    tuning_results[tuning_key] = episode_rewards

# Plot all tuning results
plt.figure(figsize=(12, 8))
for key, rewards in tuning_results.items():
    plt.plot(range(1, len(rewards) + 1), rewards, label=key)

plt.title("Reward Graphs for Different Hyperparameter Settings")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
