import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_space, action_space, net_width, max_action):
		super(Actor, self).__init__()

		# Define NN input layer by using the state space
		self.l1 = nn.Linear(state_space, net_width)
		# Define NN middle layer by # of neutron - which is 128 by default
		self.l2 = nn.Linear(net_width, net_width)
		# Define NN output layer according to the action space
		self.l3 = nn.Linear(net_width, action_space)

		self.max_action = max_action

	# Forward propagation
	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.max_action
		return a

# A pair of Critic is defined as we are using TD3
class Q_Critic(nn.Module):
	def __init__(self, state_space, action_space, net_width):
		super(Q_Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_space + action_space, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		# Only 1 deterministric action should be defined as output
		self.l3 = nn.Linear(net_width, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_space + action_space, net_width)
		self.l5 = nn.Linear(net_width, net_width)
		# Only 1 deterministric action should be defined as output
		self.l6 = nn.Linear(net_width, 1)

	# Forward propagation
	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		# Concatenates the given sequence of seq tensors in the given dimension
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1



class TD3(object):
	def __init__(
		self,
		has_terminal_state,
		state_space,
		action_space,
		max_action,
		# Discount factor. (Always between 0 and 1.)
		gamma=0.99,
		net_width=128,
		# Learning rate for policy - actor network
		a_lr=1e-4,
		# Learning rate for Q-networks - Critic network
		c_lr=1e-4,
		q_batchsize = 256
	):

		# Remark: highly recommend to train your own with GPU resources
		self.actor = Actor(state_space, action_space, net_width, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		# Remark: highly recommend to train your own with GPU resources
		self.q_critic = Q_Critic(state_space, action_space, net_width).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.has_terminal_state = has_terminal_state
		self.action_space = action_space
		self.max_action = max_action
		# Discount for future rewards
		self.gamma = gamma
		self.policy_noise = 0.2 * max_action
		# Limit for absolute value of target policy smoothing noise
		self.noise_clip = 0.5 * max_action
		# Target policy update parameter (1-tau)
		self.tau = 0.005
		# Num of transitions sampled from replay buffer
		self.q_batchsize = q_batchsize
		# Delayed policy updates parameter
		self.delay_counter = -1
		self.delay_freq = 1

	#Remark: select_action is only used when interact with the environment (Non-training)
	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a = self.actor(state)
		return a.cpu().numpy().flatten()

	def train(self,replay_buffer):
		# abbreviations used in the below training part
        # a = action
     	# s = current state
    	# r = reward
    	# s_prime = target state
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_prime, dead_mask = replay_buffer.sample(self.q_batchsize)
			noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			smoothed_target_a = (
					self.actor_target(s_prime) + noise  # Noisy on target action
			).clamp(-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
		# Take the smallest number between Q1 and Q2
		target_Q = torch.min(target_Q1, target_Q2)
		# Decide whether it reaches terminal state
		if self.has_terminal_state:
			# when terminal state is reached
			target_Q = r + (1 - dead_mask) * self.gamma * target_Q  
		else:
			# when terminal state is not reached
			target_Q = r + self.gamma * target_Q  


		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		if self.delay_counter == self.delay_freq:
			# Update Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = -1

	# Save the Actor and Critic models
	def save(self,episode):
		torch.save(self.actor.state_dict(), "ppo_actor{}.pth".format(episode))
		torch.save(self.q_critic.state_dict(), "ppo_q_critic{}.pth".format(episode))

	# Save the Actor and Critic models
	def load(self,episode):
		self.actor.load_state_dict(torch.load("ppo_actor{}.pth".format(episode)))
		self.q_critic.load_state_dict(torch.load("ppo_q_critic{}.pth".format(episode)))


