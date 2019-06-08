import gym
from random import choice,uniform
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time


class Planner:

	'''
	Initialization of all necessary variables to generate a policy:
		discretized state space
		control space
		discount factor
		learning rate
		greedy probability (if applicable)
	'''
	def __init__(self,env,alpha,gamma,epsilon):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.xmin = -1.2
		self.xmax = 0.6
		self.vmin = -0.07
		self.vmax = 0.07

		self.resolution_x = 0.1
		self.resolution_v = 0.01

		self.x_size = int(1.8 / self.resolution_x + 1)
		self.v_size = int(0.14 / self.resolution_v + 1)

		self.policy = [[0 for _ in range(self.v_size)] for _ in range(self.x_size)]
		self.q = [[[0,0,0] for _ in range(self.v_size)] for _ in range(self.x_size)]

		self.i0,self.j0 = self.state_to_grid((0,0))
		self.i1,self.j1 = self.state_to_grid((-1,0.05))
		self.i2,self.j2 = self.state_to_grid((0.25,-0.05))



	'''
	Learn and return a policy via model-free policy iteration.
	'''
	# def __call__(self, mc=False, on=True):
	# 	return self._td_policy_iter(on)

	def best_move(self,q_values):
		optimal_move = q_values.index(min(q_values))
		rand = uniform(0,1)
		if rand < epsilon:
			while True:
				move = choice((0,1,2))
				if not move == optimal_move:
					return move
		return optimal_move



	def state_to_grid(self,state):
		return int((state[0] - self.xmin) / self.resolution_x),int((state[1] - self.vmin) / self.resolution_v)





	'''
	TO BE IMPLEMENT
	TD Policy Iteration
	Flags: on : on vs. off policy learning
	Returns: policy that minimizes Q wrt to controls
	'''
	def td_policy_iter(self, on=True, time_stamp = 300, save_q = None):
		env = gym.make('MountainCar-v0')
		if on:
			c_state = env.reset()
			for _ in range(time_stamp):
				i,j = self.state_to_grid(c_state)
				# print('i,j',i,j)
				move = self.best_move(self.q[i][j])
				n_state, reward, done, _ = env.step(move)
				x,y = self.state_to_grid(n_state)
				cost = -reward
				next_move = self.best_move(self.q[x][y])
				self.q[i][j][move] = self.q[i][j][move] + self.alpha * (cost + self.gamma * self.q[x][y][next_move] - self.q[i][j][move])
				c_state = n_state
			if save_q:
				for u in range(3):
					save_q[0][u].append(self.q[self.i0][self.j0][u])
					save_q[1][u].append(self.q[self.i1][self.j1][u])
					save_q[2][u].append(self.q[self.i2][self.j2][u])
		else:
			c_state = env.reset()
			for _ in range(time_stamp):
				i,j = self.state_to_grid(c_state)
				# print('i,j',i,j)
				move = self.best_move(self.q[i][j])
				n_state, reward, done, _ = env.step(move)
				x,y = self.state_to_grid(n_state)
				cost = -reward
				self.q[i][j][move] = self.q[i][j][move] + self.alpha * (cost + self.gamma * min(self.q[x][y]) - self.q[i][j][move])
				c_state = n_state
			if save_q:
				for u in range(3):
					save_q[0][u].append(self.q[self.i0][self.j0][u])
					save_q[1][u].append(self.q[self.i1][self.j1][u])
					save_q[2][u].append(self.q[self.i2][self.j2][u])

	def find_optimal_policy(self):
		for i in range(len(self.policy)):
			for j in range(len(self.policy[0])):
				q_values = self.q[i][j]
				self.policy[i][j] = q_values.index(min(q_values))



	'''
	Sample trajectory based on a policy
	'''
	def rollout(self, env, policy=None, render=False):
		traj = []
		t = 0
		done = False
		c_state = env.reset()
		if policy is None:
			while not done or t < 10000:
				action = env.action_space.sample()
				if render:
					env.render()
				n_state, reward, done, _ = env.step(action)
				traj.append((c_state, action, reward))
				c_state = n_state
				t += 1

			env.close()
			return traj

		else:
			while not done or t < 10000:
				i,j = self.state_to_grid(c_state)
				action = policy[i][j]
				if render:
					env.render()

				n_state, reward, done, _ = env.step(action)
				traj.append((c_state, action, reward))
				c_state = n_state
				t += 1

			env.close()
			return traj

def plot_q(q,on_policy):
	for i in range(3):
		for j in range(3):
			plt.figure()
			plt.plot(q[i][j])
			if on_policy:
				if i == 0:
					plt.title('q(x = (0,0),u = {}), SARSA'.format(j),size = 14)
				elif i == 1:
					plt.title('q(x = (-1,0.05),u = {}), SARSA'.format(j),size = 14)
				else:
					plt.title('q(x = (0.25,-0.05),u = {}), SARSA'.format(j),size = 14)
			else:
				if i == 0:
					plt.title('q(x = (0,0),u = {}), Q-Learning'.format(j),size = 14)
				elif i == 1:
					plt.title('q(x = (-1,0.05),u = {}), Q-Learning'.format(j),size = 14)
				else:
					plt.title('q(x = (0.25,-0.05),u = {}), Q-Learning'.format(j),size = 14)
			plt.xlabel('Episodes',size = 12)
			plt.ylabel('q value',size = 12)
			plt.show(block = False)
	plt.pause(10000)

def plot_policy(policy,on_policy,x_max,y_max):
	# cmap = ListedColormap(['red','white','blue'])
	plt.figure()
	plt.imshow(policy)
	if on_policy:
		plt.title('optimal policy obtained by SARSA',size = 14)
	else:
		plt.title('optimal policy obtained by Q-Learning',size = 14)
	heatmap = plt.pcolor(policy)
	plt.colorbar(heatmap)
	plt.xticks([0,x_max],['-0.07','0.07'],size = 12)
	plt.yticks([0,y_max],['-1.2','0.6'],size = 12)
	plt.xlabel('Velocity',size = 12)
	plt.ylabel('Position',size = 12)
	plt.pause(10000)


if __name__ == '__main__':
	alpha = 0.25
	gamma = 0.9
	epsilon = 0.05
	episodes = 500
	time_stamp = 300
	on_policy = False
	q = [[[] for _ in range(3)] for _ in range(3)] #q[x][u], x -> (0,0),(-1,0.05),(0.25,-0.05); u -> 0,1,2

	env = gym.make('MountainCar-v0')
	planner = Planner(env,alpha,gamma,epsilon)
	for _ in range(episodes):
		planner.td_policy_iter(on = on_policy,time_stamp = time_stamp,save_q = q)
	planner.find_optimal_policy()

	# plot_q(q,on_policy)
	# plot_policy(planner.policy,on_policy,len(planner.policy[0]),len(planner.policy))

	traj = planner.rollout(env, policy=planner.policy, render=True)
	print(traj)

