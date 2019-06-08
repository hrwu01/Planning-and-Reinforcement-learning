"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal as multi_normal 
import random
import time
from copy import deepcopy

class EnvAnimate:

	'''
	Initialize Inverted Pendulum
	'''
	def __init__(self, a = 1, b = 0.5, sigma = 1, k = 1, r = 1, gamma = 0.9, dt = 0.1, theta_grid = 20, v_max = 30, v_grid = 15, u_max = 10, u_grid = 10, episode_length = 500):

		# Change this to match your discretization
		self.a = a 
		self.b = b 
		self.sigma = sigma 
		self.k = k
		self.r = r 
		self.gamma = gamma 
		self.dt = dt
		self.episode_length = episode_length
		self.t = []

		for x in range(episode_length):
			self.t.append(x * self.dt)

		self.covariance = sigma * dt * np.eye(2)

		self.theta_min = -np.pi
		self.theta_max = np.pi
		self.theta_grid = theta_grid
		self.theta_res = np.pi / theta_grid
		self.theta_range = theta_grid * 2 + 1

		self.v_max = v_max
		self.v_min = -v_max
		self.v_grid = v_grid
		self.v_res = v_max / v_grid
		self.v_range = v_grid * 2 + 1

		self.u_max = u_max
		self.u_min = -u_max
		self.u_grid = u_grid
		self.u_res = u_max / u_grid
		self.u_range = u_grid * 2 + 1

		self.pdf_helper = np.array([[self.grid_to_state(i,j) for j in range(self.v_range)] for i in range(self.theta_range)])

		# print(self.pdf_helper[-1,-1])

		self.value = np.zeros((self.theta_range,self.v_range))
		self.policy = np.zeros((self.theta_range,self.v_range))
		self.policy_index = np.zeros((self.theta_range,self.v_range))

		self.control = []
		c = self.u_min
		while c <= u_max:
			self.control.append(c)
			c += self.u_res

		self.control_to_index = dict()
		index = 0
		for c in self.control:
			self.control_to_index[c] = index
			index += 1
		self.index_to_control = {v:k for k,v in self.control_to_index.items()}

		self.pdfs = self.compute_pdfs()

		self.cost = self.compute_cost()

		print('finish init')


		# # Random trajectory for example
		# self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
		# self.x1 = np.sin(self.theta)
		# self.y1 = np.cos(self.theta)
		# self.u = np.zeros(self.t.shape[0])

		# self.fig = plt.figure()
		# self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
		# self.ax.grid()
		# self.ax.axis('equal')
		# plt.axis([-2, 2, -2, 2])

		# self.line, = self.ax.plot([],[], 'o-', lw=2)
		# self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
		# self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)

	def compute_pdfs(self):
		result = {(i,j):dict() for i in range(self.theta_range) for j in range(self.v_range)}
		for i in range(self.theta_range):
			for j in range(self.v_range):
				state = self.grid_to_state(i,j)
				# print('state',state)
				for u in self.control:
					# print('mean',self.mean_new(state,u))
					distribution = multi_normal.pdf(self.pdf_helper,mean = self.mean_new(state,u),cov = self.covariance)
					# print('distribution',distribution)
					norm = np.sum(distribution)
					
					# time.sleep(1000)
					result[(i,j)][u] = distribution / norm
				# print('norm',norm)
				# time.sleep(0.01)
		return result

	def mean_new(self,state,u):
		new = state + np.array([state[1],self.a * np.sin(state[0]) - self.b * state[1] + u]) * self.dt
		while new[0] > np.pi:
			new[0] -= 2 * np.pi
		while new[0] < -np.pi:
			new[0] += 2 * np.pi
		return new

	def state_to_grid(self,state):
		theta = state[0]
		while theta > np.pi:
			theta -= 2 * np.pi
		while theta < -np.pi:
			theta += 2 * np.pi
		minimum_i = 1000
		for i in range(self.theta_range):
			distance = abs(theta - self.grid_to_state_i(i))
			if distance < minimum_i:
				minimum_i = distance
				best_i = i

		v = state[1]
		minimum_j = 1000
		for j in range(self.v_range):
			distance = abs(v - self.grid_to_state_j(j))
			if distance < minimum_j:
				minimum_j = distance
				best_j = j

		return best_i,best_j

	def grid_to_state(self,i,j):
		return np.array([i * self.theta_res + self.theta_min, j * self.v_res + self.v_min])

	def grid_to_state_i(self,i):
		return i * self.theta_res + self.theta_min

	def grid_to_state_j(self,j):
		return j * self.v_res + self.v_min

	def cost(self,i,j,u):
		state = self.grid_to_state(i,j)
		return (1 - np.exp(self.k * np.cos(state[0]) - self.k) + 0.5 * self.r * u * u) * self.dt

	def compute_cost(self):
		result = dict()
		for i in range(self.theta_range):
			for j in range(self.v_range):
				for u in self.control:
					result[(i,j,u)] = self.cost(i,j,u)
		return result

	def hamitonian(self,i,j,u):
		return self.cost[(i,j,u)] + self.gamma * np.sum(self.pdfs[(i,j)][u] * self.value)


	# def value_iteration(self):
	# 	new_value = np.zeros((self.theta_range,self.v_range))
	# 	new_policy = np.zeros((self.theta_range,self.v_range))
	# 	for i in range(self.theta_range):
	# 		for j in range(self.v_range):
	# 			minimum = 10000
	# 			for u in self.control:
	# 				v = self.hamitonian(i,j,u)
	# 				if v < minimum:
	# 					minimum = v 
	# 					best_move = u
	# 			new_value[i,j] = minimum
	# 			new_policy[i,j] = best_move
	# 	print('vi')
	# 	print(np.sum(self.value))
	# 	return new_value,new_policy


	def value_iteration(self):
		'''
		matrix-nize
		'''
		all_values = []
		for k in range(self.u_range):
			u = self.index_to_control[k]
			l = np.zeros((self.theta_range,self.v_range))
			pvsum = np.zeros((self.theta_range,self.v_range))
			for i in range(self.theta_range):
				for j in range(self.v_range):
					l[i,j] = self.cost[(i,j,u)]
					pvsum[i,j] = np.sum(self.pdfs[(i,j)][u] * self.value)
			v_slice = l + self.gamma * pvsum
			all_values.append(v_slice)
		value_matrix = np.array(all_values)
		new_value = np.min(value_matrix,axis = 0)
		new_policy = np.argmin(value_matrix,axis = 0)
		return new_value,new_policy

	def do_value_iteration(self,iteration = 100):
		for _ in range(iteration):
			new_value,new_policy = self.value_iteration()
			self.value,self.policy_index = new_value,new_policy

	def policy_evaluation(self):
		for _ in range(20):
			new_value = np.zeros((self.theta_range,self.v_range))
			for i in range(self.theta_range):
				for j in range(self.v_range):
					u = self.policy[i,j]
					new_value[i,j] = self.hamitonian(i,j,u)
			self.value = new_value

	def policy_improvement(self):
		all_values = []
		for k in range(self.u_range):
			u = self.index_to_control[k]
			l = np.zeros((self.theta_range,self.v_range))
			pvsum = np.zeros((self.theta_range,self.v_range))
			for i in range(self.theta_range):
				for j in range(self.v_range):
					l[i,j] = self.cost[(i,j,u)]
					pvsum[i,j] = np.sum(self.pdfs[(i,j)][u] * self.value)
			v_slice = l + self.gamma * pvsum
			all_values.append(v_slice)
		value_matrix = np.array(all_values)
		self.policy_index = np.argmin(value_matrix,axis = 0)
		self.update_policy()

	def do_policy_iteration(self,iteration = 40):
		for _ in range(iteration):
			self.policy_evaluation()
			self.policy_improvement()
			



	def next_state(self,state,u):
		mean = self.mean_new(state,u)
		theta,v = multi_normal.rvs(mean = self.mean_new(state,u), cov = np.array([[0.01,0],[0,0.01]]))
		while theta > np.pi:
			theta -= 2 * np.pi
		while theta < -np.pi:
			theta += 2 * np.pi
		return np.array([theta,v])


	def update_policy(self):
		for i in range(self.theta_range):
			for j in range(self.v_range):
				self.policy[i,j] = self.index_to_control[self.policy_index[i,j]]

	def generate_trajectory(self,start_theta = 1,start_v = 0, episode_length = 200):
		current_state = np.array([start_theta,start_v])
		theta = [start_theta]
		v = [start_v]
		u = []
		for _ in range(episode_length):
			i,j = self.state_to_grid(current_state)
			current_u = self.policy[i,j]
			new_state = self.next_state(current_state,current_u)
			theta.append(new_theta)
			v.append(new_v)
			u.append(current_u)
			current_state = new_state
		return theta,v,u



	'''
	Provide new rollout theta values to reanimate
	'''
	def new_data(self, theta, u):
		self.theta = theta
		self.x1 = np.sin(theta)
		self.y1 = np.cos(theta)
		self.u = u

		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
		self.ax.grid()
		self.ax.axis('equal')
		plt.axis([-2, 2,-2, 2])
		self.line, = self.ax.plot([],[], 'o-', lw=2)
		self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
		self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

	def init(self):
		self.line.set_data([], [])
		self.time_text.set_text('')
		return self.line, self.time_text

	def _update(self, i):
		thisx = [0, self.x1[i]]
		thisy = [0, self.y1[i]]
		self.line.set_data(thisx, thisy)
		self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
		return self.line, self.time_text

	def start(self):
		print('Starting Animation')
		print()
		# Set up plot to call animate() function periodically
		self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
		plt.show()


if __name__ == '__main__':
	# a = 1
	# b = 1
	# sigma = 0.5
	# k = 1
	# r = 1
	# gamma = 0.9

	# dt = 0.05
	# theta_grid = 20
	# v_max = 40
	# v_grid = 20
	# u_max = 15
	# u_grid = 15

	a = 1
	b = 0.5
	sigma = 1
	k = 1
	r = 1
	gamma = 0.9

	dt = 0.1
	theta_grid = 20
	v_max = 40
	v_grid = 20
	u_max = 10
	u_grid = 10

	start_theta = 0
	start_v = 10
	episode_length = 500

	# start_theta = random.uniform(-2,2)
	# start_v = random.uniform(-20,20)
	# episode_length = 500

	VI_iteration = 100
	PI_iteration = 100

	animation = EnvAnimate(a = a, b = b, sigma = sigma, k = k, r = r, gamma = gamma, dt = dt, theta_grid = theta_grid, v_max = v_max, v_grid = v_grid, u_max = u_max, u_grid = u_grid, episode_length = episode_length)






