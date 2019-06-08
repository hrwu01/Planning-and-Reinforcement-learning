import numpy as np 
from copy import deepcopy

MOVES = {(1,0):'S',(-1,0):'N',(0,1):'E',(0,-1):'W'}
MOVES_INVERTED = {v:k for k,v in MOVES.items()}

def print_value_and_policy(gamma,value,policy,iteration = 60,pi = False):
	if pi:
		v,p = do_policy_iteration(gamma,value,policy,iteration = iteration)
	else:
		v,p = do_value_iteration(gamma,value,policy,iteration = iteration)
	print('Optimal Value Function')
	print(np.array(v))
	print()
	print('Optimal Policy')
	print(np.array(p))

def	print_q_star_and_policy(gamma,q,policy,iteration = 60):
	new_q,new_policy = do_q_value_iteration(gamma,q,policy,iteration = iteration)
	q_star = [[0 for _ in range(5)] for _ in range(5)]
	for i in range(5):
		for j in range(5):
			q_star[i][j] = min(new_q[i][j].values())
	print('Optimal Q Function')
	print(np.array(q_star))
	print()
	print('Optimal Policy')
	print(np.array(new_policy))


def valid(i,j):
	return 0 <= i <= 4 and 0 <= j <= 4

def cost(i,j,di,dj):
	if i == 0:
		if j == 1:
			return -10
		if j == 3:
			return -5
	if valid(i + di, j + dj):
		return 0
	return 1

def next_position(i,j,di,dj):
	if i == 0:
		if j == 1:
			return 4,1
		if j == 3:
			return 2,3
	if valid(i + di, j + dj):
		return i + di, j + dj
	return i,j

def hamitonian(i,j,di,dj,gamma,value):
	new_i, new_j = next_position(i,j,di,dj)
	return cost(i,j,di,dj) + gamma * value[new_i][new_j]

def value_iteration(gamma,value,policy):
	'''
	return the updated value matrix and policy matrix after an iteration.
	'''
	new_value = [[0 for _ in range(5)] for _ in range(5)]
	new_policy = [[0 for _ in range(5)] for _ in range(5)]
	for i in range(5):
		for j in range(5):
			minimum = 10000
			for di,dj in MOVES:
				v = hamitonian(i,j,di,dj,gamma,value)
				if v < minimum:
					minimum = v
					best_move = MOVES[(di,dj)]
			new_value[i][j] = minimum
			new_policy[i][j] = best_move
	return new_value,new_policy

def do_value_iteration(gamma,value,policy,iteration = 50):
	for _ in range(iteration):
		new_value,new_policy = value_iteration(gamma,value,policy)
		if new_value == value and new_policy == policy:
			return new_value,new_policy
		value,policy = new_value,new_policy
	return value,policy

def policy_evaluation(gamma,value,policy):
	'''
	Compute the value matrix given a policy
	'''
	for _ in range(30):
		new_v = [[0 for _ in range(5)] for _ in range(5)]
		for i in range(5):
			for j in range(5):
				di,dj = MOVES_INVERTED[policy[i][j]]
				new_v[i][j] = hamitonian(i,j,di,dj,gamma,value)
		if new_v == value:
			return value 
		value = deepcopy(new_v)
	return value

def policy_improvement(gamma,value,policy):
	'''
	Compute the optimal policy matrix given the value
	'''
	for i in range(5):
		for j in range(5):
			minimum = 10000
			for di,dj in MOVES:
				v = hamitonian(i,j,di,dj,gamma,value)
				if v < minimum:
					minimum = v 
					best_move = MOVES[(di,dj)]
			policy[i][j] = best_move
	return policy

def do_policy_iteration(gamma,value,policy,iteration = 50):
	for i in range(iteration):
		v = policy_evaluation(gamma,value,policy)
		p = policy_improvement(gamma,v,policy)
		if v == value and p == policy:
			return v,p 
		value,policy = v,p 
	return value,policy

def q_value_iteration(gamma,q,policy):
	new_q = [[{'N':0,'S':0,'W':0,'E':0} for _ in range(5)] for _ in range(5)]
	new_policy = [['N' for _ in range(5)] for _ in range(5)]
	for i in range(5):
		for j in range(5):
			for move in MOVES_INVERTED:
				di,dj = MOVES_INVERTED[move]
				new_i,new_j = next_position(i,j,di,dj)
				new_q[i][j][move] = cost(i,j,di,dj) + gamma * min(q[new_i][new_j].values())
	for i in range(5):
		for j in range(5):
			minimum = 10000
			for move in MOVES_INVERTED:
				if new_q[i][j][move] < minimum:
					minimum = new_q[i][j][move]
					best_move = move 
			new_policy[i][j] = best_move
	return new_q,new_policy

def do_q_value_iteration(gamma,q,policy,iteration = 50):
	for _ in range(iteration):
		new_q,new_policy = q_value_iteration(gamma,q,policy)
		if new_q == q and new_policy == policy:
			return new_q,new_policy
		q,policy = new_q,new_policy
	return q,policy 


if __name__ == '__main__':
	# INITIALIZATION
	gamma = 0.9
	value = [[0 for _ in range(5)] for _ in range(5)]
	policy = [['N' for _ in range(5)] for _ in range(5)]
	q = [[{'N':0,'S':0,'W':0,'E':0} for _ in range(5)] for _ in range(5)]
	pi = True

	print_value_and_policy(gamma,value,policy,iteration = 100,pi = pi)
	
	# print_q_star_and_policy(gamma,q,policy,iteration = 95)



