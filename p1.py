U = {10000:{'red_10k','black_10k'},20000:{'red_10k','red_20k','black_10k','black_20k'}}
STATE_SPACE = {0,10000,20000,30000,40000,110000,120000,220000}

def print_value(value):
	print('Value Function')
	for k,v in value.items():
		print('V({}) = {}'.format(k,v))

def print_policy(policy):
	print('Policy')
	for k,v in policy.items():
		print('Pi({}) = {}'.format(k,v))

def print_value_and_policy(value,policy):
	print_value(value)
	print()
	print_policy(policy)

def initialize_value():
	return {0:0,10000:0,20000:0,30000:-30000,40000:-40000,110000:-110000,120000:-120000,220000:-220000}

def initialize_policy():
	return {10000:'red_10k',20000:'red_10k'}

def hamitonian(state,move,v):
	if move == 'red_10k':
		return 0.7 * v[state + 10000] + 0.3 * v[state - 10000]
	if move == 'red_20k':
		return 0.7 * v[state + 20000] + 0.3 * v[state - 20000]
	if move == 'black_10k':
		return 0.2 * v[state + 100000] + 0.8 * v[state - 10000]
	if move == 'black_20k':
		return 0.2 * v[state + 200000] + 0.8 * v[state - 20000]
	raise ValueError('Move not Available')

def value_iteration(value,policy):
	new_value = initialize_value()
	new_policy = initialize_policy()
	for state in U:
		minimum = 100000
		for move in U[state]:
			v = hamitonian(state,move,value)
			if v < minimum:
				minimum = v 
				best_move = move 
			new_value[state] = minimum
			new_policy[state] = best_move
	return new_value,new_policy

def do_value_iteration(value,policy,iteration = 50):
	for _ in range(iteration):
		new_value,new_policy = value_iteration(value,policy)
		if new_value == value and new_policy == policy:
			return new_value,new_policy
		value,policy = new_value,new_policy
	return value,policy

def policy_evaluation(value,policy,iteration = 50):
	for _ in range(iteration):
		new_value = initialize_value()
		for state in U:
			new_value[state] = hamitonian(state,policy[state],value)
		if new_value == value:
			return value 
		print(value)
		value = new_value
	return value

def policy_improvement(value,policy):
	for state in U:
		minimum = 100000
		for move in U[state]:
			v = hamitonian(state,move,value)
			if v < minimum:
				minimum = v 
				best_move = move
		policy[state] = best_move
	return policy

def do_policy_iteration(value,policy,total_iteration = 50,evaluation_iteration = 50):
	for i in range(total_iteration):
		v = policy_evaluation(value,policy,iteration = evaluation_iteration)
		p = policy_improvement(v,policy)
		if v == value and p == policy:
			return v,p 
		value,policy = v,p 
	return value,policy

if __name__ == '__main__':
	value = initialize_value()
	policy = initialize_policy()

	# value,policy = do_value_iteration(value,policy,iteration = 50)
	
	value,policy = do_policy_iteration(value,policy,total_iteration = 1,evaluation_iteration = 4)

	print_value_and_policy(value,policy)
