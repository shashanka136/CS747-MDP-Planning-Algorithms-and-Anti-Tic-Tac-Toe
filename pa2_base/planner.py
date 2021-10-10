import numpy as np
# import sys
from pulp import *
import argparse

def get_mdp(mdp_file):
	global s,a, ends, set_ends, t, p, r, mdptype, gamma, totpr
	global maxrw, minrw
	file = open(mdp_file, 'r')
	maxrw = 0
	minrw = 0
	l = file.readline()
	s = int(l.rstrip().lstrip().split()[1].lstrip())
	l = file.readline()
	a = int(l.rstrip().lstrip().split()[1].lstrip())
	l = file.readline()
	ends = [int(x.lstrip().rstrip()) for x in l.rstrip().lstrip().split()[1:]]
	if ends[0] == -1:
		ends = []
	ends.sort()
	set_ends = set(ends)
	t = [[[] for _ in range(a)] for _ in range(s)]
	r = [[[] for _ in range(a)] for _ in range(s)]
	p = [[[] for _ in range(a)] for _ in range(s)]
	totpr = [[0.00 for _ in range(a)] for _ in range(s)]
	while True:
		l = file.readline()
		l = [x.lstrip().rstrip() for x in l.lstrip().rstrip().split()]
		if l[0] != 'transition':
			break
		maxrw = max(maxrw, float(l[4]))
		minrw = min(minrw, float(l[4]))
		if float(l[5]) == 0.00:
			continue
		totpr[int(l[1])][int(l[2])] += float(l[5])*float(l[4])
		if int(l[3]) in set_ends:
			continue
		t[int(l[1])][int(l[2])].append(int(l[3]))
		r[int(l[1])][int(l[2])].append(float(l[4]))
		p[int(l[1])][int(l[2])].append(float(l[5]))
	t = [[np.array(y, dtype = np.int32) for y in x] for x in t]
	p = [[np.array(y) for y in x] for x in p]
	r = [[np.array(y) for y in x] for x in r]
	totpr = np.array(totpr)
	mdptype = l[1]
	l = file.readline()
	gamma = float(l.rstrip().lstrip().split()[1].lstrip())
	file.close()
	pass

def get_full(V, policy):
	# V -> (nonts,1)
	# policy -> (nonts,1)
	Vfull = np.zeros((s,1))
	Vfull = V[state_map]
	policyfull = np.zeros((s,1))
	policyfull = policy[state_map]
	Vfull[ends] = 0.
	policyfull[ends] = 0
	return Vfull, policyfull
	pass

def get_vpi(policy): # get Value function for given policy
	# policy -> (nonts,1)
	A = np.zeros((nonts, nonts))
	end_ind = 0
	B = np.zeros((nonts, 1))
	for i in range(s):
		if end_ind >= len(ends) or ends[end_ind] != i:
			A[state_map[i]][state_map[i]] = 1
			B[state_map[i]][0] = totpr[i][policy[state_map[i]][0]]
		else:
			end_ind += 1
			continue
		for j, nxt in enumerate(t[i][policy[state_map[i]][0]]):
			A[state_map[i]][state_map[nxt]] -= gamma * p[i][policy[state_map[i]][0]][j]
	
	V = np.matmul(np.linalg.inv(A), B)
	return V

def get_Q(V): # get action value function for given value function
	Q = np.zeros((nonts, a))
	end_ind = 0
	for i in range(s):
		if end_ind < len(ends) and ends[end_ind] == i:
			end_ind += 1
			continue
		# Tp = np.zeros((nonts,a))
		for j in range(a): 
			for k, nxt in enumerate(t[i][j]):
				# Tp[state_map[nxt]][j] = gamma*p[i][j][k]
				Q[state_map[i]][j] += gamma*p[i][j][k]*V[state_map[nxt]][0]
			# print(type(t[i][j]), file = sys.stderr)
			# print(list(state_map[t[i][j]]), file = sys.stderr)
			# Tp[np.ix_(list(state_map[t[i][j]]),[j])] = gamma * p[i][j].reshape((len(t[i][j]),1))
			Q[state_map[i]][j] += totpr[i][j]
		# Q[[state_map[i]], :]+= (np.matmul(V.T, Tp)).astype(float)
	return Q

def value_iteration(): # run value iteration
	np.random.seed(0)
	V = np.zeros((2,nonts,1))
	x = 1
	V[0] = np.random.uniform(minrw, maxrw, (nonts,1))
	V[1] = np.max(get_Q(V[0]), axis = 1).reshape((nonts,1))
	while not (V[0] == V[1]).all():
		x ^= 1
		# print(V[0], V[1])
		V[x] = np.max(get_Q(V[x^1]), axis = 1).reshape((nonts,1))
	# print(get_Q(V[x])[0], file=sys.stderr)
	V_ret, policy_ret = get_full(V[x], np.argmax(get_Q(V[x]), axis = 1).reshape((nonts,1)))
	return V_ret, policy_ret

def linear_programming():
	LpSolverDefault.msg = 0
	prob = LpProblem("LP_formulation", LpMaximize)
	V = {i : LpVariable(name =f'V{i}') for i in range(nonts)}
	prob += -lpSum(V.values())
	for i in range(s):
		if i in set_ends:
			continue
		for j in range(a):
			prob += V[state_map[i]] >= totpr[i][j] + \
							gamma*lpSum([p[i][j][k]*V[state_map[nxt]] for k, nxt in enumerate(t[i][j]) if nxt not in set_ends])
	prob.solve()
	Vx = np.zeros((nonts,1))
	for i in range(nonts):
		Vx[i][0] = V[i].value()
	# print(get_Q(Vx)[0], file=sys.stderr)
	
	V_ret, policy_ret = get_full(Vx, np.argmax(get_Q(Vx), axis = 1).reshape((nonts,1)))
	return V_ret, policy_ret

def howard_policy_iteration():
	np.random.seed(0)
	policy = np.random.randint(low = 0, high = a, size = (nonts,1))
	V = get_vpi(policy)
	Q = get_Q(V)
	new_policy = np.argmax(Q, axis = 1).reshape((nonts,1))
	while not (policy == new_policy).all():
		policy = new_policy
		V = get_vpi(policy)
		Q = get_Q(V)
		equal_to_max = (np.max(Q, axis = 1).reshape((nonts,1))) -V > 0.00
		new_policy= np.where(equal_to_max, np.argmax(Q, axis = 1).reshape((nonts,1)), policy)
		
	# print(Q[0], file=sys.stderr)
	V, policy = get_full(V, policy)
	return V,policy
	pass

if __name__ == '__main__':
	global s, nonts, a, ends, set_ends, t, p, r, mdptype, gamma, totpr, state_map
	parser = argparse.ArgumentParser()

	parser.add_argument('--mdp', type=str)
	parser.add_argument('--algorithm', type=str, default='nothing')
	args = parser.parse_args()
	mdp_file = args.mdp
	algo = args.algorithm
	get_mdp(mdp_file)
	nonts= s -len(ends)
	state_map = np.arange(s, dtype=np.int32)
	ends = np.array(ends, dtype=np.int32)
	# print(len(ends))
	end_ind = 0
	for i in range(s):
		state_map[i] = i-end_ind
		if end_ind < len(ends) and ends[end_ind] == i:
			end_ind += 1
	state_map[ends] = 0
	if algo == 'vi' or algo == 'nothing':
		V,pi = value_iteration()
	elif algo == 'hpi':
		V,pi = howard_policy_iteration()
	elif algo == 'lp':
		V,pi = linear_programming()
	for i in range(s):
		print('{:.8f}'.format(V[i][0]), end = ' ')
		print(pi[i][0])