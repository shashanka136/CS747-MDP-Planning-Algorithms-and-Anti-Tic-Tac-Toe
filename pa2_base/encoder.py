import argparse

def terminal(state): # check if someone losing or winning
	#don't check that if it's a draw
	for i in range(3):
		if state[3+i] == state[i] ==state[6+i] and state[i] != '0':
			return True
		if state[3*i+1] == state[3*i] ==state[3*i+2] and state[3*i] != '0':
			return True
	
	if state[0] == state[4] ==state[8] and state[0] != '0':
		return True
	if state[2] == state[4] ==state[6] and state[2] != '0':
		return True
	return False

def add_possible_transitions(state, action):
	assert not terminal(state), 'states file is incorrect, contains a terminal state'
	assert '0' in state, 'states file is incorrect, contains a terminal state'
	global state_map, rev_state_map, player_id, player_id2, transitions, probs
	num1 = rev_state_map[state]
	if state[action] != '0':
		num2 = len(state_map)
		transitions.append(" ".join([str(x) for x in [num1, str(action), num2, -100.00, 1.00]]))
		return
	state = state[:action] + player_id + state[action+1:]
	loss = terminal(state)
	if loss or (not '0' in state):
		num2 = len(state_map)
		rew = 0.0
		# rew = -1.0 if loss else 0.0
		transitions.append(" ".join([str(x) for x in [num1, str(action), num2, rew, 1.00]]))
		return

	assert state in probs, f'There should be a policy for {state} for player{player_id2}'
	
	for i, prob in enumerate(probs[state]):
		if prob == 0.0 or state[i] != '0':
			continue
		temp_state = state[:i] + player_id2 + state[i+1:]
		win = terminal(temp_state)
		rew = 1.0 if win else 0.0
		num2 = len(state_map) if win or (not '0' in temp_state) else rev_state_map[temp_state]
		transitions.append(" ".join([str(x) for x in [num1, str(action), num2, rew, '{:.20f}'.format(prob)]]))
	return
		

def get_state_map(statesfile):
	global state_map, rev_state_map
	file = open(statesfile, 'r')
	state_map = []
	rev_state_map = {}
	while True:
		line = file.readline()
		if not line:
			break
		line = line.lstrip().rstrip()
		rev_state_map[line] = len(state_map)
		state_map.append(line)
	file.close()
	return

def get_transitions(policyfile):
	global state_map, rev_state_map, player_id, player_id2, probs
	file = open(policyfile, 'r')
	player_id2 = file.readline().lstrip().rstrip()
	player_id = str(3-int(player_id2))
	probs = {}
	while True:
		line = file.readline()
		if not line:
			break
		line = line.lstrip().rstrip().split()
		probs[line[0]] = [float(x) for x in line[1:]]
	
	for state in state_map:
		for action in range(9):
			add_possible_transitions(state, action)
	file.close()
	pass

if __name__ == "__main__":
	global state_map, rev_state_map, player_id, player_id2, probs, transitions
	parser = argparse.ArgumentParser()
	parser.add_argument('--policy', type = str)
	parser.add_argument('--states', type = str)
	args = parser.parse_args()
	policyfile = args.policy
	statesfile = args.states
	transitions = []
	get_state_map(statesfile)
	get_transitions(policyfile)
	print(f'numStates {len(state_map)+1}')
	print('numActions 9')
	print('end', end = ' ')
	print(len(state_map))
	for transition in transitions:
		print(f'transition {transition}')
	print('mdptype episodic')
	print('discount 1.0')
	