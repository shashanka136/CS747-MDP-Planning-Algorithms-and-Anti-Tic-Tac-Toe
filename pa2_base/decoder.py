import numpy as np
import argparse

def get_state_map(statesfile):
	global state_map
	file = open(statesfile, 'r')
	state_map = []
	while True:
		line = file.readline()
		if not line:
			break
		line = line.lstrip().rstrip()
		state_map.append(line)
	file.close()
	return

def print_policy(value_policy_file):
	global state_map
	file = open(value_policy_file, 'r')
	for i, state in enumerate(state_map):
		pi = int(file.readline().lstrip().rstrip().split()[1])
		print(state, end = ' ')
		x = ['0']*9
		x[pi] = '1'
		print(' '.join(x))
	return

if __name__ == '__main__':
	global state_map
	parser = argparse.ArgumentParser()
	parser.add_argument('--value-policy', dest = 'vp', type = str)
	parser.add_argument('--player-id', dest = 'player_id', type = str)
	parser.add_argument('--states', type = str)
	args = parser.parse_args()
	value_policy_file = args.vp
	states_file = args.states
	player_id = args.player_id
	print(player_id)
	get_state_map(states_file)
	print_policy(value_policy_file)
