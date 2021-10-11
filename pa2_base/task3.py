import argparse, os, subprocess
def get_state_map(statesfile):
	file = open(statesfile, 'r')
	state_mapping = []
	while True:
		line = file.readline()
		if not line:
			break
		line = line.lstrip().rstrip()
		state_mapping.append(line)
	file.close()
	return state_mapping

def rem(file):
    if os.path.exists(file):
        os.remove(file)
    return

def generate_first_policy():
    global deldir
    deldir = 0
    if not os.path.exists('task3'):
        os.makedirs('task3')
        deldir = 1
    file = open(f'./task3/p{first_player}_policy0.txt', 'w')
    file.write(f'{first_player}\n')
    for i,state in enumerate(state_map[first_player-1]):
        file.write(f'{state} ')
        ind = -1
        for i in range(9):
            if state[i] == '0':
                ind = i
                break
        x = ['0']*9
        x[ind] = '1'
        file.write(' '.join(x) + '\n')
    file.close()
    return

def generate_next_policy(player, num_iter):
    policy = f'./task3/p{3-player}_policy{num_iter-1}.txt'
    print(f"Running iteration {num_iter}")
    cmd_encoder = "python","encoder.py","--policy",policy,"--states",statesfile[player-1]
    f = open('attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","attt_mdp"
    # print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    f = open(f'./task3/p{player}_policy{num_iter}.txt', 'w')
    cmd_decoder = "python","decoder.py","--value-policy","attt_planner","--states",statesfile[player-1] ,"--player-id",str(player)
    # print("\n","Generating the decoded policy file using decoder.py")
    subprocess.call(cmd_decoder,stdout = f)
    f.close()
    return
    # if you want to terminate when policies converge uncomment the below part and comment the return in above line
    # if num_iter <2 :
    #     return False
    # f = open('diff','w')
    # cmd_diff = "diff",f'./task3/p{player}_policy{num_iter-2}.txt',f'./task3/p{player}_policy{num_iter}.txt'
    # subprocess.call(cmd_diff, stdout = f)
    # f.close()
    # f = open('diff', 'r')
    # line = f.readline()
    # f.close()
    # os.remove('diff')
    # return not line

if __name__ == "__main__":
    global state_map, first_player, keep, statesfile, deldir
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1_states', type=str, default='./data/attt/states/states_file_p1.txt')
    parser.add_argument('--p2_states', type=str, default='./data/attt/states/states_file_p2.txt')
    parser.add_argument('--first_player', type=int, default = 2)
    parser.add_argument('--keep', type = int, default = 1)
    parser.add_argument('--iterations', type = int, default = 20)
    args = parser.parse_args()
    statesfile = []
    statesfile.append(args.p1_states)
    statesfile.append(args.p2_states)
    first_player = args.first_player
    keep = args.keep
    iterations = args.iterations
    state_map = []
    state_map.append(get_state_map(statesfile[0]))
    state_map.append(get_state_map(statesfile[1]))
    generate_first_policy()
    player = 3-first_player
    for i in range(1,iterations+1):
        generate_next_policy(player, i)
        player = 3 - player
    # if you want to terminate when policies converge, uncomment the below part and comment the above for loop
    # i = 1
    # while i <= iterations and not generate_next_policy(player, i):
    #     player = 3- player
    #     i += 1
    # if i <= iterations:
    #     print('Policies converged')
    # else:
    #     print('Policies not converged')
    
    rem('attt_mdp')
    rem('attt_planner')
    if keep == 0:
        player = first_player
        for i in range(iterations):
            rem(f'./task3/p{player}_policy{i}.txt')
            player = 3-player
        if deldir == 1:
            os.removedirs('task3')
    
