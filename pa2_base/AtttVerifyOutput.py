import random,argparse,sys,subprocess,os

from numpy import cdouble
parser = argparse.ArgumentParser()

def run(states,policy,player):
    cmd_encoder = "python","encoder.py","--policy",policy,"--states",states
    print("\n","Generating the MDP encoding using encoder.py")
    f = open('verify_attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","verify_attt_mdp"
    print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('verify_attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy","verify_attt_planner","--states",states ,"--player-id",str(player)
    print("\n","Generating the decoded policy file using decoder.py")
    cmd_output = subprocess.check_output(cmd_decoder,universal_newlines=True)

    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')
    return cmd_output

def verifyOutput(states, output, player):
    output = output.split('\n')
    if output[0] != player:
        print("\n","*"*10,f"Mistake: First line of policy file should be the player id, i.e.'{player}'")
        sys.exit()
    output.remove('')
    with open(states,'r') as file:
        lines = file.readlines()
    states = [line.strip() for line in lines]
    if len(output)-1 != len(states):
        print("\n","*"*10,f"Mistake: Expected {len(states)} policy lines, got {len(output)-1}")
        sys.exit()
    
    policy_states=[]
    for idx,out in enumerate(output[1:]):
        terms = out.split(' ')
        if len(terms) !=10:
            print("\n","*"*10,f"Mistake: In line {idx+2}, expected 10 terms , got {len(terms)}. {out}")
            sys.exit()
        policy_states.append(terms[0])
        try:
            p = list(map(float,terms[1:]))
        except:
            print("\n","*"*10,f"Mistake: In line {idx+2}, Number format excpetion. {out}")
            sys.exit()
    
    states_intersection = set(states).intersection(set(policy_states))
    if len(states_intersection) != len(states):
        print("\n","*"*10,f"Mistake: States in policy file and input states file do not match")
        sys.exit()
    
    print("OK")

def getPlayerId(policy):
    with open(policy,'r') as file:
        line = file.readline()
    opponent_player = line.strip()
    if opponent_player=='1':
        player = '2'
    else:
        player = '1'
    return player

if __name__ == '__main__':
    parser.add_argument("--states",required=True,type=str,help="File with valid states of the player")
    parser.add_argument("--policy",required=True,type=str,help="Policy file of the opponent player")
    args = parser.parse_args()
    player = getPlayerId(args.policy)
    output = run(args.states,args.policy,player)
    verifyOutput(args.states,output,player)


