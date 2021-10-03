import argparse
import numpy as np

np.random.seed(0)

BLANK = 0
grid = np.zeros((3,3), dtype=int)

p1_policy = {}
p2_policy = {}
indices = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

auto_p1 = False
auto_p2 = False

def get_args():
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('-p1', help='Policy file for player 1')
    parser.add_argument('-p2', help='Policy file for player 2')
    return vars(parser.parse_args())

def rep(grid):
    l = []
    for x,y in indices:
        l.append(str(grid[x][y]))
    return "".join(l)

def get_policy(filepath):
    policy = {}
    with open(filepath,'r') as file:
        lines = file.readlines()
    for line in lines:
        line_split = line.split()
        policy[line_split[0]] = list(map(float,line_split[1:]))
    return policy

def end_game_cond(grid):
    for i in range(3):
        if grid[i][0]==grid[i][1] and grid[i][1] == grid[i][2] and grid[i][0] != BLANK:
            return 1
    for j in range(3):
        if grid[0][j]==grid[1][j] and grid[1][j] == grid[2][j] and grid[0][j] != BLANK:
            return 1
    if grid[0][0]==grid[1][1] and grid[1][1] == grid[2][2] and grid[1][1] != BLANK:
        return 1
    if grid[2][0]==grid[1][1] and grid[1][1] == grid[0][2] and grid[1][1] != BLANK:
        return 1
    if (grid == BLANK).sum() == 0:
        return 2
    return 0

def get_plist(v):
    l = []
    p = []
    for i,x in enumerate(v):
        if x>0:
            l.append(i)
            p.append(x)
    return l,p

def p1():
    global grid, indices
    DrawGrid()
    s = rep(grid)
    choice = 0
    if auto_p1:
        p = p1_policy[s]
        l,p = get_plist(p)
        x = np.random.choice(l,1,p)
        choice = x[0]
    else:
        print("Enter choice for player 1: (1:9)")
        while True:
            inp = input()
            if inp.isnumeric():
                i = int(inp)
                if i>0 and i <10:
                    x,y = indices[i-1]
                    if grid[x][y] == BLANK:
                        choice = i-1
                        break
            print("Enter correct grid cell from 1:9")

    x,y = indices[choice]
    grid[x][y] = 1
    res = end_game_cond(grid)
    if res == 0:
        return p2()
    elif res == 1:
        return 2
    else:
        return 0

def p2():
    global grid, indices
    DrawGrid()
    s = rep(grid)
    choice = 0
    if auto_p2:
        p = p2_policy[s]
        l,p = get_plist(p)
        x = np.random.choice(l,1,p)
        choice = x[0]
    else:
        print("Enter choice for player 2: (1:9)")
        while True:
            inp = input()
            if inp.isnumeric():
                i = int(inp)
                if i>0 and i <10:
                    x,y = indices[i-1]
                    if grid[x][y] == BLANK:
                        choice = i-1
                        break
            print("Enter correct grid cell from 1:9")

    x,y = indices[choice]
    grid[x][y] = 2
    res = end_game_cond(grid)
    if res == 0:
        return p1()
    elif res == 1:
        return 1
    else:
        return 0


def cell_value(i):
    if i == 1:
        return '1'
    elif i == 2:
        return '2'
    else:
        return ' '

def DrawGrid():    
    global grid
    print()
    print("     |     |     ")  
    print(f"  {cell_value(grid[0][0])}  |  {cell_value(grid[0][1])}  |  {cell_value(grid[0][2])}  ")    
    print("_____|_____|_____")    
    print("     |     |     ")  
    print(f"  {cell_value(grid[1][0])}  |  {cell_value(grid[1][1])}  |  {cell_value(grid[1][2])}  ")    
    print("_____|_____|_____")      
    print("     |     |     ")  
    print(f"  {cell_value(grid[2][0])}  |  {cell_value(grid[2][1])}  |  {cell_value(grid[2][2])}  ")    
    print("     |     |     ")  
    print()

if __name__ == '__main__':
    args = get_args()
    
    if args["p1"] is not None:
        p1_policy = get_policy(args["p1"])
        auto_p1 = True
    if args["p2"] is not None:
        print(args["p2"])
        p2_policy = get_policy(args["p2"])
        auto_p2 = True

    # p2_policy = get_policy('unif_rand_p2_policy')
    # auto_p2=True
    print("Anti-Tic-Tac-Toe Game")    
    print("Player 1 --- Player 2")    

    result = p1()

    DrawGrid()
    if result == 1:
        print("Player 1 wins")
    elif result == 2:
        print("Player 2 wins")
    else:
        print("Draw game")
