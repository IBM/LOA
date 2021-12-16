import sys
sys.path.append('../../src/meta_rule/')
sys.path.append('../../dd_lnn/')

import argparse
import pandas as pd
import numpy as np
from meta_interpretive import BasePredicate, MetaPredicate, MetaRule
import torch
import torch.optim as optim

def print_predicates_neurallp(obstacle4left, obstacle4right, obstacle4up, obstacle4down, \
                              target4left, target4right, target4up, target4down):
    params = obstacle4left.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go left  :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4left.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
    
    params = obstacle4right.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go right :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4right.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
        
    params = obstacle4up.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go up    :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4up.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
    
    params = obstacle4down.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go down  :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4down.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))

def print_predicates(obstacle4left, obstacle4right, obstacle4up, obstacle4down, \
                     target4left, target4right, target4up, target4down, \
                     left_rule, right_rule, up_rule, down_rule):
    params = obstacle4left.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go left  :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4left.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
    lnn_beta, lnn_wts, slacks = left_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))
    
    params = obstacle4right.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go right :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4right.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
    lnn_beta, lnn_wts, slacks = right_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))
        
    params = obstacle4up.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go up    :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4up.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
    lnn_beta, lnn_wts, slacks = up_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))
    
    params = obstacle4down.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go down  :::  No/Has-Obstacle (left/right/up/down): " + str(params))
    params = target4down.alpha.get_params().detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params))
    lnn_beta, lnn_wts, slacks = down_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))        

def print_predicates1(obstacle4left, obstacle4right, obstacle4up, obstacle4down, \
                      target4left, target4right, target4up, target4down, \
                      left_rule, right_rule, up_rule, down_rule):
    beta, params = obstacle4left.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go left  :::  No/Has-Obstacle (left/right/up/down): " + str(params) + " " + str(beta))
    beta, params = target4left.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params) + " " + str(beta))
    lnn_beta, lnn_wts, slacks = left_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))
    
    beta, params = obstacle4right.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go right :::  No/Has-Obstacle (left/right/up/down): " + str(params) + " " + str(beta))
    beta, params = target4right.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params) + " " + str(beta))
    lnn_beta, lnn_wts, slacks = right_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))
        
    beta, params = obstacle4up.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go up    :::  No/Has-Obstacle (left/right/up/down): " + str(params) + " " + str(beta))
    beta, params = target4up.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params) + " " + str(beta))
    lnn_beta, lnn_wts, slacks = up_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))
    
    beta, params = obstacle4down.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("Go down  :::  No/Has-Obstacle (left/right/up/down): " + str(params) + " " + str(beta))
    beta, params = target4down.alpha.get_params()
    params = params.detach().numpy()
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("              Has/No-Target (left/right/up/down): " + str(params) + " " + str(beta))
    lnn_beta, lnn_wts, slacks = down_rule.AND.cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("              LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))        

def genblocksworld(dim, obs):
    """ 
    Generates a grid of size dim X dim with obs obstacles and one
    target destination.  Origin (0,0) of the grid is in the top left
    corner. X dim increases towards the right and Y increases as we go
    down.

    Parameters:
    ----------
    dim : size of the world / grid. This function only generates squares.
    obs : number of obstacles

    Returns a bunch of predicates (pandas DataFrames)
    -------

    hasobstacle_left, hasobstacle_right, hasobstacle_up,
    hasobstacle_down :  
    Each is a dataframe with 2 attributes x, y. Each row lists a cell
    (x,y) for which the dataframe/predicate is true. E.g., if (3,2) is
    present in hasobstacle_left then that means (2,2) is an obstacle.

    hastarget_left/right/up/down :  
    Each is a dataframe with 2 attributes x, y. E.g. (3,2) in
    hastarget_left means the target destination has x coordinate < 3
    
    doesnothaveobstacle_left/right/up/down,
    doesnothavetarget_left/right/up/down : 
    These are the complements of the above predicates. Note that, its
    a bit complicated to implement negations in general. For instance,
    given hastarget_left, generating its complement
    doesnothavetarget_left would mean creating a predicate that
    contains (x,y) if it is not present in hastarget_left. This
    implies knowledge of the universe of (x,y) cells.  Because
    universe is only known in this function, I explicitly generate
    these  complement/negated predicates here.

    """

    #generating the universe of x, y cells
    cells = [(x,y) for x in range(dim) for y in range(dim)]

    #generating obstacles and target: this is achieved in two steps:
    # - randomly choose (without replacement) obs+1 cells from the universe
    # - generate an int between 0 and obs (both inclusive), call this i
    #   - pick ith randomly chosen cell as target
    rng = np.random.default_rng()    
    obstacles = rng.choice(cells, size=obs+1, replace=False).tolist()

    target_pos = rng.integers(obs+1)
    
    target = obstacles[target_pos]
    if target_pos > 0 and target_pos < obs:
        o = obstacles[0:target_pos] + obstacles[target_pos+1:len(obstacles)]
    elif target_pos == 0:
        o = obstacles[1:len(obstacles)]
    else:
        o = obstacles[0:target_pos]
    obstacles = o
        
    print("Obstacles: " + str(obstacles))
    print("Target=" + str(target))

    #to generate the relevant predicates we first build a DataFrame with all cells in the universe
    #with appropriate cols. for instance, since we want to return a predicate hastarget_left,
    #we first populate a col hastarget_left. For each cell (x,y) this col is 1 if the target
    #is to the left of (x,y).
    col_names = ["x", "y", \
                 "hasobstacle_left", "hasobstacle_right", "hasobstacle_up", "hasobstacle_down", \
                 "hastarget_left", "hastarget_right", "hastarget_up", "hastarget_down", \
                 "go_left", "go_right", "go_up", "go_down"]
    rows = []
    for x,y in cells:
        #the next few lines assume edges of the grid are obstacles

        #col obstacle_left is true if we are at the left edge (i.e., x is 0)
        #or the cell to the immediate left is an obstacle
        hasobstacle_left = 1 if x-1 < 0 or any([True for o in obstacles if o[0]==x-1 and o[1]==y]) else 0

        #col obstacle_right is true if we are at the right edge (i.e., x is dim-1)
        #or the cell to the immediate right is an obstacle
        hasobstacle_right = 1 if x+1 >= dim or any([True for o in obstacles if o[0]==x+1 and o[1]==y]) else 0

        #col obstacle_up is true if we are at the top edge (i.e., y == 0)
        #or the cell immediately above is an obstacle
        hasobstacle_up = 1 if y-1 < 0 or any([True for o in obstacles if o[0]==x and o[1]==y-1]) else 0

        #col obstacle_down is true if we are at the bottom edge (i.e., y == dim-1)
        #or the cell immediately below is an obstacle
        hasobstacle_down = 1 if y+1 >= dim or any([True for o in obstacles if o[0]==x and o[1]==y+1]) else 0

        #col hastarget_left is true if the target is to the left of us (in the strict inequality sense)
        #one way to express this is to simply say x > target.x
        #what i'm doing here seems to be the more complicated way, which is to compare
        #distance with the target in the x direction after moving to the immediate left cell
        hastarget_left = 1 if np.fabs(x-target[0]) > np.fabs(x-1-target[0]) else 0

        #col hastarget_right is true if the target is to the right of us (in the strict inequality sense)
        #compare distance with the target in the x direction after moving to the immediate right cell
        hastarget_right = 1 if np.fabs(x-target[0]) > np.fabs(x+1-target[0]) else 0

        #col hastarget_up is true if the target is above us (in the strict inequality sense)
        #compare distance with the target in the y direction after moving to the cell immediately above us
        hastarget_up = 1 if np.fabs(y-target[1]) > np.fabs(y-1-target[1]) else 0

        #col hastarget_down is true if the target is below us (in the strict inequality sense)
        #compare distance with the target in the y direction after moving to the cell immediately below us
        hastarget_down = 1 if np.fabs(y-target[1]) > np.fabs(y+1-target[1]) else 0
        
        #not sure about this, but i don't add obstacles or the target to the dataframe
        if not (target[0]==x and target[1]==y) and not any([True for o in obstacles if o[0]==x and o[1]==y]):
            rows.append(pd.Series([x, y, \
                                   hasobstacle_left, hasobstacle_right, hasobstacle_up, hasobstacle_down, \
                                   hastarget_left, hastarget_right, hastarget_up, hastarget_down, \
                                   -2 if hasobstacle_left==1 else 2*hastarget_left-1, \
                                   -2 if hasobstacle_right==1 else 2*hastarget_right-1, \
                                   -2 if hasobstacle_up==1 else 2*hastarget_up-1,
                                   -2 if hasobstacle_down==1 else 2*hastarget_down-1], \
                                  index=col_names))
    df = pd.DataFrame(rows, columns=col_names)
    #turn this stmt on if you want to see the dataframe we built
    #print(df.to_markdown())

    #since we have built a dataframe containing for all relevant cells all required cols, its now
    #easy to generate our predicates. just project on the relevant cols, rename them (if needed)
    #and reset index.
    go_left = df[['x', 'y', 'go_left']].rename(columns={"go_left": "Label"}).reset_index(drop=True)
    go_right = df[['x', 'y', 'go_right']].rename(columns={"go_right": "Label"}).reset_index(drop=True)
    go_up = df[['x', 'y', 'go_up']].rename(columns={"go_up": "Label"}).reset_index(drop=True)
    go_down = df[['x', 'y', 'go_down']].rename(columns={"go_down": "Label"}).reset_index(drop=True)
    hasobstacle_left = df[df["hasobstacle_left"] == 1][['x', 'y']].reset_index(drop=True)
    hasobstacle_right = df[df["hasobstacle_right"] == 1][['x', 'y']].reset_index(drop=True)
    hasobstacle_up = df[df["hasobstacle_up"] == 1][['x', 'y']].reset_index(drop=True)
    hasobstacle_down = df[df["hasobstacle_down"] == 1][['x', 'y']].reset_index(drop=True)
    hastarget_left = df[df["hastarget_left"] == 1][['x', 'y']].reset_index(drop=True)
    hastarget_right = df[df["hastarget_right"] == 1][['x', 'y']].reset_index(drop=True)
    hastarget_up = df[df["hastarget_up"] == 1][['x', 'y']].reset_index(drop=True)
    hastarget_down = df[df["hastarget_down"] == 1][['x', 'y']].reset_index(drop=True)
    doesnothaveobstacle_left = df[df["hasobstacle_left"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothaveobstacle_right = df[df["hasobstacle_right"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothaveobstacle_up = df[df["hasobstacle_up"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothaveobstacle_down = df[df["hasobstacle_down"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothavetarget_left = df[df["hastarget_left"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothavetarget_right = df[df["hastarget_right"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothavetarget_up = df[df["hastarget_up"] == 0][['x', 'y']].reset_index(drop=True)
    doesnothavetarget_down = df[df["hastarget_down"] == 0][['x', 'y']].reset_index(drop=True)

    return go_left, go_right, go_up, go_down, \
        hasobstacle_left, hasobstacle_right, hasobstacle_up, hasobstacle_down, \
        hastarget_left, hastarget_right, hastarget_up, hastarget_down, \
        doesnothaveobstacle_left, doesnothaveobstacle_right, doesnothaveobstacle_up, doesnothaveobstacle_down, \
        doesnothavetarget_left, doesnothavetarget_right, doesnothavetarget_up, doesnothavetarget_down


def main():
    #example invocation: python3 blocksworld.py -n 5 -o 5 -a 0.8
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", required=True, help="number of cells in blocksworld")
    parser.add_argument("-o", "--obstacles", required=True, help="number of obstacles in blocksworld")
    parser.add_argument("-a", "--alpha", required=True, help="LNN hyperparameter alpha")
    
    cl_args = parser.parse_args()
    dim = int(cl_args.size)
    obs = int(cl_args.obstacles)
    a = float(cl_args.alpha)

    option = 1

    old_obstacle4left = None
    old_obstacle4right = None
    old_obstacle4up = None
    old_obstacle4down = None
    old_target4left = None
    old_target4right = None
    old_target4up = None
    old_target4down = None
    for i in range(50): #500
        go_left, go_right, go_up, go_down, \
            left_hasobstacle, right_hasobstacle, up_hasobstacle, down_hasobstacle, \
            left_hastarget, right_hastarget, up_hastarget, down_hastarget, \
            left_noobstacle, right_noobstacle, up_noobstacle, down_noobstacle, \
            left_notarget, right_notarget, up_notarget, down_notarget \
            = genblocksworld(dim, obs)

        #the basic rule structure is:
        #       obstacle pred \wedge target pred -> go pred
        #problem is we don't know which obstacle predicate is relevant to moving
        #in which direction and similary, we don't know which target predicate
        #is relevant to moving in which direction. this is the learning task.
        #so we create all possible obstacles predicates and their negations,
        #and encapsulate these in a MetaPredicate to express a choice over these.
        #similarly, we create all possible target predicates and their negations
        #to encapsulate them in a MetaPredicate to express a choice. lastly,
        #we use a MetaRule object to express a conjunction between the chosen
        #obstacle predicate (or its negation) and the chosen target predicate
        #(or its negation).
        obstacle4left = MetaPredicate([BasePredicate(left_noobstacle) \
                                       , BasePredicate(right_noobstacle)\
                                       , BasePredicate(up_noobstacle) \
                                       , BasePredicate(down_noobstacle) \
                                       , BasePredicate(left_hasobstacle) \
                                       , BasePredicate(right_hasobstacle) \
                                       , BasePredicate(up_hasobstacle) \
                                       , BasePredicate(down_hasobstacle)], option)
        obstacle4left.alpha = old_obstacle4left.alpha if old_obstacle4left else obstacle4left.alpha
        target4left = MetaPredicate([BasePredicate(left_hastarget) \
                                     , BasePredicate(right_hastarget) \
                                     , BasePredicate(up_hastarget) \
                                     , BasePredicate(down_hastarget)
                                     , BasePredicate(left_notarget) \
                                     , BasePredicate(right_notarget) \
                                     , BasePredicate(up_notarget) \
                                     , BasePredicate(down_notarget)], option)
        target4left.alpha = old_target4left.alpha if old_target4left else target4left.alpha
        left_rule = MetaRule([obstacle4left, target4left], [[['x', 'y'], ['x', 'y']]], a, False, option)

        obstacle4right = MetaPredicate([BasePredicate(left_noobstacle) \
                                        , BasePredicate(right_noobstacle)\
                                        , BasePredicate(up_noobstacle) \
                                        , BasePredicate(down_noobstacle) \
                                        , BasePredicate(left_hasobstacle) \
                                        , BasePredicate(right_hasobstacle) \
                                        , BasePredicate(up_hasobstacle) \
                                        , BasePredicate(down_hasobstacle)], option)
        obstacle4right.alpha = old_obstacle4right.alpha if old_obstacle4right else obstacle4right.alpha
        target4right = MetaPredicate([BasePredicate(left_hastarget) \
                                      , BasePredicate(right_hastarget) \
                                      , BasePredicate(up_hastarget) \
                                      , BasePredicate(down_hastarget)
                                      , BasePredicate(left_notarget) \
                                      , BasePredicate(right_notarget) \
                                      , BasePredicate(up_notarget) \
                                      , BasePredicate(down_notarget)], option)
        target4right.alpha = old_target4right.alpha if old_target4right else target4right.alpha
        right_rule = MetaRule([obstacle4right, target4right], [[['x', 'y'], ['x', 'y']]], a, False, option)

        obstacle4up = MetaPredicate([BasePredicate(left_noobstacle) \
                                     , BasePredicate(right_noobstacle)\
                                     , BasePredicate(up_noobstacle) \
                                     , BasePredicate(down_noobstacle) \
                                     , BasePredicate(left_hasobstacle) \
                                     , BasePredicate(right_hasobstacle) \
                                     , BasePredicate(up_hasobstacle) \
                                     , BasePredicate(down_hasobstacle)], option)
        obstacle4up.alpha = old_obstacle4up.alpha if old_obstacle4up else obstacle4up.alpha
        target4up = MetaPredicate([BasePredicate(left_hastarget) \
                                   , BasePredicate(right_hastarget) \
                                   , BasePredicate(up_hastarget) \
                                   , BasePredicate(down_hastarget)
                                   , BasePredicate(left_notarget) \
                                   , BasePredicate(right_notarget) \
                                   , BasePredicate(up_notarget) \
                                   , BasePredicate(down_notarget)], option)
        target4up.alpha = old_target4up.alpha if old_target4up else target4up.alpha
        up_rule = MetaRule([obstacle4up, target4up], [[['x', 'y'], ['x', 'y']]], a, False, option)

        obstacle4down = MetaPredicate([BasePredicate(left_noobstacle) \
                                       , BasePredicate(right_noobstacle)\
                                       , BasePredicate(up_noobstacle) \
                                       , BasePredicate(down_noobstacle) \
                                       , BasePredicate(left_hasobstacle) \
                                       , BasePredicate(right_hasobstacle) \
                                       , BasePredicate(up_hasobstacle) \
                                       , BasePredicate(down_hasobstacle)], option)
        obstacle4down.alpha = old_obstacle4down.alpha if old_obstacle4down else obstacle4down.alpha
        target4down = MetaPredicate([BasePredicate(left_hastarget) \
                                     , BasePredicate(right_hastarget) \
                                     , BasePredicate(up_hastarget) \
                                     , BasePredicate(down_hastarget)
                                     , BasePredicate(left_notarget) \
                                     , BasePredicate(right_notarget) \
                                     , BasePredicate(up_notarget) \
                                     , BasePredicate(down_notarget)], option)
        target4down.alpha = old_target4down.alpha if old_target4down else target4down.alpha
        down_rule = MetaRule([obstacle4down, target4down], [[['x', 'y'], ['x', 'y']]], a, False, option)

        left_rule.df['id'] = list(left_rule.df.index)
        yl = torch.FloatTensor(go_left.merge(left_rule.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)
        left_rule.df.drop(['id'], axis=1, inplace=True)
        right_rule.df['id'] = list(right_rule.df.index)
        yr = torch.FloatTensor(go_right.merge(right_rule.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)
        right_rule.df.drop(['id'], axis=1, inplace=True)
        up_rule.df['id'] = list(up_rule.df.index)
        yu = torch.FloatTensor(go_up.merge(up_rule.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)
        up_rule.df.drop(['id'], axis=1, inplace=True)
        down_rule.df['id'] = list(down_rule.df.index)
        yd = torch.FloatTensor(go_down.merge(down_rule.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)        
        down_rule.df.drop(['id'], axis=1, inplace=True)
        y = torch.cat((yl, yr, yu, yd), 1)

        #we will be learning rules to move in all 4 directions simultaeneously
        rewards = []
        optimizer = optim.Adam([{'params': left_rule.parameters()}, \
                                {'params': right_rule.parameters()}, \
                                {'params': up_rule.parameters()}, \
                                {'params': down_rule.parameters()}] , lr=0.01)

        x = torch.arange(len(left_rule.df.index))
        for iter in range(10):
            left_rule.train()
            right_rule.train()
            up_rule.train()
            down_rule.train()
            optimizer.zero_grad()

            left_yhat, slacks = left_rule(x)
            right_yhat, slacks = right_rule(x)
            up_yhat, slacks = up_rule(x)
            down_yhat, slacks = down_rule(x)
            unnorm_probs = torch.cat((left_yhat, right_yhat, up_yhat, down_yhat), 1)
            probs = torch.log(torch.div(unnorm_probs, torch.sum(unnorm_probs, 0)))
            mean_rewards = torch.sum(torch.mul(y, probs), 0)
            loss = -mean_rewards.mean()
            rewards.append(-loss.item())

            loss.backward()
            optimizer.step()

        np.set_printoptions(precision=3, suppress=True)
        print("Epoch " + str(i) + " rewards: " + str(np.around(rewards[0], decimals=3)) \
              + " -> " + str(np.around(rewards[len(rewards)-1], decimals=3)))

        #printing the learned parameters so far
        #print_predicates(obstacle4left, obstacle4right, obstacle4up, obstacle4down, \
        #                 target4left, target4right, target4up, target4down, \
        #                 left_rule, right_rule, up_rule, down_rule)
        if option == 0:
            print_predicates1(obstacle4left, obstacle4right, obstacle4up, obstacle4down, \
                              target4left, target4right, target4up, target4down, \
                              left_rule, right_rule, up_rule, down_rule)
        else:
            print_predicates_neurallp(obstacle4left, obstacle4right, obstacle4up, obstacle4down, \
                                      target4left, target4right, target4up, target4down)
        
        old_obstacle4left = obstacle4left
        old_obstacle4right = obstacle4right
        old_obstacle4up = obstacle4up
        old_obstacle4down = obstacle4down
        old_target4left = target4left
        old_target4right = target4right
        old_target4up = target4up
        old_target4down = target4down

    rewards = []
    for i in range(50): #500
        go_left, go_right, go_up, go_down, \
            left_hasobstacle, right_hasobstacle, up_hasobstacle, down_hasobstacle, \
            left_hastarget, right_hastarget, up_hastarget, down_hastarget, \
            left_noobstacle, right_noobstacle, up_noobstacle, down_noobstacle, \
            left_notarget, right_notarget, up_notarget, down_notarget \
            = genblocksworld(dim, obs)

        obstacle4left_test = MetaPredicate([BasePredicate(left_noobstacle) \
                                       , BasePredicate(right_noobstacle)\
                                       , BasePredicate(up_noobstacle) \
                                       , BasePredicate(down_noobstacle) \
                                       , BasePredicate(left_hasobstacle) \
                                       , BasePredicate(right_hasobstacle) \
                                       , BasePredicate(up_hasobstacle) \
                                            , BasePredicate(down_hasobstacle)], option)
        obstacle4left_test.alpha = obstacle4left.alpha 
        target4left_test = MetaPredicate([BasePredicate(left_hastarget) \
                                     , BasePredicate(right_hastarget) \
                                     , BasePredicate(up_hastarget) \
                                     , BasePredicate(down_hastarget)
                                     , BasePredicate(left_notarget) \
                                     , BasePredicate(right_notarget) \
                                     , BasePredicate(up_notarget) \
                                     , BasePredicate(down_notarget)], option)
        target4left_test.alpha = target4left.alpha
        left_rule_test = MetaRule([obstacle4left_test, target4left_test], [[['x', 'y'], ['x', 'y']]], a, False, option)

        obstacle4right_test = MetaPredicate([BasePredicate(left_noobstacle) \
                                        , BasePredicate(right_noobstacle)\
                                        , BasePredicate(up_noobstacle) \
                                        , BasePredicate(down_noobstacle) \
                                        , BasePredicate(left_hasobstacle) \
                                        , BasePredicate(right_hasobstacle) \
                                        , BasePredicate(up_hasobstacle) \
                                        , BasePredicate(down_hasobstacle)], option)
        obstacle4right_test.alpha = obstacle4right.alpha
        target4right_test = MetaPredicate([BasePredicate(left_hastarget) \
                                      , BasePredicate(right_hastarget) \
                                      , BasePredicate(up_hastarget) \
                                      , BasePredicate(down_hastarget)
                                      , BasePredicate(left_notarget) \
                                      , BasePredicate(right_notarget) \
                                      , BasePredicate(up_notarget) \
                                      , BasePredicate(down_notarget)], option)
        target4right_test.alpha = target4right.alpha
        right_rule_test = MetaRule([obstacle4right_test, target4right_test], [[['x', 'y'], ['x', 'y']]], a, False, option)

        obstacle4up_test = MetaPredicate([BasePredicate(left_noobstacle) \
                                     , BasePredicate(right_noobstacle)\
                                     , BasePredicate(up_noobstacle) \
                                     , BasePredicate(down_noobstacle) \
                                     , BasePredicate(left_hasobstacle) \
                                     , BasePredicate(right_hasobstacle) \
                                     , BasePredicate(up_hasobstacle) \
                                     , BasePredicate(down_hasobstacle)], option)
        obstacle4up_test.alpha = obstacle4up.alpha
        target4up_test = MetaPredicate([BasePredicate(left_hastarget) \
                                   , BasePredicate(right_hastarget) \
                                   , BasePredicate(up_hastarget) \
                                   , BasePredicate(down_hastarget)
                                   , BasePredicate(left_notarget) \
                                   , BasePredicate(right_notarget) \
                                   , BasePredicate(up_notarget) \
                                   , BasePredicate(down_notarget)], option)
        target4up_test.alpha = target4up.alpha
        up_rule_test = MetaRule([obstacle4up_test, target4up_test], [[['x', 'y'], ['x', 'y']]], a, False, option)

        obstacle4down_test = MetaPredicate([BasePredicate(left_noobstacle) \
                                       , BasePredicate(right_noobstacle)\
                                       , BasePredicate(up_noobstacle) \
                                       , BasePredicate(down_noobstacle) \
                                       , BasePredicate(left_hasobstacle) \
                                       , BasePredicate(right_hasobstacle) \
                                       , BasePredicate(up_hasobstacle) \
                                       , BasePredicate(down_hasobstacle)], option)
        obstacle4down_test.alpha = obstacle4down.alpha
        target4down_test = MetaPredicate([BasePredicate(left_hastarget) \
                                     , BasePredicate(right_hastarget) \
                                     , BasePredicate(up_hastarget) \
                                     , BasePredicate(down_hastarget)
                                     , BasePredicate(left_notarget) \
                                     , BasePredicate(right_notarget) \
                                     , BasePredicate(up_notarget) \
                                     , BasePredicate(down_notarget)], option)
        target4down_test.alpha = target4down.alpha
        down_rule_test = MetaRule([obstacle4down_test, target4down_test], [[['x', 'y'], ['x', 'y']]], a, False, option)

        left_rule_test.df['id'] = list(left_rule_test.df.index)
        yl = torch.FloatTensor(go_left.merge(left_rule_test.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)
        left_rule_test.df.drop(['id'], axis=1, inplace=True)
        right_rule_test.df['id'] = list(right_rule_test.df.index)
        yr = torch.FloatTensor(go_right.merge(right_rule_test.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)
        right_rule_test.df.drop(['id'], axis=1, inplace=True)
        up_rule_test.df['id'] = list(up_rule_test.df.index)
        yu = torch.FloatTensor(go_up.merge(up_rule_test.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)
        up_rule_test.df.drop(['id'], axis=1, inplace=True)
        down_rule_test.df['id'] = list(down_rule_test.df.index)
        yd = torch.FloatTensor(go_down.merge(down_rule_test.df, on=["x", "y"], how='right').sort_values('id')[["Label"]].values)        
        down_rule_test.df.drop(['id'], axis=1, inplace=True)
        y = torch.cat((yl, yr, yu, yd), 1)

        for iter in range(10):
            left_yhat, slacks = left_rule_test(x)
            right_yhat, slacks = right_rule_test(x)
            up_yhat, slacks = up_rule_test(x)
            down_yhat, slacks = down_rule_test(x)
            unnorm_probs = torch.cat((left_yhat, right_yhat, up_yhat, down_yhat), 1)
            probs = torch.log(torch.div(unnorm_probs, torch.sum(unnorm_probs, 0)))
            mean_rewards = torch.sum(torch.mul(y, probs), 0)
            loss = -mean_rewards.mean()
            rewards.append(-loss.item())

        print(rewards)

if __name__ == "__main__":
    main()
