import sys
sys.path.append('../../src/meta_rule/')
sys.path.append('../../dd_lnn/')

from lnn_operators import and_lukasiewicz
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    #this is a hyperparameter
    alpha = 0.9

    #we will be learning a conjunction op with 2 args
    op = and_lukasiewicz(alpha, 2, False)

    #op's cdd member variable allows access to the LNN conjunction
    #parameters
    beta, argument_wts, slacks = op.cdd()
    print("beta (pre-training): " + str(beta.item()))
    print("argument weights (pre-training): " + str(argument_wts.detach()))

    #to train a conjunction operator we need the conjunction truth table
    x = torch.from_numpy(np.array([[0, 0], \
                                   [0, 1], \
                                   [1, 0], \
                                   [1, 1]])).float()

    #we can check what op returns before we train it
    output, slacks = op(x)

    #the target values for each row in the truth table (conjunction)
    y = torch.from_numpy(np.array([[0], \
                                   [0], \
                                   [0], \
                                   [1]])).float()

    #we will use binary cross entropy loss and adam to train
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(op.parameters(), lr=0.1)

    for iter in range(10):
        op.train()
        optimizer.zero_grad()

        yhat, slacks = op(x)
        loss = loss_fn(yhat, y)

        print("Iteration " + str(iter) + ": " + str(loss.item()))
        loss.backward()
        optimizer.step()
    #the loss values should show a decreasing trend

    #check to see output of op post-training
    preds, slacks = op(x)
    check_values = torch.cat((preds, y), 1)
    print("Checking values produced by learned conjunction: ")
    print("(left col is computed value, right col is ground truth)")
    print(check_values.detach())
    
    #lets check the LNN conjunction parameters post-training
    #do these look different from the pre-training settings?
    beta, argument_wts, slacks = op.cdd()
    print("beta (post-training): " + str(beta.item()))
    print("argument weights (post-training): " + str(argument_wts.detach()))

    ####### Advanced #########
    #we can check if the learned parameters satisfy the LNN constraints
    #read on if you want to see how to do this

    #first lets see what the constraints are
    #once again, we need to access op's cdd member variable
    A, b = op.cdd.make_constraints(alpha, 2)
    #A, b represent the constraints in canonical form s.t.
    #         A [beta, argument_weights]' <= b
    print("A (LHS):")
    print(A)
    print("b (RHS):")
    print(b)

    #let's check if the learned parameters satisfy the constraints
    check_constraints = A.dot(np.append(beta.item(), argument_wts.detach()))

    #if every number in the left col is <= the corresponding number in
    #the right col then all constraints are satisfied    
    check_constraints = np.hstack((check_constraints.reshape((-1, 1)), b))
    print("Check constraints: (row[0] should be <= row[1])")
    print(check_constraints)

    #an interesting excercise left to the reader would be to check
    #whether the constraints hold for the parameters pre-training. you
    #should find that they do indeed hold. the code in
    #lnn_operators.py and cdd_interface.py ensure that constraints are
    #always satisfied.
    
    
if __name__ == "__main__":
    main()

