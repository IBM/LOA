
import numpy as np
def make_lnn_constraints(alpha, arity):
    #1st col is beta, ws are in the following cols
    #constructing Ax <= b; A is LHS, b is RHS

    #generating A/LHS first
    #non-negativity constraints for weights
    A = np.append(np.zeros((arity, 1)), \
                    np.diag(-np.ones(arity)), \
                    axis=1)

    #equation (1)/(3) in LNN draft
    equ_1_lhs = np.append(np.ones((arity, 1)), \
                            np.diag(np.full((arity), -alpha)), axis=1)
    A = np.append(A, equ_1_lhs, axis=0)
    
    #equation (2)/(4) in LNN draft
    equ_2_lhs = np.reshape(np.append(-1, np.full((arity), 1-alpha)), (1,-1))
    A = np.append(A, equ_2_lhs, axis=0)
    
    #generating b/RHS
    #RHS for non-negativity constraints
    b = np.zeros((arity))
    
    #RHS for equation (1)/(3) in LNN draft
    b = np.append(b, np.full((arity), 1-alpha))
    
    #RHS for equation (2)/(4) in LNN draft
    b = np.append(b, -alpha)
    
    return A, np.reshape(b, (-1, 1))

#constructs the linear inequality constraints. Slack version.
def make_lnn_constraints_with_slacks(alpha, arity):
    #1st col is beta, ws are next, followed by slacks
    #constructing Ax <= b; A is LHS, b is RHS
    
    #generating A/LHS first
    #non-negativity constraints for weights and slacks
    A = np.append(np.zeros((2*arity+1, 1)), np.diag(-np.ones(2*arity+1)), axis=1) 

    #equation (1*)/(3*) in LNN draft
    equ_1_star_lhs = np.append(np.ones((arity, 1)), \
                                np.append(np.diag(np.full((arity), -alpha)), np.diag(np.full((arity), -1)), axis=1), \
                                axis=1)
    A = np.append(A, np.append(equ_1_star_lhs, np.zeros((arity, 1)), axis=1), axis=0)
    
    #equation (2)/(4) in LNN draft
    equ_2_lhs = np.reshape(np.append(np.append(-1, np.full((arity), 1-alpha)), \
                                        np.append(np.zeros((arity)), -1)), \
                            (1,-1))

    A = np.append(A, equ_2_lhs, axis=0)

    #generating b/RHS
    #RHS for non-negativity constraints
    b = np.zeros((2*arity+1))
    
    #RHS for equation (1)/(3) in LNN draft
    b = np.append(b, np.full((arity), 1-alpha))
    
    #RHS for equation (2)/(4) in LNN draft
    b = np.append(b, -alpha)

    return A, np.reshape(b, (-1, 1))
