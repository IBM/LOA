import numpy as np
import cdd
import torch
import torch.nn as nn
import torch.nn.functional as func

class cdd_lnn(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Constructor for interface to cdd library. To be used in LNN
        operators.  Computes rays and points correponding to linear
        inequality constraints stated in
        Logical_Neural_Networks.pdf. Calls double description method
        to do this. See details in
        https://openreview.net/forum?id=ByxXZpVtPB

        Two major things happening in this constructor:
        
        - Constructs the A, b corresponding to the linear inequality
        constraints (see make_constraints below)
        
        - Calls cdd's double description method to construct rays and
        points for the inquality constraints. Note that this is a
        one-time conversion

        YOU SHOULD NOT NEED TO CALL ANYTHING FROM THIS MODULE
        '''
        
        super().__init__()
        
        self.with_slack = with_slack
        self.arity = arity
        self.alpha = alpha
        
        A, b = self.make_constraints(alpha, arity) if not with_slack else self.make_constraints_with_slacks(alpha, arity)
        is_point, rays_and_points = self.get_v_representation(A, b)

        self.has_point = False
        if np.sum(is_point) > 0:
            self.has_point = True
            self.points = torch.as_tensor(rays_and_points[is_point,], dtype=torch.float)
            self.gamma = nn.Parameter(torch.zeros(1, self.points.shape[0]).uniform_(0, 0.1))
            
        self.has_ray = False
        if np.sum(~is_point) > 0:
            self.has_ray = True
            self.rays = torch.as_tensor(rays_and_points[~is_point,], dtype=torch.float)
            self.mu = nn.Parameter(torch.zeros(1, self.rays.shape[0]).uniform_(0, 0.1))
                    
    def forward(self):
        '''
        Computes beta and argument weights using current self.gamma
        and self.mu.

        - Calls softmax on gamma and then multiplies with rays
        - Calls softplus on mu and then multiplies with points
        - Adds the above two vectors

        Returns:
        -------

        beta: a scalar
        argument weights: a vector of length self.arity
        '''
        params = self.get_params()
            
        if self.with_slack:
            return params[0], params[1:self.arity + 1], torch.sum(params[self.arity + 1:len(params)])
        else:
            return params[0], params[1:self.arity + 1], 0.0

    def get_params(self):
        if self.has_point:
            points_weights = func.softmax(self.gamma, dim=1)
            points_params = torch.matmul(points_weights, self.points).squeeze()
            
        if self.has_ray:
            rays_weights = func.relu(self.mu) #func.softplus(self.mu)
            rays_params = torch.matmul(rays_weights, self.rays).squeeze()
            
        if self.has_ray and self.has_point:
            params = points_params + rays_params
        elif self.has_point:
            params = points_params
        elif self.has_ray:
            params = rays_params

        return params
        
    #talks to cdd and gets rays and points
    def get_v_representation(self, A, b):
        h_rep = np.append(b, -A, axis=1) #cdd needs [b -A]
    
        mat = cdd.Matrix(h_rep, number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY

        poly = cdd.Polyhedron(mat)
        dd = poly.get_generators()

        V = np.asarray(dd)

        return (V[:,0] > 0.5), V[:,1:]

    #constructs the linear inequality constraints. Non-slack version.
    
    def make_constraints(self, alpha, arity):
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
    def make_constraints_with_slacks(self, alpha, arity):
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
