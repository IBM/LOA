from _lnn import and_tailored
import torch
import torch.nn as nn
import torch.nn.functional as func
from numpy import log
from cdd_interface import cdd_lnn
from lnn_utils import make_lnn_constraints, make_lnn_constraints_with_slacks

def gen_sigm(theta, partial):
    '''
    Computes the coefficients of sigmoid of the form 1 / (1 + exp(- a*x - b))

    Parameters:
    ----------
    theta: the value of the sigmoid at x=theta
    partial: the gradient of the sigmoid at x=theta

    Returns: coefficients a and b (in that order)

    ONLY MEANT FOR USE WITHIN lnn_operators.py
    '''
    return partial / theta / (1-theta), log(theta / (1-theta)) - partial / (1-theta)

#####################################################################################
#######    Constrained LNN models with weighted Lukasiewicz logic   #################
#####################################################################################
class and_lukasiewicz(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Initializes an LNN conjunction (weighted Lukasiewicz logic).

        The cdd member variable is where the double description 
        method related stuff happens.

        Parameters: 
        ---------- 
        alpha: hyperparameter that defines how farther from
        traditional logic we want to go. Note that, 0.5 < alpha < 1

        arity: the number of inputs to this conjunction. try to stick
        to 2
        
        with_slack: set to true if you want the version with
        slack. see Logical_Neural_Networks.pdf.  
        '''
        
        super().__init__()

        self.alpha = alpha
        self.arity = arity
        self.with_slack = with_slack
        self.cdd = cdd_lnn(alpha, arity, with_slack)

        #differentiable clamping
        #need a function whose gradient is non-zero everywhere
        #generalized sigmoid (gen_sigm): 1 / (1 + exp(-(ax+b)))
        #< lower: gen_sigm(x; a_lower, b_lower)
        #> upper: gen_sigm(x; a_upper, b_upper)
        #given theta, partial: solve for a,b such that
        #  - gen_sigm(theta; a,b) = theta
        #  - diff.gen_sigm(theta; a,b) =  gen_sigm(theta; a,b) (1 -  gen_sigm(theta; a,b)) = partial
        #soln is given by gen_a, gen_b (see functions above)
        self.lower = 0.01
        self.upper = 0.99
        partial = 0.01
        self.lower_a, self.lower_b = gen_sigm(self.lower, partial)
        self.upper_a, self.upper_b = gen_sigm(self.upper, partial)
        
    def forward(self, x):
        '''
        Forward function does three things:

        - Requests and obtains beta and argument weights
        from cdd (uses rays and points from double description)

        - Computes the input to the clamping function

        - Depending on the above input, uses the appropriate 
        switch to compute the output

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the conjunction operator. 

        Returns:
        -------

        column vector if non-slack version is used. also, returns a
        scalar (sum of slacks) if version with slacks is used.
        '''
        
        beta, arg_weights, sum_slacks = self.cdd()
        tmp = -torch.add(torch.unsqueeze(torch.mv(1 - x, torch.t(arg_weights)), 1), -beta)
        #ret = torch.clamp(tmp, 0, 1) #creates issues during learning. gradient vanishes if tmp is outside [0,1]
        ret = torch.where(tmp > self.upper, func.sigmoid(self.upper_a * tmp + self.upper_b), \
                          torch.where(tmp < self.lower, func.sigmoid(self.lower_a * tmp + self.lower_b), tmp))
        
        return ret, sum_slacks

    def print_truth_table(self):
        ret = torch.zeros((self.arity+1,1))
        beta, arg_weights, sum_slacks = self.cdd()
        ret[0:self.arity,] = beta - torch.mul(arg_weights.reshape((-1,1)), self.alpha)
        ret[self.arity,0] = beta - (1-self.alpha) * torch.sum(arg_weights)
        return ret
    
    
class or_lukasiewicz(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Initializes an LNN disjunction (weighted Lukasiewicz logic).

        Uses an LNN conjunction since:
                          or(x1,x2...) = 1 - and(1-x1,1-x2,...)  
        The cdd member variable is where the double description
        method related stuff happens.

        Parameters: 
        ---------- 

        alpha: hyperparameter that defines how farther from
        traditional logic we want to go. Note that, 0.5 < alpha < 1
        
        arity: the number of inputs to this disjunction. try to stick
        to 2
        
        with_slack: set to true if you want the version with
        slack. see Logical_Neural_Networks.pdf.  
        '''

        super().__init__()
        self.AND = and_lukasiewicz(alpha, arity, with_slack)

    def forward(self, x):
        '''
        Forward function invokes LNN conjunction operator. Returns
        whatever the conjunction returns. Note that, depending on
        whether with_slack was set to True or False, may return 2
        things (the results of the disjunction and sum of slacks)
        or 1 (just the results of the disjunction)

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the disjunction operator. 

        Returns:
        -------

        column vector if non-slack version is used. also, returns a
        scalar (sum of slacks) if version with slacks is used.
        '''

        ret, slacks = self.AND(1-x)
        return 1 - ret, slacks

#####################################################################################
#######    Unconstrained models with weighted Lukasiewicz logic     #################
#####################################################################################
class and_lukasiewicz_unconstrained(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Initializes an LNN conjunction (weighted Lukasiewicz logic).
        No CDD because this is unconstrained
        Parameters: 
        ---------- 
        alpha: hyperparameter that defines how farther from
        traditional logic we want to go. Note that, 0.5 < alpha < 1
        arity: the number of inputs to this conjunction. try to stick to 2
        with_slack: set to true if you want the version with
        slack. see Logical_Neural_Networks.pdf.  
        '''
        
        super().__init__()

        self.alpha = alpha
        self.arity = arity
        self.with_slack = with_slack
        
        self.lower = 0.01
        self.upper = 0.99
        partial = 0.01
        self.lower_a, self.lower_b = gen_sigm(self.lower, partial)
        self.upper_a, self.upper_b = gen_sigm(self.upper, partial)

        self.weights = nn.Parameter(torch.zeros([self.arity]).uniform_(0.0, 0.1))
        self.beta = nn.Parameter(torch.zeros([]).uniform_(0.0, 0.1))

    def get_params(self, return_joined_params=False):
        weights = func.relu(self.weights)
        beta = func.relu(self.beta)
        params = torch.cat((beta.unsqueeze(0), weights), 0)
        if return_joined_params:
            return params
        else:
            return beta, weights
        
    def forward(self, x):
        '''
        Forward function does three things:
        - beta and argument weights are nn.Parameters
        - Computes the input to the clamping function
        - Depending on the above input, uses the appropriate switch to compute the output

        Parameters:
        ----------
        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the conjunction operator. 

        Returns:
        -------
        column vector if non-slack version is used. also, returns a
        scalar (sum of slacks) if version with slacks is used.
        '''
        
        beta, arg_weights = self.get_params()
        sum_slacks = 0.0

        tmp = -torch.add(torch.unsqueeze(torch.mv(1 - x, torch.t(arg_weights)), 1), -beta)
        #ret = torch.clamp(tmp, 0, 1) #creates issues during learning. gradient vanishes if tmp is outside [0,1]
        ret = torch.where(tmp > self.upper, func.sigmoid(self.upper_a * tmp + self.upper_b), \
                          torch.where(tmp < self.lower, func.sigmoid(self.lower_a * tmp + self.lower_b), tmp))
        
        return ret, sum_slacks


class or_lukasiewicz_unconstrained(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Same as the constrained version
        '''
        super().__init__()
        self.AND = and_lukasiewicz_unconstrained(alpha, arity, with_slack)

    def forward(self, x):
        ret, slacks = self.AND(1-x)
        return 1 - ret, slacks
#####################################################################################

#####################################################################################
####### Soft Constraint lambda-LNN with weighted Lukasiewicz logic  #################
#####################################################################################
class and_lukasiewicz_lambda(and_lukasiewicz_unconstrained):
    def __init__(self, alpha, arity, with_slack):
        super().__init__(alpha, arity, with_slack)
        if self.with_slack:
            self.slack = nn.Parameter(torch.zeros([self.arity+1]).uniform_(0.0, 0.1))
        self.lam=True

    def get_params(self, return_joined_params=False):
        weights = func.relu(self.weights)
        beta = func.relu(self.beta)
        slacks = func.relu(self.slack)
        if self.with_slack:
            params = torch.cat((beta.unsqueeze(0), weights, slacks), 0)
        else:
            params = torch.cat((beta.unsqueeze(0), weights), 0)
        
        if return_joined_params:
            return params
        else:
            return beta, weights

    def compute_constraint_loss(self, lam=0.1):
        if self.with_slack:
            A, b = make_lnn_constraints_with_slacks(self.alpha, self.arity)
        else:
            A, b = make_lnn_constraints(self.alpha, self.arity)
        params = self.get_params(return_joined_params=True) 
        # params = torch.cat((self.beta.unsqueeze(0), self.weights), 0)

        A_th = torch.from_numpy(A).float()
        b_th = torch.from_numpy(b).squeeze(-1).float()   
        loss_constraint = torch.matmul(A_th, params) - b_th
        tot_loss = lam * torch.sum(loss_constraint)

        return tot_loss

    
class and_product(nn.Module):
    def __init__(self):
        '''
        Initializes a product t-norm conjunction. Is parameter-free.
        Use this if you don't know arity or if your application needs
        a conjunction operator that can take variable number of
        arguments.

        '''
        super().__init__()

    def forward(self, x):
        '''
        Forward function returns product of inputs.

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the disjunction operator. 

        Returns: column vector
        -------
        '''

        #ret = torch.exp(torch.sum(torch.log(x + 1e-17), 1, keepdim=True)) #multiplies row-wise
        ret = torch.ones(x.shape[0], 1)
        for i in range(x.shape[1]):
            ret = torch.mul(ret, torch.unsqueeze(x[:,i], 1))
        return ret, 0.0

    
class and_lukasiewicz_classic(nn.Module):
    def __init__(self):
        '''
        Initializes a product t-norm conjunction. Is parameter-free.
        Use this if you don't know arity or if your application needs
        a conjunction operator that can take variable number of
        arguments.

        '''
        super().__init__()

    def forward(self, x):
        '''
        Forward function returns product of inputs.

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the disjunction operator. 

        Returns: column vector
        -------
        '''

        return torch.max(torch.tensor(0.0), torch.sum(x, 1, keepdim=True) - 1), 0.0


class or_max(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, 1, keepdim=True)[0], 0.0
    
                
class predicates(nn.Module):
    def __init__(self, num_predicates, body_len):
        '''
        Use these to express a choice amongst predicates. For use when
        learning rules.

        Parameters:
        ----------

        num_predicates: The domain size of predicates
        body_len: The number of predicates to choose
        '''
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(body_len, num_predicates).uniform_(-0.1, 0.1))

    def forward(self, x):
        '''
        Forward function computes the attention weights and returns the result of mixing predicates.

        Parameters:
        ----------

        x: a 2D tensor whose number of columns should equal self.num_predicates

        Returns: A 2D tensor with 1 column
        -------
        '''
        weights = self.get_params()
        
        ret = func.linear(x, weights)
        #ret = torch.max(torch.mul(x, weights), 1, keepdim=True)[0]
            
        return ret

    def get_params(self):
        ret = func.softmax(self.log_weights, dim=1)
        #ret = func.sigmoid(self.log_weights)
        
        return ret

class predicates1(nn.Module):
    def __init__(self, num_predicates, body_len):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(body_len, num_predicates).uniform_(0.0, 0.1))
        self.beta = nn.Parameter(torch.ones(1, body_len))
        
    def forward(self, x):
        beta, weights = self.get_params()        
        ret = 1 - torch.clamp(- func.linear(x, weights) + beta, 0, 1) #lnn disjunction without constraints
        return ret

    def get_params(self):
        weights = func.relu(self.weights)
        beta = func.relu(self.beta)
        return beta, weights

    
#this is a wrapper .... please see _lnn.py
class and_tailored_wrapper(nn.Module):
    def __init__(self, alpha, arity):
        super().__init__()
        self.alpha = alpha
        self.weights = nn.Parameter(torch.zeros(arity).uniform_(0.0, 0.5))

    def forward(self, x):
        batch_size = x.shape[0]
        input = torch.unsqueeze(x, 1).repeat(1, 2, 1) #converts 2D to 3D tensor, copies the vals in the 2dim so upper and lower bounds are identical

        #also need to rep the weights batch_size times
        ret = \
            and_tailored(input, self.alpha, self.weights.repeat(batch_size, 1)) \
            .sum(0) \
            .unsqueeze(1) / 2.0
        #averages the computed upper and lower bounds to compute 1 value
        
        return ret, 0.0
        
    
def negation(x):
    '''
    Negation is just pass-through, parameter-less.
    '''
    return 1 - x
