import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import lnn_operators
import math

class TransE(nn.Module):
    def __init__(self, num_e, num_r, dim):
        super().__init__()
        
        self.E_v = nn.Embedding(num_e, dim)
        self.E_v.weight.data.uniform_(-6 / math.sqrt(dim), 6 / math.sqrt(dim))

        self.E_r = nn.Embedding(num_r, dim)
        self.E_r.weight.data.uniform_(-6 / math.sqrt(dim), 6 / math.sqrt(dim))
        
    def forward(self, src, pred, tail):
        S = self.E_v(src)
        R = self.E_r(pred)        
        T = self.E_v(tail)

        return - torch.norm(S.div(S.norm(p=2, dim=1, keepdim=True)) \
                            + R \
                            - T.div(T.norm(p=2, dim=1, keepdim=True)), \
                            p=1, dim=1, keepdim=True)

class DistMult(nn.Module):
    def __init__(self, num_e, num_r, dim):
        super().__init__()
        
        self.E_v = nn.Embedding(num_e, dim)
        self.E_v.weight.data.uniform_(-6 / math.sqrt(dim), 6 / math.sqrt(dim))

        self.E_r = nn.Embedding(num_r, dim)
        self.E_r.weight.data.uniform_(-6 / math.sqrt(dim), 6 / math.sqrt(dim))
        
    def forward(self, src, pred, tail):
        S = self.E_v(src)
        S_norm = func.tanh(S.div(S.norm(p=2, dim=1, keepdim=True)))

        R = self.E_r(pred)        

        T = self.E_v(tail)
        T_norm = func.tanh(T.div(T.norm(p=2, dim=1, keepdim=True)))

        return torch.sum(S_norm.mul(T_norm).mul(R), 1, keepdim=True)
    
    
class Meta(nn.Module):
    '''
    Base class for meta-interpretive constructs. Provides basic
    functions, e.g. join with lineage. Every meta-interpretive class
    must contain a dataframe (df) and a matrix (2D tensor). Otherwise,
    the semantics underlying df and mat, along with most of the logic
    is available from the subclasses. 
    '''
    def __init__(self):
        super().__init__()

    #during initialization, we perform a (partial) grounding. this is
    #done by performing a join. for instance, during meta-rule
    #creation we join the body predicates to find out what are the
    #tuples produced by the rule. use this function to perform that
    #join.    
    def lineage_join(self, df, other, left_on, right_on, how):
        '''
        Join function provided for convenience. Besides the resulting
        dataframe, this function also returns vectors capturing
        lineage, that is which tuples from the left and right input
        frames joined together. Very useful for initializing
        meta-interpretive objects.

        Parameters:
        ----------

        df, other: the two dataframes to join

        left_on, right_on: equality predicates (see pandas merge
        documentation)

        how: which kind of join to perform (e.g. inner/outer) (see
        pandas merge documentation)

        Returns: the resulting dataframe and linear vectors
        '''
        df['left'] = list(df.index)
        other['right'] = list(other.index)

        ret = df.merge(other, left_on=left_on, right_on=right_on, how=how)
        df.drop(['left'], axis=1, inplace=True)
        other.drop(['right'], axis=1, inplace=True)

        left_rows = ret[['left']].values.squeeze()
        right_rows = ret[['right']].values.squeeze()
        
        return ret.drop(['left', 'right'], axis=1), left_rows, right_rows

    def rename(self, df, old, name):
        df.rename(columns={old: name}, inplace=True)
    
    def tostring(self):
        return self.df.to_markdown(), str(self.mat)

    #given a vector with integers and nan's, returns the portion
    #consisting of integers    
    def get_indices(self, arr):
        return torch.unique(arr[arr==arr]).long()

    #given a vector (cond) containing nan's and integers, and a vector
    #(left) containing values, does the following:    
    # - if cond[i] is nan then replace with 0
    # - if cond[i] is j (not nan), then replace with left[cond[i]]
    #returns the resulting vector
    def my_where(self, cond, left):
        ret = torch.zeros(cond.size()[0], 1)
        tmp = cond.view(-1)
        valid_idx = (tmp==tmp).nonzero().view(-1).long()
        if len(valid_idx) > 0:
            ret[valid_idx,] = left[tmp[valid_idx].long(),]
        return ret


class Negation(Meta):
    def __init__(self, meta):
        super().__init__()
        self.rule = meta
        self.df = meta.df
        self.mat = meta.mat

    def forward(self, x, choice='default', mask=[]):
        preds, slacks = self.rule(x, choice, mask)
        return lnn_operators.negation(preds), slacks


class BasePredicate(Meta):
    def __init__(self, df):
        '''
        Wraps a dataframe into a meta-interpretive object
        '''
        super().__init__()
        self.df = df
        self.mat = torch.ones(len(self.df.index),1)

    def forward(self, x, choice='default', mask=[]):
        ret = self.mat[x,]
        return ret, 0.0


class BaseMetaPredicate(Meta):
    def __init__(self, dfs, choices=['default']):
        super().__init__()
        attrs = dfs[0].columns.values.tolist()
        attrs.append("ID")
        
        self.df = pd.DataFrame(columns=attrs)
        for i in range(len(dfs)):
            df = dfs[i]
            df['ID'] = [i] * len(df.index)
            self.df = self.df.append(df)
            df.drop(['ID'], axis=1, inplace=True)

        attrs = attrs[0:len(attrs)-1]
        grouped = self.df.groupby(attrs)['ID'].apply(lambda group: np.bincount(group.to_numpy().astype(int), minlength=len(dfs)))

        self.mat = []
        groups = []
        for index, val in grouped.items():
            groups += [index]
            self.mat += [val.tolist()]
        self.df = pd.DataFrame(groups, columns=attrs)
        self.mat = torch.FloatTensor(self.mat)

        #self.alpha = lnn_operators.predicates(len(dfs), 1)
        #self.alpha = lnn_operators.predicates1(len(dfs), 1)
        self.alpha = nn.ModuleDict()
        for choice in choices:
            #self.alpha[choice] = lnn_operators.predicates(len(dfs), 1)
            self.alpha[choice] = lnn_operators.predicates1(len(dfs), 1)
        
    def forward(self, x, choice='default', mask=[]):
        if len(mask) == 0:
            activations = self.mat[x,:]
        else:
            activations = torch.mul(self.mat[x,:], torch.FloatTensor(mask))
        ret = self.alpha[choice](activations)
        return ret, 0.0

    
class MetaPredicate(Meta):
    '''
    Expresses a choice among meta-interpretive objects. Attention
    weights specifying the choice become learnable parameters.
    
    Semantics of df, mat: Member variable df contains every
    possible tuple which can be produced by this object. Member
    variable mat describes which tuples from which
    meta-intepretive object contributes to each df tuple's
    creation. If a meta-interpretive object does not contribute to
    a certain tuple then this is indicated by 'nan'.
    '''
    def __init__(self, rules, choices=['default']):
        '''
        Takes a list of meta-intepretive objects and expresses a
        choice among them

        Parameters: list of meta-interpretive objects
        '''
        super().__init__()

        attrs = rules[0].df.columns.values.tolist()
        
        self.df = rules[0].df

        self.mat = np.reshape(range(len(self.df.index)), (-1,1)).tolist()

        self.rules = nn.ModuleList()
        self.rules.append(rules[0])
        if len(rules) > 1:
            for i in range(1,len(rules)):
                self.rules.append(rules[i])
                self.df, left_rows, right_rows = self.lineage_join(self.df, rules[i].df, left_on=attrs, right_on=attrs, how='outer')
                
                self.mat = [[float('nan')] * i if np.isnan(x) else self.mat[int(x)] for x in left_rows]
                self.mat = np.hstack((np.array(self.mat), np.reshape(right_rows, (-1,1)))).tolist()

            #self.alpha = lnn_operators.predicates1(len(rules), 1)
            #self.alpha = lnn_operators.predicates(len(rules), 1)
            self.alpha = nn.ModuleDict()
            for choice in choices:
                self.alpha[choice] = lnn_operators.predicates1(len(rules), 1)

        self.mat = torch.from_numpy(np.asarray(self.mat)).float()
        
    def forward(self, x, choice='default', mask=[]):
        '''
        Takes a vector of tuple ids and computes values corresponding
        to these. The vector represents the mini-batch, for instance
        [0,1,2] would compute the results for the 0th, first and
        second tuples in self.df.

        Parameters: the vector of tuple ids describing the mini-batch
        '''
        tmp = self.mat[x,0].view(-1, 1)
        idx = self.get_indices(self.mat[x,0])

        slacks = 0.0
        if len(idx) > 0:
            rule_output = torch.zeros(len(self.rules[0].df.index), 1)
            preds, local_slacks = self.rules[0](idx, choice, mask)
            rule_output[idx,] = preds
            slacks = slacks + local_slacks
            activations = self.my_where(tmp, rule_output)
        else:
            activations = torch.zeros(tmp.size()[0], 1)

        ret = activations
        if len(self.rules) > 1:
            for i in range(1,len(self.rules)):
                tmp = self.mat[x,i].view(-1, 1)
                idx = self.get_indices(self.mat[x,i])
                if len(idx) > 0:
                    rule_output = torch.zeros(len(self.rules[i].df.index), 1)
                    preds, local_slacks = self.rules[i](idx, choice, mask)
                    rule_output[idx,] = preds
                    slacks = slacks + local_slacks
                    activation = self.my_where(tmp, rule_output)
                else:
                    activation = torch.zeros(tmp.size()[0], 1)
                activations = torch.cat((activations, activation), 1)

            if len(mask) == 0:
                ret = self.alpha[choice](activations)
            else:
                ret = self.alpha[choice](torch.mul(activations, torch.FloatTensor(mask)))

        return ret, slacks

            
class MetaRule(Meta):
    '''
    Expresses a meta-rule. Givens other meta-intepretive objects, and
    attribute names describing equality join conditions, will use LNN
    conjunctions to express a complex rule that can be learned. If the
    body predicates represent MetaPredicates then training this object
    will also learn how to choose from the contents of the
    MetaPredicates.

    Semantics of df, mat: Member variable df contains every possible
    tuple which can be produced by this object. Member variable mat
    describes which tuples from which body meta-interpretive objects
    join to construct the output tuple.
    '''
    def __init__(self, preds, eq_conditions, alpha, with_slack, choices=['default']):
        '''
        Parameters:
        ----------

        preds: list of meta-intepretive objects that feature in the body

        eq_conditions: list of list of strings (denoting attribute
        names). eq_conditions[i-1][0] denotes a list of attribute
        names that should be equal to eq_conditions[i-1][1] when
        performing the join between preds[i-1] and
        preds[i]. len(eq_conditions) should be one less than
        len(preds).

        alpha: LNN hyperparameter
        '''        
        super().__init__()
        formula_len = len(preds)

        self.with_slack = with_slack

        #self.AND = lnn_operators.and_lukasiewicz(alpha, formula_len, with_slack)
        self.AND = nn.ModuleDict()
        for choice in choices:
            self.AND[choice] = lnn_operators.and_lukasiewicz(alpha, formula_len, with_slack)
        
        self.df = preds[0].df

        self.mat = np.reshape(np.arange(len(self.df.index)), (-1, 1)).tolist()
        
        self.body = nn.ModuleList()
        self.body.append(preds[0])
        for i in range(1, formula_len):
            self.body.append(preds[i])
            
            left = eq_conditions[i-1][0]
            right = eq_conditions[i-1][1]

            self.df, left_rows, right_rows = self.lineage_join(self.df, preds[i].df, left, right, 'inner')

            self.mat = [self.mat[x] for x in left_rows]
            self.mat = np.hstack((np.array(self.mat), np.reshape(right_rows, (-1, 1)))).tolist()

        self.mat = torch.from_numpy(np.asarray(self.mat)).float()

    def forward(self, x, choice='default', mask=[]):
        '''
        Takes a vector of tuple ids and computes values corresponding
        to these. The vector represents the mini-batch, for instance
        [0,1,2] would compute the results for the 0th, first and
        second tuples in self.df.

        Parameters: the vector of tuple ids describing the mini-batch
        '''
        idx = self.get_indices(self.mat[x,0])
        body_pred = torch.zeros(len(self.body[0].df.index), 1)
        preds, slacks = self.body[0](idx, choice, mask)
        body_pred[idx,] = preds
        activations = body_pred[self.mat[x,0].long(),:]
        
        for i in range(1,len(self.body)):
            idx = self.get_indices(self.mat[x,i])
            body_pred = torch.zeros(len(self.body[i].df.index), 1)
            preds, local_slacks = self.body[i](idx, choice, mask)
            body_pred[idx,] = preds
            slacks = slacks + local_slacks
            activation = body_pred[self.mat[x,i].long(),:]
            activations = torch.cat((activations, activation), 1)

        ret, local_slacks = self.AND[choice](activations)
        slacks = slacks + local_slacks
        return ret, slacks
            
        
class Project(Meta):
    '''
    Use this class to perform projections on an existing
    meta-interpretive object (expresses existential statements)

    Semantics of df, mat: Member variable df contains every possible
    tuple which can be produced by this object. Member variable mat is
    a binary matrix (cells contain 0 or 1). The ith row describes
    which tuples from the input meta-interpretive object contribute to
    the ith tuple.
    '''
    def __init__(self, rule, out_attr):
        '''
        Parameters:
        ----------
        rule: the meta-intepretive object to perform the projection on

        out_attr: list of attributes to keep in the output        
        '''
        super().__init__()
        self.rule = rule
    
        grouped = rule.df.groupby(out_attr)
        numgroups = len(grouped)
        
        self.mat = []
        groups = []
        pos = 0
        for name, indices in grouped.groups.items():
            if pos % 100000 == 0:
                print(str(pos) + "/" + str(numgroups))
            self.mat += [indices.to_numpy().tolist()]
            groups += [name]
            pos = pos + 1

        self.df = pd.DataFrame(groups, columns=out_attr)

    def forward(self, x, choice='default', mask=[]):
        '''
        Takes a vector of tuple ids and computes values corresponding
        to these. The vector represents the mini-batch, for instance
        [0,1,2] would compute the results for the 0th, first and
        second tuples in self.df.

        Parameters: the vector of tuple ids describing the mini-batch
        '''
        idx = []
        cnts = []
        for i in range(x.size()[0]):
            tmp = self.mat[x[i].item()]
            cnts += [len(tmp)]
            idx += tmp

        preds, slacks = self.rule(torch.LongTensor(idx), choice, mask)

        pos = 0
        ret = torch.zeros(x.size()[0], 1)
        for i in range(x.size()[0]):
            cnt = cnts[i]
            ret[i,0] = torch.max( preds[ pos : pos+cnt , 0 ] )
            #ret[i,0] = torch.sum( preds[ pos : pos+cnt , 0 ] )
            pos += cnt
        
        return ret, slacks


class DisjunctionRule(Meta):
    '''
    Use this to perform a disjunction amongst multiple
    meta-intepretive objects. Note that, all of these must have the
    same schema. Uses LNN disjunction.

    Semantics of df, mat: Member variable df contains every possible
    tuple which can be produced by this object. Member variable mat
    specifies which tuples from which input meta-interpretive objects
    contribute to the output tuple (through the LNN disjunction).
    '''
    def __init__(self, rules, alpha, dropout, choices=['default']):
        '''
        Parameters:
        ----------
        rules: list of meta-intepretive objects to disjunct (schema
        must be identical)

        alpha: LNN hyperparameter
        '''
        super().__init__()

        self.dropout = dropout #set dropout to 0 to deactivate
        
        #self.OR = lnn_operators.or_lukasiewicz(alpha, len(rules), False)
        #self.OR = lnn_operators.or_max()
        self.OR = nn.ModuleDict()
        for choice in choices:
            self.OR[choice] = lnn_operators.or_max()
        
        attrs = rules[0].df.columns.values.tolist()

        self.df = rules[0].df

        self.mat = np.reshape(range(len(self.df.index)), (-1,1)).tolist()
        
        self.rules = nn.ModuleList()
        self.rules.append(rules[0])
        if len(rules) > 1:
            for i in range(1,len(rules)):
                self.rules.append(rules[i])
                self.df, left_rows, right_rows = self.lineage_join(self.df, rules[i].df, left_on=attrs, right_on=attrs, how='outer')

                self.mat = [[float('nan')] * i if np.isnan(x) else self.mat[int(x)] for x in left_rows]
                self.mat = np.hstack((np.array(self.mat), np.reshape(right_rows, (-1,1)))).tolist()

        self.mat = torch.from_numpy(np.asarray(self.mat))
        
    def forward(self, x, choice='default', mask=[]):
        '''
        Takes a vector of tuple ids and computes values corresponding
        to these. The vector represents the mini-batch, for instance
        [0,1,2] would compute the results for the 0th, first and
        second tuples in self.df.

        Parameters: the vector of tuple ids describing the mini-batch
        '''
        tmp = self.mat[x,0].view(-1, 1)
        idx = self.get_indices(self.mat[x,0])

        slacks = 0.0
        if len(idx) > 0:
            rule_output = torch.zeros(len(self.rules[0].df.index), 1)
            preds, local_slacks = self.rules[0](idx, choice, mask)
            rule_output[idx,] = preds
            slacks = slacks + local_slacks
            activations = self.my_where(tmp, rule_output)
        else:
            activations = torch.zeros(tmp.size()[0], 1)

        for i in range(1,len(self.rules)):
            tmp = self.mat[x,i].view(-1, 1)
            idx = self.get_indices(self.mat[x,i])
            if len(idx) > 0:
                rule_output = torch.zeros(len(self.rules[i].df.index), 1)
                preds, local_slacks = self.rules[i](idx, choice, mask)
                rule_output[idx,] = preds
                slacks = slacks + local_slacks
                activation = self.my_where(tmp, rule_output)
            else:
                activation = torch.zeros(tmp.size()[0], 1)
            activations = torch.cat((activations, activation), 1)

        if self.training:
            ret, local_slacks = self.OR[choice](torch.mul(func.dropout(activations, p=self.dropout, training=self.training), 1-self.dropout)) #no scaling
        else:
            ret, local_slacks = self.OR[choice](activations)

        slacks = slacks + local_slacks
        return ret, slacks
