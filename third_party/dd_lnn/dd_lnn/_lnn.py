import itertools
import warnings

import math
import networkx as nx
import torch
from tqdm import tqdm


class LNN:
    """ Keeps constants, nodes and graphs and provides graph traversal """

    def __init__(self, *args, **kwargs):
        self.propositional = kwargs.get('propositional', False)
        self.alpha = torch.tensor(kwargs.get('alpha', 1.), requires_grad=kwargs.get('alpha_per_node', False))
        self.nodes = {}
        self.constants = {}
        self.graph = nx.DiGraph()
        self.learning = kwargs.get('learning', False)
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        if key in self.nodes:
            return self.nodes[key]
        return self.nodes.setdefault(key, Pred(name=key, lnn=self))

    def __setitem__(self, key, val):
        val.name = val.name if hasattr(val, 'name') else key
        self.nodes[key] = val

    def __repr__(self):
        return self.name if hasattr(self, 'name') else ''

    def const_id(self, const):  # save string id once - reference int key
        if isinstance(const, int):  # already converted to int
            return const
        return self.constants.setdefault(const, len(self.constants))

    def const_str(self, const_id):  # retrieve string id from int key
        if const_id is None:
            return None
        elif isinstance(const_id, str):
            return const_id
        return list(self.constants.keys())[const_id]

    def grounding(self, key):  # Retrieve int id tuple grounding for resource string tuple input
        if len(key) == 0: return tuple()
        tupled_key = key if isinstance(key, tuple) else tuple([key])  # unary 'const1' to ('const1')
        return tuple(self.const_id(const) for const in tupled_key)

    def grounding_str(self, key):  # Retrieve resource string tuple grounding for int id tuple input
        tupled_key = key if isinstance(key, tuple) else tuple([key])  # unary 'const1' to ('const1')
        return tuple(self.const_str(const) for const in tupled_key)

    def upward(self, source=None, func='upward', **kwargs):
        accumulate = torch.tensor([0., 0.])
        nodes = list(nx.dfs_postorder_nodes(self.graph, source))
        progress_inference = tqdm(total=len(nodes),
                                  disable=not kwargs.get('progress_inference', False))
        for node in nodes:
            val = getattr(node, func)(**kwargs) if hasattr(node, func) else None
            accumulate += val if val is not None else torch.tensor([0, 0.])
            progress_inference.set_description(func + '_' + kwargs.get('steps', '') + ', ' + str(accumulate.detach().round()))
            progress_inference.update(1)
        return accumulate

    def downward(self, source=None, func='downward', **kwargs):
        accumulate = torch.tensor([0., 0.])
        nodes = list(reversed(list(nx.dfs_postorder_nodes(self.graph, source))))
        progress_inference = tqdm(total=len(nodes),
                                  disable=not kwargs.get('progress_inference', False))
        for node in nodes:
            if func == 'downward':
                kwargs['direction'] = 'downward'  # required for contradiction loss calculation
            val = getattr(node, func)(**kwargs) if hasattr(node, func) else None
            accumulate += val if val is not None else torch.tensor([0, 0.])
            progress_inference.set_description(func + '_' + kwargs.get('steps', '') + ', ' + str(accumulate.detach().round()))
            progress_inference.update(1)
        return accumulate

    def propagate(self, **kwargs):
        steps, converged = 0, False
        inference = {'default': [self.upward, self.downward], 'upward': [self.upward], 'downward': [self.downward],
                     'reverse': [self.downward, self.upward]}
        while not (converged):
            diff = 0.
            for infer in inference[kwargs.get('direction', 'default')]:  # TODO: arrest inference at contradiction
                diff += infer(steps=str(steps), **kwargs)
            steps += 1
            converged = (diff < 1e-1).prod() if steps < kwargs.get('convergence_shortcircuit', 1e3) else True
        return steps, converged == True, diff, self.upward(func='loss_terms')

    def train(self, epochs=1e3, print_epoch=None, **kwargs):
        epoch = 0
        weights_converged = False
        progress_training = tqdm(total=epochs, disable=not kwargs.get('progress_training', False), desc='training')
        while (not weights_converged) and (epoch <= epochs):  # and not(inference_converged):
            print_ = False if print_epoch is None else (True if epoch % print_epoch == 0 else False)
            if print_:
                print('\nepoch', epoch)
            self.upward(func='reset_bounds')  # reset all bounds before weight updates
            _, inference_converged, diff, loss_terms = self.propagate(**kwargs)  # inference until convergence
            loss = torch.dot(loss_terms, kwargs.get('coefficients',
                                                    torch.tensor([1., 1.])))  # coefficients [unsupervised, supervised]
            if print_:
                print(
                    'loss: {} \nloss_terms: {} \ninference converged: {} \nweights converged: {} \nbounds tightened: {}'.format(
                        loss, loss_terms.detach(), inference_converged == True, weights_converged == True,
                        diff.detach()))
                prediction_fn = kwargs.get('prediction_fn')
                if prediction_fn is not None:
                    print('Predictions before weight update:')
                    prediction_fn(self)
            if has_grad(loss):
                if print_: print('backprop...')
                loss.backward(retain_graph=True)
                if print_: print('updating weights...')
                pbar_flag, kwargs['progress_inference'] = kwargs.get('progress_inference'), False
                weight_diff = self.upward(func='update_weights', **kwargs)
                weights_converged = weight_diff[0] < 1e-3
                if print_: print('weight diff: {:.3e}'.format(weight_diff[0].item()))
                kwargs['progress_inference'] = pbar_flag
                if print_:
                    if prediction_fn is not None:
                        print('Predictions after weight update:')
                        prediction_fn(self)
            else:
                raise Exception('loss has no tracked gradients')
            epoch += 1
            progress_training.update(1)
        print('\nconverged at epoch {}'.format(epoch))
        print(
            'loss: {} \nloss_terms: {} \ninference converged: {} \nweights converged: {} \nbounds tightened: {}'.format(
                loss, loss_terms.detach(), inference_converged == True, weights_converged == True, diff.detach()))
        progress_training.close()


class Node:
    ''' Supports LNN node API (var matching and bindings) and connects new node to graph 
    
    mypred = Pred(lnn=g)  # add basic predicate to lnn
    g['pred6'] = Pred(name='diffname', lnn=g)  # predicate added and referenced in lnn

    # node operands can be a 1) node object reference, or 2) node string id
    # node defaults: batch_size=lnn.batch_size, trainable=False
    mynode = Or(g['pred2'], 'pred3')  # adds node to LNN graph, keep reference
    mynode2 = Or('pred2', 'pred3', lnn=g)  # specify lnn or inherit from operands
    mynode3 = Not(mynode2)  # Not performs passthrough to 1 - operand[[1,0]]
    g['mynode3'] = Implies(g['pred1'], 'pred3')  # lnn keeps reference
    And(mynode, 'pred1', 'pred3', trainable=True)  # adds node - traverse LNN to access

    # constant matching / participation: defaults: matched for predicates, inherited for operations
    # constant index refers to global variable identifier
    And('pred2', g['pred3'])  # And(x,y) = pred2(x) & pred3(x, y)
    And('pred1', 'pred2', g['pred3'])  # And(x,y) = pred1(x)&pred2(x)&pred3(x, y)
    And(('pred2', 0), (g['pred3'], 0, 1))  # And(0,1) = pred2(0) & pred3(0, 1)
    And(('pred2', 1), ('pred3', 0, 1), lnn=g)  # And(1,0) = pred2(1) & pred3(0, 1)
    Or(('pred2', 1), And(('pred2', 1), ('pred3', 0, 1), lnn=g))  #Or(1,0) = pred2(1) & And(1,0)
    Or(('pred2', 1), (And(('pred2', 1), ('pred3', 0, 1), lnn=g), 2, 3))  #Or(1,0) = pred2(1) & And(2,3)

    # variable binding: default no binding
    And(g['pred2'], ('pred3', 'const1', ''))  # pred2(x) & pred3(x=const1, y)
    And(g['pred2'], ('pred3', ['const1', 'const2'], ''))  # pred2(x) & pred3(x=[const1, const2], y)

    # constant matching & variable binding
    And((g['pred2'], 0), ('pred3', (0, 'const1'), 1))  # pred2(x) & pred3(x=const1, y)
    And((g['pred2'], 0), ('pred3', (0, ['const1', 'const2']), (1, '')))  # pred2(x=[const1, const2]) & pred3(x=[const1, const2], y)
    And((g['pred2'], (0, 'const1')), ('pred3', (0, ['const1', 'const2']), (1, '')))  # pred2(x=const1) & pred3(x=const1, y) [intersect]

    '''

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.opvarmap = [[] for i in range(len(args))]  # [op_index] [op slot] -> formula opvar slot position
        self.opmap = (lambda op_index, grounding: tuple(grounding[i] for i in self.opvarmap[op_index])
        if grounding else grounding)  # maps grounding to op grounding
        self.map = (lambda op_index, grounding:
                    tuple((None if i is None else grounding[i]) for i in self.varmap[op_index])
                    if grounding else grounding)  # maps var grounding to op grounding
        self.bindings = {}  # [slot position] -> opvar bindings
        self.operands = []
        self.__dict__.update(kwargs)

        if not hasattr(self, 'lnn'):  # inherit dd_lnn from operands
            for op in args:
                if not isinstance(op, str):
                    if isinstance(op, tuple):
                        if not isinstance(op[0], str):
                            self.lnn = op[0].lnn
                            break
                    elif isinstance(op, object):
                        self.lnn = op.lnn
                        break

        self.lnn.graph.add_node(self)
        if hasattr(self, 'name'):
            self.lnn[self.name] = self

        varids = {}
        for op_index, op in enumerate(args):
            if isinstance(op, tuple):  # ('PredID', ...) or (PredObject, ...) matching / binding
                self.operands.append(self.lnn[op[0]] if isinstance(op[0], str) else op[0])

                for loc, var in enumerate(op[1:]):  # each variable of operand
                    #                     print('op', op, 'loc', loc, 'var', var, 'opvarmap', self.opvarmap)
                    # ('PredID', (0, 'const1'), ...) or ('PredID', 0, ...)
                    if (isinstance(var, tuple) or isinstance(var, int)
                            or (var and isinstance(var, str) and var[0] == '?')):
                        varid = var[0] if isinstance(var, tuple) else var
                    else:  # ('PredID', 'const1', ...)
                        # And(('pred1', 0, 2), And(('pred2', 0, 1), ('pred3', 1, 2)))
                        varid = (self.operands[-1].vars[loc] if loc < len(self.operands[-1].vars)
                                 else loc)
                    self.opvarmap[op_index].append(varids.setdefault(varid, len(varids)))
                    #                     print('after: opvarmap', self.opvarmap)

                    # (..., (0, 'const1'), ...) or (..., (0, ['const1', 'const2']), ...)
                    # (..., 'const1', ...) or (..., ['const1', 'const2'], ...)
                    bindings = (var[1] if isinstance(var, tuple) else
                                var if (var and isinstance(var, str) and var[0] != '?') else '')
                    if bindings != '':
                        bindings = [bindings] if isinstance(bindings, str) else bindings
                        self.bindings.setdefault(varids[varid], []).extend(
                            [self.lnn.const_id(b) for b in bindings])
            else:  # PredObject or 'PredID' 
                self.operands.append(self.lnn[op] if isinstance(op, str) else op)
                var_list = self.operands[-1].vars if len(self.operands[-1].vars) else [] # else [0, 1]
                for varid in var_list:
                    self.opvarmap[op_index].append(varids.setdefault(varid, len(varids)))

            self.lnn.graph.add_edge(self, self.operands[-1])

        self.opvars = list(varids.keys()) # [slot position] -> varid
        # [op_index] [formula opvar slot position] -> op slot
        self.opmapvar = {op_index: {slot: i for i, slot in enumerate(self.opvarmap[op_index])}
                         for op_index in range(len(self.operands))}
        self.opvars2vars = [_ for _ in range(len(self.opvars))]
        self.vars = self.opvars + []
        if 'Quantifier' in [c.__name__ for c in self.__class__.__mro__]:
            dims = ([_ for _ in range(len(self.opvars))] if not hasattr(self, 'dim')
                                                            or self.dim is None or self.dim == [] else
                    self.dim if isinstance(self.dim, list) else [self.dim])
            self.vars = []
            self.vars2opvars = {}
            for i, v in enumerate(self.opvars):
                if v not in dims:
                    self.vars2opvars[len(self.vars)] = i
                    self.opvars2vars[i] = len(self.vars)
                    self.vars.append(v)

        self.update_varmap()

        self.nomap = []
        for op_index, op in enumerate(self.operands):
            if self.opvarmap[op_index] == [_ for _ in range(len(self.vars))]:  # no mapping required
                self.nomap.append(True)
            #                 print('nomap', self, op, self.opvarmap[op_index], self.vars)
            else:
                self.nomap.append(False)

        for op_index, op in enumerate(self.operands):  # apply child bindings
            m = self.opmap(op_index, [_ for _ in range(len(self.opvars))])
            for i in op.bindings:
                j = op.opvars2vars[i]
                if m[j] in self.bindings:
                    self.bindings[m[j]].extend(op.bindings[i])
                else:
                    self.bindings[m[j]] = op.bindings[i]

    def update_varmap(self):  # [op_index] [op slot] -> formula var slot position
        self.opvars_count = [0 for _ in self.opvars]
        self.varmap = [[] for i in self.operands]
        for op_index, op in enumerate(self.operands):
            for slot in self.opvarmap[op_index]:
                varid = self.opvars[slot]
                self.opvars_count[slot] += 1
                if varid in self.vars:
                    self.varmap[op_index].append(self.vars.index(varid))
                else:
                    self.varmap[op_index].append(None)
            
    def mapvar(self, op_index, partial, default=None):  # [0, 2] -> [0, None, 2],   [2, 2] -> [2]
        ''' Maps operand grounding onto this node's opvars grounding '''
        if partial is False or default is False:
            return False
        pattern = [None for _ in range(len(self.opvars))] if default is None else list(default)
#         print('partial', partial, 'pattern', pattern)
#         print('opvarmap', self.opvarmap)
        for i, const in enumerate(partial):
            slot = self.opvarmap[op_index][i]
            if slot is None:
                continue
            if pattern[slot] is not None and const is not None and pattern[slot] != const:
                return False
            pattern[slot] = pattern[slot] if pattern[slot] is not None else const
#             print('slot', slot, 'pattern', pattern)
        return tuple(pattern)

    def __repr__(self):
        return self.name if hasattr(self, 'name') else self.__class__.__name__

    def one(self):
        return 1

    def args_str(self):
        return ', '.join(["'" + op + "'" if isinstance(op, str) else str(op) for op in self.args])

    def declare(self):  # original declaration string
        return (self.__class__.__name__ + '(' + self.args_str()
                + (', ' if self.args_str() and self.kwargs else '')
                + ', '.join([kw + "='" + self.kwargs[kw] + "'" if isinstance(self.kwargs[kw], str)
                             else kw + '=' + str(self.kwargs[kw]) for kw in self.kwargs]) + ')')

    def op_bindings(self, op_index, bindings=None):  # maps variable bindings to operand compatible
        bindings = bindings if bindings else self.bindings
        if isinstance(bindings, list):
            return [self.opmap(op_index, binding) for binding in bindings]
        elif isinstance(bindings, tuple):
            return self.opmap(op_index, bindings)

        op_bindings = {}
        for op_loc, loc in enumerate(self.opvarmap[op_index]):
            if loc in bindings:
                op_bindings[op_loc] = bindings[loc]
        return op_bindings


class Formula(Node):
    ''' Introduce groundings and associated truth-value bounds '''

    true = lambda requires_grad=False: torch.tensor([1, 1.], requires_grad=requires_grad)
    false = lambda requires_grad=False: torch.tensor([0, 0.], requires_grad=requires_grad)
    unknown = lambda requires_grad=False: torch.tensor([0, 1.], requires_grad=requires_grad)

    def __init__(self, *args, **kwargs):
        self.leaves = {}
        if not isinstance(self, PassThrough):
            self.bounds = {}
        self.init_bounds = None
        self.indices = {}
        self.default = Formula.unknown  # open-world assumption
        self.target_bounds = kwargs.get('target', {})
        super().__init__(*args, **kwargs)
#         self.updates = 0
#         self.op_version = [op.version-1 for op in self.operands]

    @property
    def alpha(self):
        return self.lnn.alpha

    def __getitem__(self, key):  # Item-based get with default bounds for missing elements
        grounding = self.lnn.grounding(key)
        if grounding in self.bounds:
            return self.bounds[grounding]

        if grounding in self.leaves:
            return self.leaves[grounding]

        default = self.default()
        if default.is_leaf:
            self.leaves[grounding] = default
        return default

    def __setitem__(self, key, val):  # Item-based set with automatic discarding of false values
        grounding = self.lnn.grounding(key)
        if val.is_leaf and grounding not in self.leaves:
            self.leaves[grounding] = val
        self.bounds[grounding] = val
#         self.updates += 1

    def aggregate(self, key, new_bounds):
        """ Aggregating proofs returns boolean flag stating whether the bounds were updated """
        grounding = self.lnn.grounding(key)
        prev_bounds = self[grounding] + 0
        if grounding in self.bounds \
        and has_grad(prev_bounds) \
        and (new_bounds[0] <= prev_bounds[0] or new_bounds[1] >= prev_bounds[1]):            
            return 0.
        self[key] = torch.stack([torch.max(prev_bounds[0], new_bounds[0]), torch.min(prev_bounds[1], new_bounds[1])])
#         print(self, 'updated', prev_bounds.clone(), '->', new_bounds.clone(), '=', self[key].clone())
        return (self[key] - prev_bounds).abs()

    def fits(self, key, pattern):  # does a grounding fit a pattern?
        for i, const_id in enumerate(key):
            if pattern[i] is not None and const_id != pattern[i]:
                break
        else:
            return True
        return False

    def groundings(self, bindings=None):  # retrieve node's groundings under bindings
        if not bindings:
            yield from self.bounds
            return
        
        if isinstance(bindings, tuple):  # binding is direct grounding
            if (None not in bindings) and (True not in bindings) and bindings in self.bounds:
                yield bindings
                return
            
            bind = []
            for i, binding in enumerate(bindings):
                if bindings[i] is not None:
                    bind.append((i, bindings[i]))
                    
            for key in self.bounds:
                for i, binding in bind:
                    if key[i] != binding:
                        break
                else:
                    yield key
            return
                    
        for key in self.bounds:
            for i, const_id in enumerate(key):
                if i in bindings and bindings[i] is not None:
                    if isinstance(bindings[i], list):
                        if const_id not in bindings[i]:
                            break
                    elif const_id != bindings[i]:
                        break
            else:
                yield key

    
    def state(self, key=(), shorthand=False, bounds=None):
        if bounds is None:
            bounds = self.bounds[key]
        if (bounds[0] <= 1 - self.alpha) and (bounds[1] >= self.alpha):
            return 'U' if shorthand else 'Unknown'
        elif (bounds[0] >= self.alpha) and (bounds[1] >= self.alpha):
            return 'T' if shorthand else 'True'
        elif (bounds[0] <= 1 - self.alpha) and (bounds[1] <= 1 - self.alpha):
            return 'F' if shorthand else 'False'
        elif bounds[0] >= torch.tensor(0.5) and bounds[1] <= torch.tensor(1.) and bounds[0] <= bounds[1]:
            return '~T' if shorthand else '~True'
        elif bounds[1] <= torch.tensor(0.5) and bounds[0] >= torch.tensor(0.) and bounds[0] <= bounds[1]:
            return '~F' if shorthand else '~False'
        elif bounds[0] > bounds[1]:
            return 'C' if shorthand else 'Contradiction'
        elif bounds[0] > 1 - self.alpha or bounds[1] < self.alpha:
            return '~U' if shorthand else '~Unknown'
        else:
            raise Exception('out of bounds:', bounds, 'alpha', self.alpha)

    def print_graph(self, **kwargs):
        """ Print the states of all nodes in the graph"""
        alpha = "𝛼[{:.2e}]".format(self.alpha.item()) if kwargs.get('alpha', False) else ''
        grounding = kwargs.get('grounding')
        if not kwargs.get('get_facts', True) and isinstance(self, Pred): return
        print(self)
        if grounding is None:
            for g in self.bounds:
                print(self.lnn.grounding_str(g), self.state(g), self.bounds[g], alpha)
        else:
            print(self.lnn.grounding_str(grounding), self.state(grounding), self[grounding], alpha)
        if kwargs.get('weights', False) and isinstance(self, Neuron): print('weights:', self.weights)

    def reset_bounds(self,
                     **kwargs):  # TODO: should only be the logic-cone of changed node (graph partitioning and inference counter), to reset only downstream from weight changes
        def reset():
            if self.init_bounds is None:
                self.init_bounds = self.bounds.copy()
            else:
                self.bounds = self.init_bounds.copy()

        if kwargs.get('reset_facts', False):
            if type(self) == Pred: reset()  # reset only the facts
        else:
            reset()  # reset all bounds

    def loss_terms(self, **kwargs):
        """ Calculates contradiction, task_focus """
        loss_terms = torch.tensor([0, 0.])
        diff = 0.
        for grounding in self.groundings(self.bindings):
            if grounding is False: continue
            if None in grounding: continue
            #             if kwargs.get('direction') == 'downward' and kwargs.get('downward_learning', False):
            #                 with torch.no_grad():
            #                     loss_terms += torch.cat((self.contradiction(grounding), self.task_focus(grounding)))
            #             else:
            loss_terms += torch.cat((self.contradiction(grounding), self.task_focus(grounding)))
        return loss_terms

    def contradiction(self, key):
        contradiction = self[key][0] - self[key][1]
        if self.state(bounds=self[key]) == 'Contradiction':
            return torch.max(torch.tensor([0.]), contradiction)
        else:
            return torch.min(torch.tensor([0.]), contradiction.abs())  # tracked intra-classical contradiction

    def task_focus(self, key):
        MSELoss = torch.nn.MSELoss()
        if not self.target_bounds or self.target_bounds[key] is None:
            return torch.tensor([0.])
        else:
            return MSELoss(self[key], self.target_bounds[key]).reshape(1)


class Neuron(Formula):
    ''' Introduces weight parameters and logical connective calculations '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = torch.tensor(1.)
        self.weights = torch.ones(len(args), requires_grad=True)
        min_alpha = self.weights.sum() / (self.weights.sum() + self.weights.max()) # alpha constraint
        if self.lnn.alpha <= min_alpha:
            self.lnn.alpha = min_alpha + 1e-10

    def inference_groundings(self, bindings=None):
        ''' Generator: Determine operator/operand(s) grounding cover under bindings '''
        
        '''return nullary grounding if propositional lnn'''
        if self.lnn.propositional: 
            yield ()
        
        ''' List all binding variations in bindmaps '''
        bindings = bindings if bindings else self.bindings
        isbound = bindings and (len(bindings) > 0)
        bindmaps = [[None for _ in self.opvars]]
        for i in bindings:
            newbindmaps = []
            for bindmap in bindmaps:
                for b in bindings[i]:
                    newbindmap = bindmap + []
                    newbindmap[i] = b
                    newbindmaps.append(newbindmap)
            bindmaps = newbindmaps
        
        ''' List all shared variables in joinvars '''
        joinvars = []
        for i, count in enumerate(self.opvars_count):
            if count > 1:
                joinvars.append(i)
        
        ''' Obtain (partial) groundings and yield full groundings from operands '''
        yielded = set()
        patterns = set()
        for op_index, op in enumerate(self.operands):
            groundings = set()
            for bindmap in bindmaps:
                for partial in op.groundings(self.map(op_index, bindmap) if isbound else None):
                    if partial is False:
                        continue
                        
                    pattern = self.mapvar(op_index, partial, bindmap if isbound else None)
                    if pattern is False:
                        continue
                        
                    if None not in pattern:  # full grounding - all variables instantiated
                        groundings |= {pattern}
                    else:
                        patterns |= {pattern}  # partial grounding - to be joined

            for pattern in groundings:
                if pattern not in yielded:
#                     print('    [######] add pattern', pattern, self.lnn.grounding_str(pattern))
                    yielded |= {pattern}
                    yield pattern
        
        ''' [Repartitioned nested loop join]
            Join procedure: combine partials, yield full groundings '''
        while len(patterns):  # continue to join remaining partials
            ''' Partitioning '''
            partitions = {}
            for pattern in patterns:
                for v in joinvars:  # only join over shared variables
                    const = pattern[v]
                    if const is not None:
                        partitions.setdefault((v, const), [])
                        partitions[(v, const)].append(pattern)
#             print('\nRepartitioned: keys', [(k[0], self.lnn.const_str(k[1])) for k in partitions])
            
            ''' Nested loop join '''
            patterns = set()
            for (joinvar, joinval), tojoin in partitions.items():
#                 print('\njoinvar', joinvar, 'joinval', joinval, 'tojoin', [self.lnn.grounding_str(_) for _ in tojoin])
                len_tojoin = len(tojoin)
                for i, pattern1 in enumerate(tojoin):  # outer loop
                    for j in range(i+1, len_tojoin):  # inner loop
                        pattern2 = tojoin[j]
                        
                        pattern = ()
                        for c1, c2 in zip(pattern1, pattern2):  # merge patterns
                            if c1 is not None and c2 is None:
                                pattern += (c1,)
                            elif c1 is None and c2 is not None:
                                pattern += (c2,)
                            elif c1 == c2:
                                pattern += (c1,)
                            elif c1 != c2:
                                break
                        else:  # no merge conflict
                            print('Joined [on ' + self.lnn.const_str(joinval) + ']', self.lnn.grounding_str(pattern1), 
                                  self.lnn.grounding_str(pattern2), ':  ', self.lnn.grounding_str(pattern))
                            if None not in pattern:  # pattern complete
                                if pattern not in yielded:
#                                     print('    [######] add grounding', pattern, self.lnn.grounding_str(pattern))
                                    yielded |= {pattern}
                                    yield pattern
                            else:
                                patterns |= {pattern}  # (fuller) partial - for another joining round
        
    def input_bounds(self, grounding):  #stack input bounds together
        ''' Stack input operand bounds for given grounding '''
        
        bounds = ()
        for op_index, op in enumerate(self.operands):
#             print('  -- Get bounds:', op, self.lnn.grounding_str(self.map(op_index, grounding)), op[self.map(op_index, grounding)])
            op_bounds = op[self.map(op_index, grounding)]
    
#             if op_bounds[0] > op_bounds[1]:  # don't propagate contradiction
#                 return False
            bounds += (op_bounds,)
        return torch.stack(bounds, dim=1)

    def upward(self, groundings=None, **kwargs):
        ''' Upward inference at neuron
            Produces and aggregates new proofs for all joined groundings '''
#         print('upward @', self)
        # for op_index, op in enumerate(self.operands): # Node-level version check
        #     if self.op_version[op_index] < op.version:  # outdated - operand has new info
        #         break
        # else:
        #     return 0  # nothing to update
        groundings = groundings if groundings else self.inference_groundings(self.bindings)
        diff = 0.
        for grounding in groundings:
            if None in grounding:
                continue
            input_bounds = self.input_bounds(grounding)
            new_bounds = self.func(input_bounds, self.alpha, self.weights)
            diff += self.aggregate(grounding, new_bounds)
        return diff

    def downward(self, groundings=None, **kwargs):
#         print('downward @', self)
        groundings = groundings if groundings else self.inference_groundings(self.bindings)
        diff = 0.
        for grounding in groundings:
            if None in grounding:
                continue
            input_bounds = self.input_bounds(grounding)
            if input_bounds is False:
                continue
            op_bounds = self[grounding]
            new_bounds = self.func_inv(op_bounds, input_bounds, self.alpha, self.weights)
            for op_index, op in enumerate(self.operands):
                diff += op.aggregate(self.map(op_index, grounding), new_bounds[:, op_index])
        return diff

    def update_weights(self, lr=1, verbose=False, **kwargs):
        if has_grad(self.weights, fn=False):
            if has_nan(self.weights.grad) or (self.weights.grad == 0).prod():  # ignore updates with NaNs or all zeros
                if has_nan(self.weights):
                    raise Exception('Nan weights', self.weights, self.weights.grad)
                elif (self.weights.grad == 0).prod():
                    return
                else:
                    if not hasattr(self, 'warning'):
                        warnings.warn('Nan gradients detected @: {}'.format(self))
                    return
            before = (self.weights.clone(), self.weights.grad.clone())
            with torch.no_grad():
                if (self.weights.grad.min() == self.weights.grad.max()) or (self.weights == 0).sum():
                    self.weights = (self.weights - self.weights.grad * lr).clamp(
                        min=0)  # don't scale gradients when zero weights exists or gradients drop simulataneously
                else:
                    scaled_grad = self.weights.grad / self.weights.grad.abs().max()  # extra gradient optimisation
                    self.weights = (
                            self.weights - scaled_grad * lr)  # only scale gradients when different directions exist
                self.weights = (self.weights / self.weights.max()).clamp(min=0)

            diff = (before[0] - self.weights).abs().sum()
            if verbose:
                print('updating', self)
                print('weights: (', *['{:.5e}, '.format(_) for _ in before[0]], ')',
                      '\n-grads: (', *['{:.5e}, '.format(-_) for _ in before[1]], ')',
                      '\nupdate: (', *['{:.5e}, '.format(_) for _ in self.weights], ')')
            self.weights.requires_grad_(True)
            if (self.weights > 1).sum() or (self.weights < 0).sum():
                raise Exception('weight update discrepency')
            return torch.stack((diff, torch.tensor(0.)))


class Pred(Formula):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vars == []:
            self.vars = [0, 1]


class Quantifier(Formula):
    ''' # quantifiers
        q = ForAll(g['pred2'])  # e.g. unary predicate
        print('query value', q[()])  # reduced to nullary predicate
        q = ForAll(g['pred3'], dim=0)  # specify variable(s) to quantify over - produces groundings on remainder
        q = ForAll((g['pred10'], 0, 1, 4, 3, 5, 2), dim=[0, 3, 4])  # groundings on [1, 5, 2]
        q = Exists(g['pred2'])  # e.g. unary predicate, only one variable
        q.upward()
        print('query value', q[()], 'query arg', q.arg.get((), torch.tensor([0, 1.])))  
        # q.arg returns set of constants for which q.bounds == 1 for Exists '''

    istrue = lambda self: torch.tensor([self.state(key) == 'True' for key in self.groundings(self.bindings)]).prod()
    hastrue = lambda self: torch.tensor([self.state(key) == 'True' for key in self.groundings(self.bindings)]).sum()

    def upward(self, **kwargs):
        self.arg = {}
        op = self.operands[0]

        best_bounds = {}
        for grounding in op.groundings(self.op_bindings(0)):
            if grounding is False:
                continue
            key = ()
            val = ()
            for i, const in enumerate(grounding):  # split quantified/non-quantified variables
                if self.opvars[i] in self.vars:
                    key += (const,)
                else:
                    val += (const,)
            bounds = op[grounding]
            best_bounds[key] = self.acc(bounds, best_bounds.setdefault(key, bounds))
            if bounds[0] == 1:
                self.arg.setdefault(key, {})[val] = bounds

        diff = 0.
        for key in best_bounds:
            diff += self.aggregate(key, best_bounds[key])
        return diff


class Exists(Quantifier):
    def __init__(self, *args, **kwargs):
        #         self.acc = lambda a, b: torch.stack([(a[0] if a[0] >= b[0] else b[0]),
        #                                               (a[1] if a[1] >= b[1] else b[1])])
        self.acc = lambda a, b: torch.stack([(a[0] if a[0] >= b[0] else b[0]), torch.tensor(1.)])  # OWA
        super().__init__(*args, **kwargs)

    def downward(self, **kwargs):
        diff = 0.
        op = self.operands[0]
        for grounding in list(op.groundings(self.op_bindings(0))):
            if grounding is False:
                continue
            key = ()
            for i, const in enumerate(grounding):  # split quantified/non-quantified variables
                if self.opvars[i] in self.vars:
                    key += (const,)
            diff += op.aggregate(grounding, torch.stack([torch.tensor(0.), self[key][1]]))
        return diff

    def argset(self):  # provide all true consts - assumes one dim
        args = []
        for key in self.arg:
            for val in self.arg[key]:
                if len(val) == 1:
                    args.append(self.lnn.const_str(val[0]))
                else:
                    args.append(self.lnn.grounding_str(val))

        return set(args)


class ForAll(Quantifier):
    def __init__(self, *args, **kwargs):
        #         self.acc = lambda a, b: torch.stack([(a[0] if a[0] <= b[0] else b[0]),
        #                                              (a[1] if a[1] <= b[1] else b[1])])
        self.acc = lambda a, b: torch.stack([torch.tensor(0.), (a[1] if a[1] <= b[1] else b[1])])  # OWA
        super().__init__(*args, **kwargs)

    def downward(self, **kwargs):
        diff = 0.
        op = self.operands[0]
        for grounding in list(op.groundings(self.op_bindings(0))):
            if grounding is False:
                continue
            key = ()
            for i, const in enumerate(grounding):  # split quantified/non-quantified variables
                if self.opvars[i] in self.vars:
                    key += (const,)
            diff += op.aggregate(grounding, torch.stack([self[key][0], torch.tensor(1.)]))
        return diff


class And(Neuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = and_(self.lnn.learning)
        self.func_inv = and_inv(self.lnn.learning)
        


class Or(Neuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = or_(self.lnn.learning)
        self.func_inv = or_inv(self.lnn.learning)
        


class Implies(Neuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = implies_(self.lnn.learning)
        self.func_inv = implies_inv(self.lnn.learning)
        


class PassThrough(Formula):
    def __getitem__(self, key):
        if self.nomap[0]:
            return self.operands[0][key]
        return self.operands[0][self.map(0, key)]

    def __setitem__(self, key, val):
        if self.nomap[0]:
            self.operands[0][key] = val
        else:
            self.operands[0][self.map(0, key)] = val

    def aggregate(self, key, new_bounds):
        if self.nomap[0]:
            return self.operands[0].aggregate(key, new_bounds)
        return self.operands[0].aggregate(self.map(0, key), new_bounds)

    def groundings(self, bindings=False):
        if self.nomap[0]:
            yield from self.operands[0].groundings(self.op_bindings(0, bindings))
        else:
            for grounding in self.operands[0].groundings(self.op_bindings(0, bindings)):
                yield self.mapvar(0, grounding)

    @property
    def bounds(self):
        return self.operands[0].bounds

    @property
    def alpha(self):
        return self.operands[0].alpha


class Not(PassThrough):  # Involute negation - passthrough node
    def __getitem__(self, key):
        if self.nomap[0]:
            return not_(self.operands[0][key])
        return not_(self.operands[0][self.map(0, key)])

    def __setitem__(self, key, val):
        if self.nomap[0]:
            self.operands[0][key] = not_(val)
        else:
            self.operands[0][self.map(0, key)] = not_(val)

    def aggregate(self, key, new_bounds):
        if self.nomap[0]:
            return self.operands[0].aggregate(key, not_(new_bounds))
        return self.operands[0].aggregate(self.map(0, key), not_(new_bounds))


BiConditional = lambda *args, **kwargs: And(Implies(args[0], args[1], **kwargs),
                                            Implies(args[1], args[0], **kwargs))


def and_(mode):
    return {0: and_lukasiewicz,
            1: and_tailored}[mode]


def and_lukasiewicz(bounds, beta, weights):  # Weighted Łukasiewicz t-norm
    return val_clamp(beta - torch.mv(1 - bounds, weights))


def and_tailored(bounds, alpha, weights):  # Tailored activation function
    batch = len(bounds.shape) == 3
    if not batch: bounds, weights = bounds[None, :], weights.repeat(2, 1)
    result = torch.empty((bounds.shape[0], 2))
    for i, (b, w) in enumerate(zip(bounds, weights)):
        result[i] = tailored_piecewise(torch.mv(b, w), alpha, w)  # TODO: vectorise
    if has_nan(result): raise Exception('Nans detected', result)  # for debugging purposes
    return result.t() if batch else result[0]


def tailored_piecewise(x, alpha, weights, inverse=False):
    beta = weights.sum()
    Yf = 1 - alpha
    Yt = alpha
    Xf = beta - weights.max() * alpha
    Xt = beta * alpha
    result = 0
    if inverse:
        if ((0 <= x) * (x <= Yf)).sum():
            Gf = Yf / Xf
            result += (0 <= x) * (x <= Yf) * ((x / Gf) if (Yf != 0) else torch.zeros(1))
        elif ((Yf < x) * (x < Yt)).sum():
            Gm = (Yt - Yf) / (Xt - Xf)
            result += (Yf < x) * (x < Yt) * ((Xf + (x - Yf) / Gm) if (Yf != Yt) else torch.zeros(1))
        elif ((Yt <= x) * (x <= 1)).sum():
            Gt = (1 - Yt) / (beta - Xt)
            result += (Yt <= x) * (x <= 1) * ((((x - Yt) / Gt) if (Xt != beta) else torch.zeros(1)) + Xt)
    else:
        if ((0 <= x) * (x <= Xf)).sum():
            Gf = Yf / Xf
            result += (0 <= x) * (x <= Xf) * ((x * Gf) if (Xf != 0) else torch.zeros(1))  # else at identity
        elif ((Xf < x) * (x < Xt)).sum():
            Gm = (Yt - Yf) / (Xt - Xf)
            result += (Xf < x) * (x < Xt) * (
                (Yf + (x - Xf) * Gm) if (Xf != Xt) else torch.zeros(1))  # else at strictly classical
        elif ((Xt <= x) * (x <= beta)).sum():
            Gt = (1 - Yt) / (beta - Xt)
            result += (Xt <= x) * (x <= beta) * (
                    (((x - Xt) * Gt) if (Xt != beta) else torch.zeros(1)) + Yt)  # else at perfect rules
    return result

def tailored_sigmoid(x, alpha, weights):
    beta = weights.sum()
    Yf = 1 - alpha
    Yt = alpha
    Xf = beta - weights.max() * alpha
    Xt = beta * alpha
    print(Xf, Xt)
    result = 0
    A = 2*torch.log(Yf/Yt)/(Xf-Xt)
    B = torch.log(Yt/Yf)+A*Xf
    return 1/(1+torch.pow(math.e, -A*x+B))

def and_inv(mode):
    return {0: and_inv_lukasiewicz,
            1: and_inv_tailored}[mode]


def and_inv_lukasiewicz(out_bounds, bounds, beta, weights):  # Weighted Łukasiewicz downward inference with unclamping
    w_terms = (1 - bounds) * weights
    f_inv = (out_bounds + (out_bounds <= 0).float() * torch.stack([beta - weights.sum(), torch.tensor(0.)]) +
             (out_bounds >= 1).float() * torch.stack([torch.tensor(0.), beta - 1]))  # unclamping: bounds extension
    result = 1 + (f_inv[:, None] - (beta - (w_terms.sum(dim=1)[:, None] - w_terms)[[1, 0]])) / weights
    return val_clamp(result)


def and_inv_tailored(out_bounds, bounds, alpha, weights, mode=1):  # tailored activation functional inverse
    unpack_partial = lambda input_, index: torch.cat([input_[:, :index], input_[:, index + 1:]], dim=1)
    f_inv = and_f_inv_tailored(out_bounds, bounds, alpha, weights)
    if bounds.shape[1] > 2:
        partial_bounds = torch.stack([unpack_partial(bounds, b) for b in range(bounds.shape[1])])  # TODO: vectorise
        partial_weights = torch.cat(
            [unpack_partial(weights[None, :], w) for w in range(weights.shape[0])])  # TODO: vectorise
        partial_and = and_(mode)(partial_bounds, alpha, partial_weights)
    elif bounds.shape[1] == 2:
        partial_and = bounds.flip(dims=(1,))
    else:
        raise Exception('incorrect bounds shape', bounds.shape)
    stack = implies_(mode)(torch.stack([partial_and, out_bounds.repeat(partial_and.shape[1], 1).t()]).permute(2, 1, 0),
                           alpha, torch.ones(partial_and.shape[1], 2))
    stack_div = divisor_fill(divident=val_clamp(stack), divisor=weights, fill=1.)  # divide by weight & prevent div 0
    
    '''conditioning the logical inverse with the functional inverse'''
    non_classical = (partial_and[1, :] <= 1 - alpha).float() * (out_bounds[1].repeat(1, partial_and.shape[
        1]) <= 1 - alpha).float()  # flags for non classical points of logical inverse
    classical = torch.where(non_classical.repeat(2, 1) == True, tracked_unknown(stack_div),
                         stack_div)  # classical equality does not hold
    result = classical.clone()
    result[0, :] = torch.where((classical[0, :] > 1 - alpha).float() * (classical[0, :] > f_inv[0, :]).float() == True,
                               f_inv[0, :], classical[0, :])  # lower bound corrections
    result[1, :] = torch.where((classical[1, :] < alpha).float() * (classical[1, :] < f_inv[1, :]).float() == True,
                               f_inv[1, :], classical[1, :])  # upper bound corrections
    if has_nan(result): raise Exception('Nans detected', result)  # for debugging purposes
    return result


def and_f_inv_tailored(out_bounds, bounds, alpha, weights):  # tailored functional inverse
    if (weights == 0).prod():  # bypass node
        f_inv = torch.ones_like(bounds)
    else:
        w_sum = bounds * weights
        f_inv = tailored_piecewise(out_bounds, alpha, weights, inverse=True)
        f_inv = f_inv[:, None] - (w_sum.sum(dim=1)[:, None] - w_sum)
        f_inv = divisor_fill(divident=f_inv, divisor=weights,
                             fill=weights.sum())  # fill to make the f_inv True where weights are zero
    return f_inv.clamp(0, 1.)


def not_(bounds, transpose=False):
    return (1 - bounds.t()[[1, 0]]).t() if transpose else 1 - bounds[[1, 0]]


def or_(mode):
    return {0: or_lukasiewicz,
            1: or_tailored}[mode]


def or_lukasiewicz(bounds, beta, weights, mode=0):
    return not_(and_(mode)(not_(bounds), beta, weights))


def or_tailored(bounds, alpha, weights, mode=1):
    return not_(and_(mode)(not_(bounds), alpha, weights))


def or_inv(mode):
    return {0: or_inv_lukasiewicz,
            1: or_inv_tailored}[mode]


def or_inv_lukasiewicz(out_bounds, bounds, beta, weights, mode=0):
    return not_(and_inv(mode)(not_(out_bounds), not_(bounds), beta, weights))


def or_inv_tailored(out_bounds, bounds, alpha, weights, mode=1):
    return not_(and_inv(mode)(not_(out_bounds), not_(bounds), alpha, weights))


def implies_(mode):
    return {0: implies_lukasiewicz,
            1: implies_tailored}[mode]


def implies_lukasiewicz(bounds, beta, weights, mode=0):
    return not_(and_(mode)(torch.stack((bounds[:, 0], not_(bounds[:, 1]))).t(), beta, weights))


def implies_tailored(bounds, alpha, weights, mode=1):
    batch = len(bounds.shape) == 3
    if not batch: bounds, weights = bounds[None, :], weights[None, :]
    stack = torch.stack([bounds[:, :, 0], not_(bounds[:, :, 1], transpose=True)]).permute(1, 2,
                                                                                          0)  # negate and stack dims: 0=batch, 1=[lower,upper], 2=[lhs,rhs]
    result = not_(and_(mode)(stack, alpha, weights)).t()
    return result.t() if batch else result[0]


def implies_inv(mode):
    return {0: implies_inv_lukasiewicz,
            1: implies_inv_tailored}[mode]


def implies_inv_lukasiewicz(out_bounds, bounds, beta, weights, mode=0):
    implies_bounds = torch.stack((bounds[:, 0], not_(bounds[:, 1]))).t()
    tmp_bounds = and_inv(mode)(not_(out_bounds), implies_bounds, beta, weights)
    return torch.stack((tmp_bounds[:, 0], not_(tmp_bounds[:, 1]))).t()


def implies_inv_tailored(out_bounds, bounds, alpha, weights, mode=1):
    implies_bounds = torch.stack((bounds[:, 0], not_(bounds[:, 1]))).t()
    tmp_bounds = and_inv(mode)(not_(out_bounds), implies_bounds, alpha, weights)
    return torch.stack((tmp_bounds[:, 0], not_(tmp_bounds[:, 1]))).t()


def alpha_scaled(x, in_alpha, alpha, alpha_per_node=False):
    if alpha_per_node:
        return ((x >= 0).float() * (x < 1 - in_alpha).float() * x * (1 - alpha) / (1 - in_alpha)
                + (x >= 1 - in_alpha).float() * (x < 0.5).float() * (0.5 - (0.5 - x) * (alpha - 0.5) / (in_alpha - 0.5))
                + (x >= 0.5).float() * (x < in_alpha).float() * (0.5 + (x - 0.5) * alpha / in_alpha)
                + (x >= in_alpha).float() * (x < 1).float() * (1 - (1 - x) * (1 - alpha) / (1 - in_alpha)))
    else:
        return x


def divisor_fill(divident, divisor, fill=1.):
    """ 
    Divide the bounds tensor (divident) by weights (divisor) while respecting gradient connectivity,
    shortcurcuits a div 0 error with the fill value 
    """

    L, U = divident
    result = torch.stack((L.masked_scatter(divisor != 0,
                                           L.masked_select(divisor != 0) / divisor.masked_select(divisor != 0)),
                          U.masked_scatter(divisor != 0,
                                           U.masked_select(divisor != 0) / divisor.masked_select(divisor != 0))))
    result.masked_fill_(divisor == 0, fill)
    return result


val_clamp = lambda x, min_=0, max_=1, grad=False: x - (x.detach() - min_).clamp(max=0) - (x.detach() - max_).clamp(
    0) if grad else x.clamp(0, 1.)  # gradient-transparent clamp
has_grad = lambda tensor, fn=True: True if (tensor.grad_fn is not None and fn) else (
    True if tensor.grad is not None else False)
has_nan = lambda tensor: (torch.stack([torch.isnan(tensor.flatten()[i]) for i in range(tensor.numel())])).sum()
tracked_unknown = lambda tensor1, tensor2=Formula.unknown(): torch.stack((torch.min(tensor1[0], tensor2[0]), torch.max(tensor1[1], tensor2[1])))
