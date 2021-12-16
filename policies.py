import os
import sys

import torch.nn as nn

try:
    DDLNN_HOME = os.environ['DDLNN_HOME']
except BaseException:
    DDLNN_HOME = os.path.expanduser('third_party/dd_lnn')

meta_rule_home = '{}/src/meta_rule/'.format(DDLNN_HOME)
src_rule_home = '{}/dd_lnn/'.format(DDLNN_HOME)

sys.path.append(meta_rule_home)
sys.path.append(src_rule_home)

EPS = 1e-10

if True:
    from lnn_operators \
        import and_lukasiewicz, \
        and_lukasiewicz_unconstrained, and_lukasiewicz_lambda


class SimpleAndLNN(nn.Module):

    def __init__(self, arity=4, use_slack=True, alpha=0.95, constrained=True,
                 use_lambda=True):
        super().__init__()
        self.alpha = alpha
        self.use_slack = use_slack
        self.arity = arity
        self.constrained = constrained
        self.use_lambda = use_lambda
        if use_lambda:
            assert constrained, \
                'Lambda LNN can only be used for constrained version'
        if constrained:
            if use_lambda:
                self.and_node = and_lukasiewicz_lambda(alpha, arity, use_slack)
            else:
                self.and_node = and_lukasiewicz(alpha, arity, use_slack)
        else:
            self.and_node = \
                and_lukasiewicz_unconstrained(alpha, arity, use_slack)

    def forward(self, x):
        final_pred, final_slack = self.and_node(x)
        return final_pred, final_slack

    def extract_weights(self, normed=True, verbose=False):

        if self.constrained:
            if self.use_lambda:
                beta, wts = self.and_node.get_params()
            else:
                beta, wts, slacks = self.and_node.cdd()

        else:
            beta, wts = self.and_node.get_params()

        if normed:
            wts = wts / wts.max()

        if verbose:
            print('beta : ' + str(beta.item()))
            print('argument weights : ' + str(wts.detach()))

        return beta, wts


class PolicyLNNTWC_SingleAnd(nn.Module):
    def __init__(self,
                 admissible_verbs,
                 use_constraint=True,
                 num_by_arity=None):
        super().__init__()
        alpha = 0.95
        use_slack = True

        self.alpha = alpha
        self.use_slack = use_slack

        self.use_constraint = use_constraint
        self.admissible_verbs = admissible_verbs

        self.models = nn.ModuleDict()
        if num_by_arity is None:
            self.total_inputs = {1: 6, 2: 12}
        else:
            self.total_inputs = num_by_arity
        for v, arity in admissible_verbs.items():
            self.init_model_for_verb(v, self.total_inputs[arity])

    def init_model_for_verb(self, v, nb_inputs):
        self.models[v] = \
            SimpleAndLNN(arity=nb_inputs, use_slack=self.alpha,
                         alpha=self.alpha, constrained=self.use_constraint)

    def compute_constraint_loss(self, lnn_model_name='go', lam=0.0001):
        return \
            self.models[lnn_model_name].\
            and_node.compute_constraint_loss(lam=lam)\
            if self.models[lnn_model_name].and_node.lam else 0.0

    def forward_eval(self, x, lnn_model_name='go', split=True):
        out, _ = self.models[lnn_model_name](x)
        activations = out.view(1, -1) + EPS
        return activations
