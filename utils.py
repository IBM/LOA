import torch


def is_whitespace(c, use_space=True):
    if (c == ' ' and use_space) or \
            c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
        return True
    return False


def instantiate(x, instance_list):
    for val in instance_list:
        item, instance = val
        if item == x:
            return instance


def ground_predicate(x, pred_name, facts):
    if pred_name in facts:
        return x in facts[pred_name]
    else:
        return False


def ground_predicate_instantiate(x, pred_name, facts):
    x0, x1 = x
    x_inst = instantiate(x0, facts['is_instance'])
    y_inst = instantiate(x1, facts['is_instance'])
    return ground_predicate((x_inst, y_inst), pred_name, facts)


def obtain_predicates_logic_vector(rule_arity,
                                   x, y=None,
                                   facts=None,
                                   template=None,
                                   add_negations=False):
    predicates_to_input = template[rule_arity]
    all_preds = []
    logic_vector = []
    for pred in predicates_to_input.split(';'):
        all_preds.append(pred)
        pred_name = pred.split('(')[0]
        inputs = pred.split('(')[-1].split(')')[0].split(',')
        assert len(inputs) <= rule_arity, \
            'Rule of arity {} should have {} or ' \
            'less inputs: found {}'.format(rule_arity, rule_arity, len(inputs))
        if len(inputs) == 1:
            inputs = x if inputs[0] == 'x' else y
        else:
            inputs = (x, y)
        if pred_name == 'atlocation':
            logic_vector.append(
                int(ground_predicate_instantiate(inputs, pred_name, facts)))
        else:
            logic_vector.append(
                int(ground_predicate(inputs, pred_name, facts)))

    value = torch.tensor(logic_vector)
    if add_negations:
        full_x = torch.cat((value, 1 - value), 1).float()
    else:
        full_x = value.float()
    return full_x, all_preds


def get_facts_state(facts):
    str_id = ''
    for k, v in facts.items():
        str_id = str_id + k + ':'
        for x in v:
            if isinstance(x, tuple):
                str_id = str_id + ','.join(x) + ';'
            else:
                str_id = str_id + x + ';'
    return str_id


def combine_cs_facts(state_facts, cs_facts):
    for k, v in cs_facts.items():
        state_facts[k] = v
    return state_facts
