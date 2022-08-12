import glob
import os
from collections import defaultdict

import numpy as np

import gym
import nltk
import textworld
import textworld.gym
from textworld import EnvInfos

EPS = 10e-8
DEFALT_TWC_HOME = 'games/twc/'


class Action2Literal:
    def __init__(self):
        self.stop_words = \
            [
                'from', 'with', 'on', 'into',
                'to', 'above', 'across', 'against',
                'along', 'among', 'around', 'at',
                'before', 'behind', 'below', 'beneath',
                'beside', 'between', 'by', 'down',
                'from', 'in', 'near', 'off',
                'toward', 'under', 'upon', 'within'
            ]

    def __call__(self, command):
        tokens = nltk.word_tokenize(command)
        verb = tokens[0]
        obj1 = None
        obj2 = None
        if len(tokens) > 1:
            found_prepo = False
            for k, word in enumerate(tokens[1:]):
                if word in self.stop_words:
                    obj1 = ' '.join(tokens[1:k + 1])
                    obj2 = ' '.join(tokens[k + 2:])
                    found_prepo = True
                    break
            if not found_prepo:
                obj1 = ' '.join(tokens[1:])
        return verb, obj1, obj2


def get_infos(eval=True, recipe=True, walkthrough=True):
    request_infos = \
        EnvInfos(verbs=True, moves=True, inventory=True, description=True,
                 objective=True, intermediate_reward=True,
                 policy_commands=True, max_score=True,
                 admissible_commands=True, last_action=True, game=True,
                 facts=True, entities=True,
                 won=True, lost=True, location=True)
    request_infos.verbs = True
    request_infos.extras = []
    if recipe:
        request_infos.extras += ['recipe']
    if walkthrough:
        request_infos.extras += ['walkthrough']
    if eval:
        request_infos.max_score = True
        request_infos.admissible_commands = True
        request_infos.command_templates = True
    return request_infos


def load_twc_game(difficulty_level,
                  split='test', max_episode_steps=50,
                  batch_size=None, game_no=None):
    try:
        twc_path = os.environ['TWC_HOME']
    except KeyError:
        os.environ['TWC_HOME'] = DEFALT_TWC_HOME
        twc_path = os.environ['TWC_HOME']

    game_file_names = \
        glob.glob('{}/{}/{}/*.ulx'.format(twc_path, difficulty_level, split))
    tsv_file_names = \
        glob.glob('{}/{}/{}/manual_subgraph_brief/*.tsv'.
                  format(twc_path, difficulty_level, split))

    if game_no is not None:
        game_file_names = game_file_names[game_no]
        hash_id = game_file_names.split('-')[-1].split('.')[0]
        tsv_file_name = [x for x in tsv_file_names if hash_id in x]

        commonsense_kb = {}
        fp = open(tsv_file_name[0], 'r')
        while (True):
            line_str = fp.readline()
            vals = line_str.split()
            if not bool(line_str):
                break

            vals = [' '.join(x.split('_')) for x in vals]
            pred_name = vals[1]
            if pred_name not in commonsense_kb:
                commonsense_kb[vals[1]] = [(vals[0], vals[2])]
            else:
                commonsense_kb[vals[1]].append((vals[0], vals[2]))

    game_file_names = \
        [game_file_names] \
        if not isinstance(game_file_names, list) else game_file_names

    request_infos = get_infos(recipe=False)
    env_id = textworld.gym.register_games(game_file_names, request_infos,
                                          max_episode_steps=max_episode_steps,
                                          name='cleanup-' + difficulty_level,
                                          batch_size=batch_size)

    env = gym.make(env_id)
    return env, game_file_names, commonsense_kb


def get_all_admissible_verbs(logical_env,
                             banned_verbs=[
                                 'look', 'inventory', 'examine',
                                 'eat', 'prepare'
                             ]):
    action2literal = Action2Literal()
    verb_set = []
    for game_no in range(len(logical_env.games)):
        obs, infos = logical_env.reset()
        comms = infos['extra.walkthrough']  # infos['admissible_commands']
        for cmd in comms:
            v, n1, n2 = action2literal(cmd)
            if v not in banned_verbs:
                verb_set.append(v)
    return list(set(verb_set))


def extract_predicate(pred):
    pred_name = pred.name
    variables = []
    for v in pred.arguments:
        variables.append(v.name.split(':')[0])
    return pred_name, variables


def get_total_score(logical_env):
    scores = []
    for game_no in range(len(logical_env.games)):
        obs, infos = logical_env.env.reset()
        score = infos['max_score']
        scores.append(score)
    return np.sum(scores), np.mean(scores)


def get_all_objects(logical_env):
    all_entities = []
    for game_no in range(len(logical_env.games)):
        obs, infos = logical_env.env.reset()
        all_entities += infos['entities']
    return list(set(all_entities))


def hasObject(x, obj):
    return float((' ' + obj.lower()) in x.lower())


class LogicalTWC:
    def __init__(self, difficulty_level,
                 max_episode_steps=50, batch_size=None,
                 split='test', use_action_pruner=False, game_number=0):

        self.env, self.games, self.commonsense_kb = \
            load_twc_game(difficulty_level,
                          split=split,
                          max_episode_steps=max_episode_steps,
                          batch_size=batch_size,
                          game_no=game_number)
        self.num_games = len(self.games)
        self.game_counter = 0
        self.use_action_pruner = use_action_pruner
        self.action_pruner = None
        self.listObjDirn = ['north', 'south', 'east', 'west']
        self.action2literal = Action2Literal()

    def reset(self):
        game_state, infos = self.env.reset()
        self.admissible_commands = infos['admissible_commands']
        return game_state, infos

    def step(self, action):
        game_state, game_score, game_done, infos = self.env.step(action)
        self.admissible_commands = infos['admissible_commands']
        return game_state, game_score, game_done, infos

    def convert2lifted(self, noun, fact_list):
        logical_state = []
        fols = ['in_room', 'in_inventory', 'closed', 'has_exit']
        for fol_item in fols:
            val = int(noun in fact_list[fol_item])
            logical_state += [val, 1 - val]
        return np.array(logical_state)

    def get_logical_state(self, infos, filter_preds=True):
        possible_props = ['in_room', 'in_inventory', 'at', 'has_exit']
        obs_text = infos['description'].lower()
        fact_list = defaultdict(list)

        facts = infos['facts']
        ingredient_dict = {}
        preds_with_requirements = []
        for k in range(len(facts)):
            pred_name, variables = extract_predicate(facts[k])
            if 'ingredient' in pred_name:
                ingredient_name = pred_name.split('_')[0] + '_' + \
                    str(int(pred_name.split('_')[1]) - 1)
                ingredient_dict[ingredient_name] = variables[0]
            if len(variables) == 1 and \
                    any(['ingredient' in v for v in variables]):
                preds_with_requirements.append(pred_name)

        for k in range(len(facts)):
            pred_name, variables = extract_predicate(facts[k])
            if 'ingredient' in pred_name:
                predicate = 'is_ingredient'
                fact_list[predicate].append(variables[0])
            if 'free' in pred_name:
                predicate = 'has_free_slot'
                fact_list[predicate] = True
            if len(variables) == 1 and pred_name in preds_with_requirements:
                if any(['ingredient' in v for v in variables]):
                    predicate = 'required_' + pred_name
                    if predicate in fact_list:
                        fact_list[predicate].append(
                            ingredient_dict[variables[0]])
                    else:
                        fact_list[predicate] = [ingredient_dict[variables[0]]]
                else:
                    fact_list[pred_name].append(variables[0])

            if pred_name == 'in':
                if variables[-1] == 'I':
                    if variables[0] == 'meal':
                        predicate = 'ready_to_eat'
                        fact_list[predicate] = True
                    predicate = 'in_inventory'
                    fact_list[predicate].append(variables[0])
                else:
                    if variables[0] in obs_text:
                        predicate = 'in_room'
                        fact_list[predicate].append(variables[0])
                    # for the at(x,y) predicate
                    if all([gnd in obs_text for gnd in variables]):
                        predicate = 'at'
                        fact_list[predicate].append(
                            (variables[0], variables[1]))

            elif pred_name == 'closed':
                if variables[0] in obs_text:
                    predicate = 'closed'
                    fact_list[predicate].append(variables[0])
            elif pred_name == 'on':
                if variables[0] in obs_text:
                    predicate = 'in_room'
                    fact_list[predicate].append(variables[0])
            elif pred_name == 'at':
                if variables[0] in obs_text:
                    # print('Matched room: ', current_room)
                    predicate = 'in_room'
                    fact_list[predicate].append(variables[0])
        predicate = 'has_exit'
        for dirn in self.listObjDirn:
            if dirn in obs_text:
                fact_list[predicate].append(dirn)
        if filter_preds:
            fact_list_filter = {key: fact_list[key] for key in possible_props}
        else:
            fact_list_filter = fact_list

        # Add commonsense graph part here
        for k, v in self.commonsense_kb.items():
            k = 'atlocation' if k == 'relatedto' else k
            fact_list_filter[k] = v

        entities = \
            fact_list_filter['in_room'] + fact_list_filter['in_inventory']
        kb_entities = []
        for x in fact_list_filter['atlocation']:
            kb_entities.append(x[0])
            kb_entities.append(x[1])
        kb_entities = list(set(kb_entities))

        # Add is_instance predicate part here
        fact_list_filter['is_instance'] = []
        for ent in (entities):
            for kb_ent in kb_entities:
                num_chars = len(kb_ent)
                if kb_ent in ent[-(num_chars + 1):]:
                    fact_list_filter['is_instance'].append((ent, kb_ent))
        return fact_list_filter
