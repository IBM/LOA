import os
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from amr_parser import (AMRSemParser, get_formatted_obs_text,
                        get_verbnet_preds_from_obslist)
from logical_twc import DEFALT_TWC_HOME, EPS, Action2Literal, LogicalTWC
from tqdm import tqdm
from utils import (combine_cs_facts, get_facts_state,
                   ground_predicate_instantiate,
                   obtain_predicates_logic_vector)

if True:
    try:
        _ = os.environ['TWC_HOME']
    except KeyError:
        print('Could not find TWC_HOME. Using default path...')
        os.environ['TWC_HOME'] = DEFALT_TWC_HOME

    try:
        _ = os.environ['DDLNN_HOME']
    except KeyError:
        print('Could not find DDLNN_HOME. Using default path...')
        os.environ['DDLNN_HOME'] = 'third_party/dd_lnn/'

    from policies import PolicyLNNTWC_SingleAnd


class LogicalTWCQuantifier(LogicalTWC):
    def get_logical_state(self, *args, **kwargs):
        facts = super().get_logical_state(*args, **kwargs)

        entities = facts['in_room'] + facts['in_inventory']
        facts['placeable'] = []
        for x in entities:
            for y in entities:
                if ground_predicate_instantiate((x, y), 'atlocation', facts):
                    facts['placeable'].append(x)
        return facts


class LOAAgent:

    def __init__(self,
                 admissible_verbs,
                 amr_server_ip='localhost',
                 amr_server_port=None,
                 prune_by_state_change=False,
                 sem_parser_mode='both'):
        if admissible_verbs is None:
            self.admissible_verbs = {}
        else:
            self.admissible_verbs = admissible_verbs

        self.amr_server_ip = amr_server_ip
        self.amr_server_port = amr_server_port

        self.action2literal = Action2Literal()
        self.buffer = None
        self.weights = None
        self.is_trained = {v: False for v, _ in self.admissible_verbs.items()}
        self.prune_by_state_change = prune_by_state_change
        self.steps = 0
        self.train_eps = 0
        self.sem_parser_mode = sem_parser_mode

        self.lr = self.wd = self.pi = self.optimizer = self.loss_fn = None
        self.arity_predicate_templates = self.predicate_templates = None

    def init_lnn_model(self, pi, lr=0.01, wd=1e-5):
        self.lr = lr
        self.wd = wd
        self.pi = pi
        self.optimizer = \
            optim.Adam(self.pi.parameters(), lr=lr, weight_decay=wd)
        self.loss_fn = nn.BCELoss()

    def get_string_templates(self,
                             pred_list,
                             two_arity_predicates=['atlocation',
                                                   'is_instance']):
        pred_1_x = [item + '(x)' for item in pred_list]
        pred_1_y = [item + '(y)' for item in pred_list]
        pred_1_xy = [item + '(x,y)' for item in two_arity_predicates]
        self.arity_predicate_templates = \
            {1: ';'.join(pred_1_x),
             2: ';'.join(pred_1_x + pred_1_y + pred_1_xy)}
        return self.arity_predicate_templates

    def obtain_templates(self):
        self.predicate_templates = {}
        for verb in self.admissible_verbs:
            arity = self.admissible_verbs[verb]
            predicates_to_input = self.arity_predicate_templates[arity]
            all_predicate_templates = []
            for pred in predicates_to_input.split(';'):
                all_predicate_templates.append(pred)
            self.predicate_templates[verb] = all_predicate_templates

    def update_buffer(self,
                      train_buffer, weights=None, update_weights=False):
        if update_weights:
            assert (weights is not None), \
                'Weights should be specified if update_weights is True'

        if self.buffer is None:
            self.buffer = train_buffer
            if update_weights:
                self.weights = weights
        else:
            for k, v in self.admissible_verbs.items():
                # positive data
                if len(train_buffer[k]['pos']):
                    if len(self.buffer[k]['pos']):
                        self.buffer[k]['pos'] = \
                            self.buffer[k]['pos'] + train_buffer[k]['pos']
                    else:
                        self.buffer[k]['pos'] = train_buffer[k]['pos']

                    if update_weights:
                        if len(self.weights[k]['pos']):
                            self.weights[k]['pos'] = \
                                self.weights[k]['pos'] + weights[k]['pos']
                        else:
                            self.weights[k]['pos'] = weights[k]['pos']

                if len(train_buffer[k]['neg']):
                    self.buffer[k]['neg'] = \
                        self.buffer[k]['neg'] + train_buffer[k]['neg']
                    if update_weights:
                        self.weights[k]['neg'] = \
                            self.weights[k]['neg'] + weights[k]['neg']

    def evaluate_env_without_action_verb(self,
                                         difficulty_level,
                                         original_episodic_actions,
                                         original_score,
                                         action_verb_to_prune,
                                         save_trajectories=False):
        true_score = 0
        pruned_score = 0
        trajs = {k: [] for k in original_episodic_actions}
        for k in original_episodic_actions:
            # Initialize the game here
            env = LogicalTWCQuantifier(difficulty_level,
                                       split='train', max_episode_steps=50,
                                       batch_size=None, game_number=k)
            game_trajs = []
            for actions, score in zip(original_episodic_actions[k],
                                      original_score[k]):
                if score == 0:
                    continue
                pruned_actions = \
                    [x for x in actions
                     if action_verb_to_prune != env.action2literal(x)[0]]
                ep_trajs = []
                _, infos = env.reset()

                unique_state = get_formatted_obs_text(infos)
                ep_trajs.append((unique_state, 'reset', 0))
                for ac in pruned_actions:

                    _, score_pruned, _, infos = env.step(ac)
                    self.steps += 1

                    unique_state = get_formatted_obs_text(infos)
                    ep_trajs.append((unique_state, ac, score_pruned))
                true_score += score
                pruned_score += score_pruned
                game_trajs.append(ep_trajs)

            trajs[k] = game_trajs
            self.train_eps += 1

        if save_trajectories:
            save_file = \
                './results/state_action_graph/trajs_{}_prune_{}.pkl'. \
                format(difficulty_level, action_verb_to_prune)
            with open(save_file, 'wb') as fp:
                pickle.dump(trajs, fp)

        return true_score, pruned_score

    def obtain_admissible_verb(self, difficulty_level='easy',
                               max_steps=50, k_subgoal=True,
                               sub_goal_based_pruning=False,
                               num_games=5, save_trajs=False, num_repeats=1):

        state_change_action = []
        game_wise_action = {k: [] for k in range(num_games)}
        game_wise_score = {k: [] for k in range(num_games)}
        for _ in range(num_repeats):
            for game_no in range(num_games):
                logical_env = \
                    LogicalTWCQuantifier(difficulty_level,
                                         split='train', max_episode_steps=50,
                                         batch_size=None, game_number=game_no)
                obs, infos = logical_env.reset()
                prev_id = get_formatted_obs_text(infos)
                episodic_actions = []
                for step in range(max_steps):
                    actions = []
                    unnormed_prob = []
                    for adm_comm in logical_env.admissible_commands:
                        rule, x, y = logical_env.action2literal(adm_comm)
                        arity = int(y is not None) + int(x is not None)
                        if rule in self.admissible_verbs:
                            self.admissible_verbs[rule].append(arity)
                        else:
                            self.admissible_verbs[rule] = [arity]
                        actions.append(adm_comm)
                        unnormed_prob.append(1.0)

                    # Probability sampling
                    unnormed_prob = np.array(unnormed_prob) + 1e-10
                    normed_prob = \
                        np.array(unnormed_prob) / (np.sum(unnormed_prob))
                    sampled_action = \
                        np.random.choice(np.arange(0, len(actions)),
                                         p=normed_prob)
                    action_command = actions[sampled_action]
                    obs, score, dones, infos = logical_env.step(action_command)
                    self.steps += 1

                    episodic_actions.append(action_command)

                    curr_id = get_formatted_obs_text(infos)
                    if prev_id != curr_id:
                        rule, x, y = logical_env.action2literal(action_command)
                        state_change_action.append(rule)
                        prev_id = curr_id

                    if dones:
                        break
                self.train_eps += 1

                game_wise_action[game_no].append(episodic_actions)
                game_wise_score[game_no].append(score)

        admissible_verbs = {}
        for k, v in self.admissible_verbs.items():
            if min(v) > 0:
                admissible_verbs[k] = min(v)

        actions_to_remove = []

        if self.prune_by_state_change:
            for k, v in self.admissible_verbs.items():
                if k not in state_change_action:
                    admissible_verbs.pop(k, None)

        if sub_goal_based_pruning:
            print('Using sub goal pruning, k_subgoal={}'.format(k_subgoal))
            if k_subgoal != 0:
                k_subgoal = min(len(admissible_verbs), k_subgoal)
                candidate_actions_to_evaluate = \
                    random.sample(list(admissible_verbs.keys()), k_subgoal)
            else:
                candidate_actions_to_evaluate = list(admissible_verbs.keys())

            for prune_ac in candidate_actions_to_evaluate:
                true_score, pruned_score = \
                    self.evaluate_env_without_action_verb(
                        difficulty_level,
                        game_wise_action,
                        game_wise_score,
                        action_verb_to_prune=prune_ac,
                        save_trajectories=save_trajs
                    )

                if pruned_score == true_score and prune_ac != 'None':
                    actions_to_remove.append(prune_ac)

            for key in actions_to_remove:
                admissible_verbs.pop(key, None)

        self.admissible_verbs = admissible_verbs
        self.is_trained = {v: False for v, _ in self.admissible_verbs.items()}

    def obtain_onpolicy_buffer(self,
                               difficulty_level='easy',
                               max_steps=50,
                               verbose=False,
                               num_games=5,
                               gamma=0.5,
                               thres=0.1,
                               save_neg_buffer=False,
                               save_weights=False):
        adm_verbs = self.admissible_verbs
        total_score = 0
        max_total_score = 0
        steps = []

        buffer = {}
        weights = {}
        for k, v in adm_verbs.items():
            buffer[k] = {'pos': [], 'neg': []}
            weights[k] = {'pos': [], 'neg': []}

        for game_no in range(num_games):
            logical_env = \
                LogicalTWCQuantifier(difficulty_level,
                                     split='train',
                                     max_episode_steps=50,
                                     batch_size=None,
                                     game_number=game_no)
            obs, infos = logical_env.reset()
            facts = logical_env.get_logical_state(infos)
            prev_rew = 0
            prev_state_ids = [get_facts_state(facts), ]

            episodic_logs = {'obs': [], 'rew': [], 'act': [], 'is_novel': []}

            for step in range(max_steps):
                actions = []
                unnormed_prob = []
                facts_cskb = \
                    {k: v for k, v in facts.items()
                     if (k == 'atlocation' or k == 'is_instance')}

                for adm_comm in logical_env.admissible_commands:
                    rule, x, y = logical_env.action2literal(adm_comm)
                    if rule in adm_verbs:
                        actions.append(adm_comm)
                        unnormed_prob.append(1.0)

                # Probability sampling
                unnormed_prob = np.array(unnormed_prob) + 1e-10
                normed_prob = np.array(unnormed_prob) / (np.sum(unnormed_prob))
                sampled_action = \
                    np.random.choice(np.arange(0, len(actions)), p=normed_prob)
                action_command = actions[sampled_action]

                raw_obs = get_formatted_obs_text(infos)
                xsave = (raw_obs, actions, sampled_action, facts_cskb)
                obs, score, dones, infos = logical_env.step(action_command)
                self.steps += 1
                rew = score - prev_rew

                prev_rew = score
                facts = logical_env.get_logical_state(infos)
                curr_state_id = get_facts_state(facts)
                is_novel_state = curr_state_id in prev_state_ids
                if is_novel_state:
                    prev_state_ids.append(curr_state_id)

                episodic_logs['obs'].append(xsave)
                episodic_logs['rew'].append(rew)
                episodic_logs['act'].append(action_command)
                episodic_logs['is_novel'].append(is_novel_state)

                if dones:
                    break

            self.train_eps += 1
            # Monte Carlo estimate of state rewards:
            discounted_rewards = []
            disc_rew = 0
            for reward in reversed(episodic_logs['rew']):
                disc_rew = reward + (gamma * disc_rew)
                discounted_rewards.insert(0, disc_rew)

            for a, d_r, r, ob in zip(episodic_logs['act'],
                                     discounted_rewards,
                                     episodic_logs['rew'],
                                     episodic_logs['obs']):
                rule, x, y = logical_env.action2literal(a)
                if d_r >= thres:
                    buffer[rule]['pos'].append(ob)
                    weights[rule]['pos'].append(d_r)
                else:
                    if save_neg_buffer:
                        buffer[rule]['neg'].append(ob)
                        weights[rule]['neg'].append(d_r)

            steps.append(step)
            total_score += score
            max_total_score += infos['max_score']
            if verbose:
                print('Obtained score: %d/%d in %d steps' %
                      (score, infos['max_score'], step))

        perc_score = 100. * total_score / max_total_score
        mean_steps = np.mean(steps)
        if verbose:
            print('Total score : {}/{}'.format(total_score, max_total_score))
            print('Percentage score : ', perc_score)
            print('Average steps : ', mean_steps)

        if save_weights:
            return perc_score, mean_steps, buffer, weights
        else:
            return perc_score, mean_steps, buffer

    def reinforce_train_lnn(self,
                            max_iters=250,
                            verbose=False,
                            lam=0.0001,
                            prune_low_rewards=False):
        assert self.weights is not None, \
            'training with reinforce needs weights'
        self.pi.train()
        for key in self.buffer.keys():
            print('Training %s DD-LNN model' % key)
            merged_x = self.buffer[key]['pos_logic']
            if not len(merged_x):
                if verbose:
                    print('Skipping training of {} '
                          'LNN because it has no positive data'.format(key))
                continue
            self.is_trained[key] = True

            merged_x = merged_x.float()
            weights = np.array(self.weights[key]['pos'])
            if weights.min() < weights.max():
                weights = \
                    (weights - weights.min()) / \
                    (weights.max() - weights.min() + EPS)
            else:
                weights = weights / weights.max()

            if prune_low_rewards:
                idx = weights > 0.5
                merged_x = merged_x[idx]
                weights = weights[idx]

            if verbose:
                print('State - weights combination')
                for i in range(len(merged_x)):
                    print(merged_x[i], ' : ', weights[i])

            assert len(merged_x) == len(weights), \
                'The num in positive data and the weights should match'

            optimizer = \
                optim.Adamax(self.pi.models[key].parameters(),
                             lr=self.lr, weight_decay=self.wd)
            loss_fn = nn.BCELoss()

            pos_y = torch.tensor([1]).float()
            num_pos_samples = merged_x.size(0)
            merged_y = torch.ones((num_pos_samples,)).float()
            for iter in range(max_iters):
                self.pi.train()
                optimizer.zero_grad()
                if prune_low_rewards:
                    yhat = self.pi.forward_eval(merged_x, lnn_model_name=key)
                    loss = loss_fn(yhat.squeeze(0), merged_y)
                    constrained_loss = \
                        self.pi.compute_constraint_loss(lnn_model_name=key,
                                                        lam=lam)
                    loss = loss + constrained_loss
                else:
                    loss = 0.0
                    for i in range(len(merged_x)):
                        try:
                            yhat = \
                                self.pi.forward_eval(merged_x[i:i + 1],
                                                     lnn_model_name=key)
                            loss_i = weights[i] * \
                                loss_fn(yhat.squeeze(0), pos_y)
                            loss += loss_i
                        except BaseException:
                            print('Loss error : ')
                    loss = loss / (len(merged_x) * 1.0)
                    constrained_loss = \
                        self.pi.compute_constraint_loss(lnn_model_name=key,
                                                        lam=lam)
                    loss = loss + constrained_loss
                loss.backward()
                optimizer.step()
            if verbose:
                print('Iteration %d: %.3f' % (iter, loss.item()))
                self.pi.models[key].extract_weights()

    def display_rules(self, th=0.5):
        rules = self.extract_rules(th)
        for key in rules.keys():
            print(key +
                  ('(x,y)' if self.admissible_verbs[key] == 2 else '(x)') +
                  ' = ' + rules[key])

    def extract_rules(self, th=0.5):
        rules = dict()
        for key in self.pi.models:
            if self.is_trained[key]:
                pred_template = self.predicate_templates[key]
                beta, wts = self.pi.models[key].extract_weights()
                wts = wts.detach().numpy()
                if np.isnan(wts[0]):
                    rules[key] = 'True'
                else:
                    learned_pos_wts = \
                        [pred_template[k] for k, x in enumerate(wts) if x > th]
                    rules[key] = ' ∧ '.join(learned_pos_wts)
        return rules

    def test_policy(self,
                    difficulty_level='easy',
                    max_steps=50,
                    split='test',
                    verbose=False,
                    num_games=5):
        rest_amr = AMRSemParser(amr_server_ip=self.amr_server_ip,
                                amr_server_port=self.amr_server_port)
        adm_verbs = self.admissible_verbs
        self.pi.eval()
        total_score = 0.
        max_total_score = 0.
        steps = []
        for game_no in tqdm(range(num_games)):
            logical_env = \
                LogicalTWCQuantifier(difficulty_level,
                                     split=split,
                                     max_episode_steps=50,
                                     batch_size=None,
                                     game_number=game_no)
            obs, infos = logical_env.reset()
            facts = logical_env.get_logical_state(infos)
            obs_text = get_formatted_obs_text(infos)
            verbnet_facts, arity =\
                rest_amr.obs2facts(obs_text, mode=self.sem_parser_mode)
            verbnet_facts['atlocation'] = facts['atlocation']
            verbnet_facts['is_instance'] = facts['is_instance']

            for step in range(max_steps):
                actions = []
                unnormed_prob = []

                if verbose:
                    print('Obs: ', obs_text)
                    at_location_list = [list(x) for x in
                                        verbnet_facts['atlocation']]
                    print('at_location: ', at_location_list)
                    print('carry: ',
                          list(verbnet_facts[
                              'carry']) if 'carry' in verbnet_facts
                          else None)

                for adm_comm in logical_env.admissible_commands:
                    rule, x, y = logical_env.action2literal(adm_comm)
                    if rule in adm_verbs:
                        rule_arity = adm_verbs[rule]
                        logic_vector, all_preds = \
                            obtain_predicates_logic_vector(
                                rule_arity, x, y,
                                facts=verbnet_facts,
                                template=self.arity_predicate_templates)
                        actions.append(adm_comm)
                        logic_vector = logic_vector.unsqueeze(0)
                        yhat = \
                            self.pi.forward_eval(logic_vector,
                                                 lnn_model_name=rule)
                        unnormed_prob.append(yhat.item())
                        if verbose:
                            print('{} : {:.2f}'.format(adm_comm, yhat.item()))

                # Probability sampling
                unnormed_prob = np.array(unnormed_prob) + 1e-10
                normed_prob = np.array(unnormed_prob) / (np.sum(unnormed_prob))
                sampled_action = \
                    np.random.choice(np.arange(0, len(actions)),
                                     p=normed_prob)
                action_command = actions[sampled_action]

                if verbose:
                    print(action_command)

                obs, rew, dones, infos = logical_env.step(action_command)

                facts = logical_env.get_logical_state(infos)
                obs_text = get_formatted_obs_text(infos)
                verbnet_facts, arity = \
                    rest_amr.obs2facts(obs_text, mode=self.sem_parser_mode)
                verbnet_facts['atlocation'] = facts['atlocation']
                verbnet_facts['is_instance'] = facts['is_instance']

                if dones:
                    break

            steps.append(step + 1)
            total_score += rew
            max_total_score += infos['max_score']
            if verbose:
                print('Obtained score: {}/{} in {} steps'.format(
                    rew, infos['max_score'], step))

        perc_score = 100. * total_score / max_total_score
        mean_steps = np.mean(steps)

        rest_amr.save_cache()

        print('Evaluating on %s games' % difficulty_level)
        print('Total score : %.1f/%.1f' % (total_score, max_total_score))
        print('Percentage score : %.1f' % perc_score)
        print('Average steps : %.1f' % mean_steps)

        return perc_score, mean_steps

    def extract_fact2logic(self,
                           difficulty_level='easy', repeats=5,
                           mincount=None,
                           verbose=False):
        adm_verbs = self.admissible_verbs
        for repeats in range(repeats):
            perc_score, mean_steps, buffer, weights = \
                self.obtain_onpolicy_buffer(
                    difficulty_level=difficulty_level,
                    max_steps=50,
                    verbose=False,
                    num_games=5,
                    gamma=0.5,
                    thres=0.5,
                    save_neg_buffer=False,
                    save_weights=True)
            self.update_buffer(buffer, weights=weights, update_weights=True)

        all_obs = []
        for k in adm_verbs:
            all_obs += [item[0] for item in self.buffer[k]['pos']]

        all_obs = list(set(all_obs))
        print('Found {} observations'.format(len(all_obs)))

        mincount = 1 if mincount is None else int(mincount * len(all_obs))
        print('Mincount: ', mincount)

        all_train_preds, train_pred_count_dict, verbnet_facts = \
            get_verbnet_preds_from_obslist(
                all_obs,
                amr_server_ip=self.amr_server_ip,
                amr_server_port=self.amr_server_port,
                mincount=mincount,
                verbose=verbose,
                sem_parser_mode=self.sem_parser_mode,
                difficulty=difficulty_level
            )
        self.get_string_templates(all_train_preds)
        self.obtain_templates()

        num_by_arity = {k: len(v.split(';'))
                        for k, v in self.arity_predicate_templates.items()}
        pi = PolicyLNNTWC_SingleAnd(adm_verbs,
                                    use_constraint=True,
                                    num_by_arity=num_by_arity)
        self.init_lnn_model(pi)

        all_background_facts = \
            {k: [] for k in all_train_preds + ['atlocation', 'is_instance']}
        positive_facts = {}
        each_step_facts = []

        action2literal = Action2Literal()
        for k in adm_verbs:
            self.buffer[k]['pos_logic'] = []
            positive_facts[k] = []
            for item in self.buffer[k]['pos']:
                raw_obs, actions, sampled_action, facts_cskb = item

                extracted_facts = verbnet_facts[raw_obs]

                extracted_facts = combine_cs_facts(extracted_facts, facts_cskb)

                for key, value in extracted_facts.items():
                    if key in all_background_facts:
                        all_background_facts[key] += value

                ac_comm = actions[sampled_action]
                rule, x, y = action2literal(ac_comm)
                if y is None:
                    positive_facts[k].append(x)
                    each_step_facts.append({'B': extracted_facts,
                                            'P': {rule: [x]}})
                else:
                    positive_facts[k].append((x, y))
                    each_step_facts.append({'B': extracted_facts,
                                            'P': {rule: [(x, y)]}})
                logic_vector, all_preds = \
                    obtain_predicates_logic_vector(
                        adm_verbs[rule], x, y,
                        facts=extracted_facts,
                        template=self.arity_predicate_templates)
                self.buffer[k]['pos_logic'].append(logic_vector.unsqueeze(0))

        for k, v in adm_verbs.items():
            if len(self.buffer[k]['pos_logic']):
                self.buffer[k]['pos_logic'] = \
                    torch.cat(self.buffer[k]['pos_logic'], 0)

        self.all_background_facts = all_background_facts
        self.positive_facts = positive_facts
        self.each_step_facts = each_step_facts

    def save_pickel(self, pickel_path):
        data = {
            'pi': self.pi,
            'admissible_verbs': self.admissible_verbs,
            'arity_predicate_templates': self.arity_predicate_templates,
            'is_trained': self.is_trained,
            'predicate_templates': self.predicate_templates,
        }
        with open(pickel_path, 'wb') as f:
            pickle.dump(data, f)

    def load_pickel(self, pickel_path):
        with open(pickel_path, 'rb') as f:
            data = pickle.load(f)
        self.pi = data['pi']
        self.admissible_verbs = data['admissible_verbs']
        self.arity_predicate_templates = data['arity_predicate_templates']
        self.is_trained = data['is_trained']
        self.predicate_templates = data['predicate_templates']
