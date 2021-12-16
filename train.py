import argparse
import os.path
import random
import time

import numpy as np

import torch
from loa_agent import LOAAgent

parser = argparse.ArgumentParser(description='Train LOA')

parser.add_argument('--difficulty_level',
                    type=str, help='Difficulty level of the TWC game',
                    default='easy', choices=['easy', 'medium', 'hard'])

parser.add_argument('--sem_parser_mode',
                    type=str, help='Which mode to use for ablation',
                    default='both',
                    choices=['both', 'verbnet', 'propbank', 'none'])

parser.add_argument('--num_repeat_prior', type=int,
                    help='Number of times the pre-training repeats',
                    default=2)

parser.add_argument('--num_repeat_train', type=int,
                    help='Number of times the training repeats',
                    default=15)

parser.add_argument('--not_prune_by_state_change',
                    help='Will not prune actions by state change',
                    action='store_true')

parser.add_argument('--not_subgoal_based_pruning',
                    help='Will not use sub-goal based action pruning',
                    action='store_true')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')

parser.add_argument('--k_subgoal', type=int,
                    default=6, help='Number of actions to prune')

parser.add_argument('--amr_server_ip', type=str,
                    default='localhost', help='IP for AMR server')

parser.add_argument('--amr_server_port', type=int,
                    default=None, help='Port number for AMR server')


args = parser.parse_args()

filename = \
    'loa-twc-dl%s-np%d-nt%d-ps%d' % \
    (args.difficulty_level, args.num_repeat_prior, args.num_repeat_train,
     0 if args.not_prune_by_state_change else 1) + \
    '-ks%d-sp%s' % \
    (0 if args.not_subgoal_based_pruning else args.k_subgoal,
     args.sem_parser_mode)

results_folder = 'results/'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

pkl_filepath = results_folder + filename + '.pkl'

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

loa_agent = \
    LOAAgent(amr_server_ip=args.amr_server_ip,
             amr_server_port=args.amr_server_port,
             admissible_verbs=None,
             prune_by_state_change=not args.not_prune_by_state_change,
             sem_parser_mode=args.sem_parser_mode)

loa_agent.obtain_admissible_verb(
    difficulty_level=args.difficulty_level, k_subgoal=args.k_subgoal,
    sub_goal_based_pruning=not args.not_subgoal_based_pruning,
    save_trajs=False, num_repeats=args.num_repeat_prior)

print('Admissible verbs: ', loa_agent.admissible_verbs)

loa_agent.extract_fact2logic(difficulty_level=args.difficulty_level,
                             repeats=args.num_repeat_train,
                             verbose=False, mincount=0.25)

starting_time = time.time()
loa_agent.reinforce_train_lnn(max_iters=1000,
                              verbose=False,
                              prune_low_rewards=True,
                              lam=0.0001)
print('Training time: %.2f' % (time.time() - starting_time))

print('Train eps:', loa_agent.train_eps)
print('Train steps:', loa_agent.steps)

loa_agent.save_pickel(pkl_filepath)

loa_agent = \
    LOAAgent(amr_server_ip=args.amr_server_ip,
             amr_server_port=args.amr_server_port,
             admissible_verbs=None,
             prune_by_state_change=not args.not_prune_by_state_change,
             sem_parser_mode=args.sem_parser_mode)

loa_agent.load_pickel(pkl_filepath)
print('Trained rules:')
loa_agent.display_rules()

perc_score, mean_steps = \
    loa_agent.test_policy(difficulty_level=args.difficulty_level,
                          max_steps=50, split='test',
                          verbose=False, num_games=5)

print('Test score:', perc_score)
print('Test steps:', mean_steps)
