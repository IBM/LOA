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

parser.add_argument('--num_repeat_pre', type=int,
                    help='Number of times the pre-training repeats',
                    default=2)

parser.add_argument('--num_repeat_train', type=int,
                    help='Number of times the training repeats',
                    default=15)

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')

parser.add_argument('--k_subgoal', type=int,
                    default=6, help='Number of actions to prune')

default_amr_server_ip = 'localhost'
parser.add_argument('--amr_server_ip', type=str,
                    default=default_amr_server_ip, help='IP for AMR server')

default_amr_server_port = 0
parser.add_argument('--amr_server_port', type=int,
                    default=default_amr_server_port,
                    help='Port number for AMR server')


args = parser.parse_args()

if args.amr_server_ip == default_amr_server_ip:
    env_amr_server_ip = os.getenv('LOA_AMR_SERVER_IP', default_amr_server_ip)
else:
    env_amr_server_ip = args.amr_server_ip

if args.amr_server_port == default_amr_server_port:
    env_amr_server_port = int(os.getenv('LOA_AMR_SERVER_PORT',
                                        str(default_amr_server_port)))
else:
    env_amr_server_port = args.amr_server_port

print('AMR IP: %s, PORT: %s' % (env_amr_server_ip, env_amr_server_port))

filename = \
    'loa-twc-dl%s-np%d-nt%d' % \
    (args.difficulty_level, args.num_repeat_pre, args.num_repeat_train) + \
    '-ks%d-sp%s' % (args.k_subgoal, args.sem_parser_mode)

results_folder = 'results/'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

pkl_filepath = results_folder + filename + '.pkl'

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

loa_agent = \
    LOAAgent(
        difficulty_level=args.difficulty_level,
        amr_server_ip=env_amr_server_ip,
        amr_server_port=env_amr_server_port,
        admissible_verbs=None,
        sem_parser_mode=args.sem_parser_mode,
        num_repeats_pre=args.num_repeat_pre,
    )

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
    LOAAgent(
        difficulty_level=args.difficulty_level,
        amr_server_ip=env_amr_server_ip,
        amr_server_port=env_amr_server_port,
        admissible_verbs=None,
        sem_parser_mode=args.sem_parser_mode
    )

loa_agent.load_pickel(pkl_filepath)
print('Trained rules:')
loa_agent.display_rules()

perc_score, mean_steps = \
    loa_agent.test_policy(difficulty_level=args.difficulty_level,
                          max_steps=50, split='test',
                          verbose=False, num_games=5)

print('Test score:', perc_score)
print('Test steps:', mean_steps)
