import sys
sys.path.append('../src/meta_rule/')
sys.path.append('../dd_lnn/')

import argparse
from meta_interpretive import BasePredicate, MetaPredicate, MetaRule, Project, DisjunctionRule
from train_test import train, test, align_labels
from read import load_data, load_metadata, load_labels

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from utils import evaluate, MyBatchSampler
from torch.utils.data import DataLoader, TensorDataset

#driver for experiments with compare and comply data (only rules)
if __name__ == "__main__":
    
    alpha = 0.95
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-bktr", "--background_train", required=True, help="file containing background for training")
    parser.add_argument("-ftr", "--facts_train", required=True, help="file containing facts for training")
    parser.add_argument("-postr", "--pos_train", required=True, help="file containing positive labels for training")
    parser.add_argument("-negtr", "--neg_train", required=True, help="file containing negative labels for training")
    parser.add_argument("-bkte", "--background_test", required=True, help="file containing background for testing")
    parser.add_argument("-fte", "--facts_test", required=True, help="file containing facts for testing")
    parser.add_argument("-poste", "--pos_test", required=True, help="file containing positive labels for testing")
    parser.add_argument("-negte", "--neg_test", required=True, help="file containing negative labels for testing")
    parser.add_argument("-k", "--num_rules", required=True, help="number of rules to learn")
    parser.add_argument("-m", "--rule_length", required=True, help="number of action predicates in each rule")
    
    cl_args = parser.parse_args()
    background_fname_train = cl_args.background_train
    facts_fname_train = cl_args.facts_train
    pos_fname_train = cl_args.pos_train
    neg_fname_train = cl_args.neg_train
    background_fname_test = cl_args.background_test
    facts_fname_test = cl_args.facts_test
    pos_fname_test = cl_args.pos_test
    neg_fname_test = cl_args.neg_test
    rulelen = int(cl_args.rule_length)
    numrules = int(cl_args.num_rules)
    
    dfs_train = load_metadata(background_fname_train)
    load_data(facts_fname_train, dfs_train)
    load_labels(pos_fname_train, dfs_train, 1.0)
    load_labels(neg_fname_train, dfs_train, 0.0)
    
    tails_relation_train = None
    labels_df_train = None
    action_attr = None
    sentence_attr = None
    action_relations_train = []
    action_rel_names_train = []
    for name, df in dfs_train.items():
        colnames = df.columns.values.tolist()
        if "Label" in colnames:
            labels_df_train = df
            sentence_attr = colnames[0]
        elif len(colnames) == 1:
            action_rel_names_train.append(name)
            action_relations_train.append(BasePredicate(df))
            action_attr = colnames[0]
        else:
            tails_relation_train = BasePredicate(df)

    projs = []
    for i in range(numrules):
        join_attrs = []
        body = [tails_relation_train]
        for j in range(rulelen):
            body.append(MetaPredicate(action_relations_train))
            join_attrs.append([[action_attr], [action_attr]])
            
        join = MetaRule(body, join_attrs, alpha, True)
        proj = Project(join, [sentence_attr])
        projs.append(proj)
    disj = DisjunctionRule(projs, alpha, 0.5) if numrules > 1 else DisjunctionRule(projs, alpha, 0.0)

    #train(disj, labels_df_train, 'Label', 1e-1, 64, 5, True) #1e-2,
    meta = disj
    df = labels_df_train
    label = 'Label'
    step = 1e-3
    batch = 64
    epochs = 50
    use_balanced = True
    y = align_labels(meta, df, label)
    data = TensorDataset(torch.arange(y.size()[0]), y)
    bal_batcher = MyBatchSampler(y)
    bal_loader = DataLoader(data, sampler=bal_batcher, batch_size=batch, shuffle=False)
    seq_loader = DataLoader(data, batch_size=batch, shuffle=True)
    loader = bal_loader if use_balanced else seq_loader
    optimizer = optim.Adam(meta.parameters(), lr=step)
    loss_fn = nn.BCELoss()

    iter = 0
    for epoch in range(epochs):
        for idx, yb in loader:
            meta.train()
            optimizer.zero_grad()
            yhat, slacks = meta(idx)
            #loss = -torch.sum(func.logsigmoid(torch.mul(meta(idx), 2*yb-1)))
            loss = loss_fn(yhat, yb) + 0.0 * slacks
            print("Epoch " + str(epoch) + " (iteration=" + str(iter) + "): " + str(loss.item()) + ", " + str(slacks.item()))

            loss.backward()
            optimizer.step()
            iter = iter + 1

        if use_balanced:
            bal_batcher.shuffle()
            bal_loader = DataLoader(data, sampler=bal_batcher, batch_size=batch, shuffle=False)
            loader = bal_loader
                
        total_loss = 0.0
        with torch.no_grad():
            meta.eval()
            for idx, yb in seq_loader:
                yhat, slacks = meta(idx)
                #loss = -torch.sum(func.logsigmoid(torch.mul(meta(idx), 2*yb-1)))
                loss = loss_fn(yhat, yb)
                total_loss = total_loss + loss.item() * len(idx)

            print("Epoch=" + str(epoch) + " Loss=" + str(total_loss))

    precision, recall, f1, yhat = test(disj, labels_df_train, 'Label')
    print("P/R/F1=" + str(precision) + "/" + str(recall) + "/" + str(f1))
            
    dfs_test = load_metadata(background_fname_test)
    load_data(facts_fname_test, dfs_test)
    load_labels(pos_fname_test, dfs_test, 1.0)
    load_labels(neg_fname_test, dfs_test, 0.0)

    tails_relation_test = None
    labels_df_test = None
    for name, df in dfs_test.items():
        colnames = df.columns.values.tolist()
        if "Label" in colnames:
            labels_df = df
        elif len(colnames) == 2:
            tails_relation_test = BasePredicate(df)

    action_relations_test = []
    for name in action_rel_names_train:
        action_relations_test.append(BasePredicate(dfs_test[name]))

    old_disj = disj
    projs = []
    for i in range(numrules):
        join_attrs = []
        body = [tails_relation_test]
        rule_i = old_disj.rules[i].rule
        for j in range(rulelen):
            action = MetaPredicate(action_relations_test)
            action.alpha = rule_i.body[1+j].alpha #body[0] is tails
            body.append(action)
            join_attrs.append([[action_attr], [action_attr]])

        join = MetaRule(body, join_attrs, alpha, True)
        join.AND = rule_i.AND

        #beta, argument_wts = join.AND.cdd()
        #print("beta: " + str(beta.item()) + " argument weights: " + str(argument_wts.detach()))
        
        proj = Project(join, [sentence_attr])
        projs.append(proj)
    disj = DisjunctionRule(projs, alpha, 0.0)
    disj.OR = old_disj.OR

    precision, recall, f1, yhat = test(disj, labels_df, 'Label')
    print("P/R/F1=" + str(precision) + "/" + str(recall) + "/" + str(f1))

    print(disj.rules[0].rule.AND.print_truth_table())
