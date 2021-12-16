import sys
sys.path.append('../src/meta_rule/')
sys.path.append('../dd_lnn/')

import time
import copy
import argparse
from meta_interpretive import BaseMetaPredicate, MetaRule, Project, DisjunctionRule
from train_test import score, align_labels
from read import load_data, load_metadata, load_labels, add_facts

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from utils import evaluate, MyBatchSampler
from torch.utils.data import DataLoader, TensorDataset

def eval(df, filter_attr, filter_val, needle_attr, needle_val, true):
    result = df.loc[(df[filter_attr] == filter_val)][[needle_attr, "prediction"]]
    
    filtered_result = []
    score = None
    for idx, r in result.iterrows():
        needle_val1 = r[needle_attr]
        prediction1 = r['prediction']
        
        if needle_val1 == needle_val:
            score = prediction1
            
        check = true.loc[(true[filter_attr] == filter_val) & (true[needle_attr] == needle_val1)]
        if check.shape[0] == 0:
            filtered_result.append([needle_val1, prediction1])
                
    if not score:
        score = 0.0

    rank = 1 + len([pair for pair in filtered_result if pair[1] > score])
            
    return score, rank

#driver for experiments with wn18 link prediction data
if __name__ == "__main__":
    begtime = time.time()
    alpha = 0.95
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-bk", "--background", required=True, help="file containing schema information")
    parser.add_argument("-ftr", "--facts_train", required=True, help="file containing facts for training")
    parser.add_argument("-fte", "--facts_test", required=True, help="file containing facts for testing")
    parser.add_argument("-t", "--target", required=True, help="the target predicate")

    cl_args = parser.parse_args()
    background_fname = cl_args.background
    facts_fname_train = cl_args.facts_train
    facts_fname_test = cl_args.facts_test
    target = cl_args.target

    dfs_train = load_metadata(background_fname)
    load_data(facts_fname_train, dfs_train)

    tests = ["./../data/nell/nell995/new_test_agentbelongstoorganization.txt", \
             "./../data/nell/nell995/new_test_athletehomestadium.txt", \
             "./../data/nell/nell995/new_test_athleteplaysforteam.txt", \
             "./../data/nell/nell995/new_test_athleteplaysinleague.txt", \
             "./../data/nell/nell995/new_test_athleteplayssport.txt", \
             "./../data/nell/nell995/new_test_organizationheadquarteredincity.txt", \
             "./../data/nell/nell995/new_test_organizationhiredperson.txt", \
             "./../data/nell/nell995/new_test_personborninlocation.txt", \
             "./../data/nell/nell995/new_test_personleadsorganization.txt", \
             "./../data/nell/nell995/new_test_teamplaysinleague.txt", \
             "./../data/nell/nell995/new_test_teamplayssport.txt", \
             "./../data/nell/nell995/new_test_worksfor.txt"]

    for f in tests:
        if f != facts_fname_test:
            add_facts(f, dfs_train)
    
    print("done reading (" + str(time.time()-begtime) + ")")
    
    attr_name = None
    labels_df_train = None
    relations_train = []
    rel_names_train = []
    for name, df in dfs_train.items():
        colnames = df.columns.values.tolist()
        attr_name = colnames[0]
        df.columns = [attr_name + "0", attr_name + "1"]
        
        if target == name:
            labels_df_train = df
        else:
            rel_names_train.append(name)
            relations_train.append(df)

            #creating inv
            name = "inv" + name
            cols = df.columns.tolist()
            inv_df = df[[cols[1], cols[0]]].copy()
            inv_df.columns = [cols[0], cols[1]]
            rel_names_train.append(name)
            relations_train.append(inv_df)

    labels_df_train.columns = [attr_name + "0", attr_name + "3"]
    labels_df_train['Label'] = 1.0
            
    body0 = BaseMetaPredicate(relations_train)
    print("done body0 (" + str(time.time()-begtime) + "s)")
    body1 = copy.deepcopy(body0)
    body1.df.columns = [attr_name + "2", attr_name + "3"]
    join = MetaRule([body0, body1], [[[attr_name + "1"], [attr_name + "2"]]], alpha, False)
    print("done join (" + str(time.time()-begtime) + "s)")
    proj = Project(join, [attr_name + "0", attr_name + "3"])
    print("done project (" + str(time.time()-begtime) + "s)")
    metap = copy.deepcopy(body0)
    metap.df.columns = [attr_name + "0", attr_name + "3"]
    disj = DisjunctionRule([metap, proj], alpha, 0)
    print("done disjunction (" + str(time.time()-begtime) + "s)")
        
    meta = disj
    df = labels_df_train
    label = 'Label'
    step = 1e-3
    batch_size = 32
    epochs = 1000
    y = align_labels(meta, df, label)

    #tmp_df = meta.df
    #tmp_df["Label"] = y.numpy()
    #links = tmp_df.loc[(tmp_df[label] == 1)]
    #for index, row in links.iterrows():
    #    reverse = tmp_df.loc[(tmp_df[attr_name + "0"] == row[attr_name + "3"]) & (tmp_df[attr_name + "3"] == row[attr_name + "0"])]        
    #    if reverse.shape[0] > 0:
    #        tmp_df.loc[(tmp_df[attr_name + "0"] == row[attr_name + "3"]) & (tmp_df[attr_name + "3"] == row[attr_name + "0"]), [label]] = 1
    #y = torch.FloatTensor(tmp_df[[label]].values)
    #tmp_df.drop([label], axis=1, inplace=True)
    
    print("done label alignment (" + str(time.time()-begtime) + "s)")
    data = TensorDataset(torch.arange(y.size()[0]), y)
    all_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    pos_idx = np.nonzero(y.numpy())[0].tolist()
    print("+ve: " + str(len(pos_idx)))
    
    if len(pos_idx) > 0:
        pos_loader = DataLoader(TensorDataset(torch.LongTensor(pos_idx)), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(meta.parameters(), lr=step)
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    
        iter = 0
        for epoch in range(epochs):
            pos_loss = 0.0
            for idx in pos_loader:
                yb = torch.ones(idx[0].size()[0], 1)
                meta.train()
                optimizer.zero_grad()
                yhat, slacks = meta(idx[0])
                #yhat = torch.min(yhat, torch.ones(yhat.size()[0], 1)) #making sure yhat does not exceed 1
                #loss = - torch.sum(yhat)
                loss = loss_fn(yhat, yb)
                pos_loss = pos_loss + loss.item()
                print("Epoch " + str(epoch) + " (iteration=" + str(iter) + "): " + str(loss.item()))
                
                loss.backward()
                optimizer.step()
                iter = iter + 1

            print("Epoch=" + str(epoch) + " +ve.loss=" + str(pos_loss))
        
    print("done training (" + str(time.time()-begtime) + "s)")
            
    dfs_test = load_metadata(background_fname)
    load_data(facts_fname_test, dfs_test)

    labels_df_test = dfs_test[target]
    labels_df_test.columns = [attr_name + "0", attr_name + "3"]
    labels_df_test['Label'] = 1.0

    true_links = pd.concat([labels_df_train, labels_df_test])
    true_links.drop(['Label'], axis=1, inplace=True)

    cols = true_links.columns.tolist()
    inv_true_links = true_links[[cols[1], cols[0]]].copy()
    inv_true_links.columns = [cols[0], cols[1]]
    
    yhat = score(disj, 1024)
    print("done evaluation (" + str(time.time()-begtime) + "s)")

    mrr = 0.0
    hits10 = 0.0
    hits1 = 0.0
    hits3 = 0.0
    ranks = []
    for index, row in labels_df_test.iterrows():
        src = row[attr_name + "0"]
        dest = row[attr_name + "3"]
        print("query: " + src + " " + dest)

        #looking for dest; minerva evaluates this way
        score4dest, rank4dest = eval(yhat, attr_name + "0", src, attr_name + "3", dest, true_links)
        #score4dest_inv, rank4dest_inv = eval(yhat, attr_name + "0", src, attr_name + "3", dest, inv_true_links)
        #looking for src; neurallp evaluates this way
        #score4src, rank4src = eval(yhat, attr_name + "3", dest, attr_name + "0", src, true_links)
        score4src_inv, rank4src_inv = eval(yhat, attr_name + "3", dest, attr_name + "0", src, inv_true_links)
        print("(" + str(score4src_inv) + ", " + str(rank4src_inv) + "), (" \
              + str(score4dest) + ", " + str(rank4dest) + ")")

        rank = min(rank4dest, rank4src_inv)
        #rank = min(rank4dest, rank4dest_inv, rank4src, rank4src_inv)
        
        mrr += 1.0 / rank
        hits10 = hits10 + 1 if rank <= 10 else hits10
        hits1 = hits1 + 1 if rank <= 1 else hits1
        hits3 = hits3 + 1 if rank <= 3 else hits3
        ranks += [rank]

    numqueries = labels_df_test.shape[0]
    print("queries=" + str(numqueries) + " mrr=" + str(mrr) + " hits10=" + str(hits10) + " hits3=" + str(hits3) + " hits1=" + str(hits1))
    
