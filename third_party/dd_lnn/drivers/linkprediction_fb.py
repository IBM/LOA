import sys
sys.path.append('../src/meta_rule/')
sys.path.append('../dd_lnn/')

import random
import time
import copy
import argparse
from meta_interpretive import BaseMetaPredicate, MetaRule, Project, DisjunctionRule
from train_test import score, align_labels
from read import load_data, load_metadata, load_labels

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
    true_links = true.loc[(true[filter_attr] == filter_val)][[needle_attr]].to_numpy()[:,0].tolist()

    filtered_result = []
    score = None
    for idx, r in result.iterrows():
        needle_val1 = r[needle_attr]
        prediction1 = r['prediction']
        
        if needle_val1 == needle_val:
            score = prediction1
            
        if needle_val1 not in true_links:
            filtered_result.append([needle_val1, prediction1])
            
    haspath = None
    if not score:
        haspath = False
        score = 0.0
    else:
        haspath = True

    rank = 1 + len([pair for pair in filtered_result if pair[1] > score])
            
    return score, rank, haspath

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
    step = 1e-2
    epochs = 50
    
    y = align_labels(meta, df, label)    
    print("done label alignment (" + str(time.time()-begtime) + "s)")

    grouped = meta.df.groupby([attr_name + "0"])
    numgroups = len(grouped)
    src_groups = []
    pos = 0
    for name, indices in grouped.groups.items():
        if pos % 100000 == 0:
            print(str(pos) + "/" + str(numgroups))

        idx = indices.to_numpy().tolist()
        if torch.sum(y[idx,:]) > 0:
            src_groups += [idx]
            
        pos = pos + 1

    print("Added: " + str(len(src_groups)) + " (" + str(time.time()-begtime)+ "s)")
        
    if len(src_groups) > 0:
        data = TensorDataset(torch.arange(len(src_groups)))
        loader = DataLoader(data, batch_size=32, shuffle=True)
        
        optimizer = optim.Adam(meta.parameters(), lr=step)
        loss_fn = nn.MarginRankingLoss(margin=0.5, reduction="mean")
    
        iter = 0
        for epoch in range(epochs):
            train_loss = 0.0
            for batch in loader:
                pos_list = []
                pos_len = []
                neg_list = []
                neg_len = []
                for idx in batch[0]:
                    pos_idx = [i for i in src_groups[idx] if y[i] == 1]
                    neg_idx = [i for i in src_groups[idx] if y[i] == 0]

                    neg_idx = random.sample(neg_idx, min(len(pos_idx), len(neg_idx)))

                    pos_list += pos_idx
                    pos_len += [len(pos_idx)]
                    neg_list += neg_idx
                    neg_len += [len(neg_idx)]

                meta.train()
                optimizer.zero_grad()
                yhat, slacks = meta(pos_list + neg_list)

                pos_mat = torch.zeros(0)
                neg_mat = torch.zeros(0)
                curr_pos = 0
                curr_neg = sum(pos_len) 
                for pos_cnt, neg_cnt in zip(pos_len, neg_len):
                    pos_yhat = yhat[curr_pos:curr_pos+pos_cnt]
                    pos_yhat = pos_yhat.repeat(neg_cnt, 1)[:,0]
                    neg_yhat = yhat[curr_neg:curr_neg+neg_cnt]
                    neg_yhat = torch.repeat_interleave(neg_yhat, pos_cnt, dim=0)[:,0]
                    
                    curr_pos += pos_cnt
                    curr_neg += neg_cnt
                    pos_mat = torch.cat((pos_mat, pos_yhat))
                    neg_mat = torch.cat((neg_mat, neg_yhat))

                loss = loss_fn(pos_mat, neg_mat, torch.ones(pos_mat.size()[0]))
                train_loss = train_loss + loss.item() * pos_mat.size()[0]
                
                print("Epoch " + str(epoch) + " (iteration=" + str(iter) + "): " + str(loss.item()))
                
                loss.backward()
                optimizer.step()
                iter = iter + 1

            print("Epoch=" + str(epoch) + " +ve.loss=" + str(train_loss))
        
    print("done training (" +str(time.time()-begtime) + "s)")
            
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

    haspath = 0
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
        score4dest, rank4dest, haspath_bool = eval(yhat, attr_name + "0", src, attr_name + "3", dest, true_links)
        if haspath_bool:
            haspath += 1
            
        score4src_inv, rank4src_inv, haspath_bool = eval(yhat, attr_name + "3", dest, attr_name + "0", src, inv_true_links)
        print("(" + str(score4src_inv) + ", " + str(rank4src_inv) + "), (" \
              + str(score4dest) + ", " + str(rank4dest) + ") (" + str(time.time()-begtime) + "s)")

        rank = min(rank4dest, rank4src_inv)
        #rank = min(rank4dest, rank4dest_inv, rank4src, rank4src_inv)
        
        mrr += 1.0 / rank
        hits10 = hits10 + 1 if rank <= 10 else hits10
        hits1 = hits1 + 1 if rank <= 1 else hits1
        hits3 = hits3 + 1 if rank <= 3 else hits3
        ranks += [rank]

    numqueries = labels_df_test.shape[0]
    print("queries=" + str(numqueries) \
          + " mrr=" + str(mrr) \
          + " hits10=" + str(hits10) \
          + " hits3=" + str(hits3) \
          + " hits1=" + str(hits1) \
          + " paths=" + str(haspath))
    
