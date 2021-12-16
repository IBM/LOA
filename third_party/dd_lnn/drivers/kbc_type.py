import sys
sys.path.append('../src/meta_rule/')
sys.path.append('../dd_lnn/')

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
    body0.df.columns = [attr_name + "0", attr_name + "1"]
    body1 = copy.deepcopy(body0)
    body1.df.columns = [attr_name + "2", attr_name + "3"]
    join = MetaRule([body0, body1], [[[attr_name + "1"], [attr_name + "2"]]], alpha, False)
    print("done join (" + str(time.time()-begtime) + "s)")
    proj = Project(join, [attr_name + "0", attr_name + "3"])
    print("done project (" + str(time.time()-begtime) + "s)")
        
    meta = proj
    df = labels_df_train
    label = 'Label'
    step = 1e-3
    batch_size = 32
    epochs = 15
    y = align_labels(meta, df, label)

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

            print("Epoch=" + str(epoch) + " +ve.loss=" + str(pos_loss) + " (" + str(time.time()-begtime) + "s)")
        
    print("done training (" +str(time.time()-begtime) + "s)")

    beta, params = body0.alpha['default'].get_params()
    params = params.detach().numpy()[0].tolist()
    print("LNN-pred (0): ", end='')
    for rel, wt in sorted(zip(rel_names_train, params), key=lambda t: t[1], reverse=True):
        np.set_printoptions(precision=3, suppress=True)
        print(rel + " (" + str(wt) + ") ", end='')
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("beta (" + str(beta[0,0]) + ")")
    beta, params = body1.alpha['default'].get_params()
    params = params.detach().numpy()[0].tolist()
    print("LNN-pred (1): ", end='')
    for rel, wt in sorted(zip(rel_names_train, params), key=lambda t: t[1], reverse=True):
        np.set_printoptions(precision=3, suppress=True)
        print(rel + " (" + str(wt) + ") ", end='')
    beta = beta.detach().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print("beta (" + str(beta[0,0]) + ")")
    lnn_beta, lnn_wts, slacks = join.AND['default'].cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("LNN AND beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))

    #get the target relation's type
    target_relname = -1
    with open('./../data/dbpedia/insnet/relmap') as f:
        for line in f:
            line = line.strip('\n')
            relname, arrow, relid = line.split(" ")
            if target == relid:
                target_relname = relname

    domain_name = "null"
    range_name = "null"
    with open('./../data/dbpedia/insnet/reltype') as f:
        for line in f:
            line = line.strip('\n')
            relname, _domain, _range = line.split(' ')
            if relname == target_relname:
                dummy, domain_name = _domain.split('=')
                dummy, range_name = _range.split('=')

    print(target_relname + " " + domain_name + " " + range_name)
    dest2type = pd.read_csv('./../data/dbpedia/insnet/instancetype', sep=' ', header=0, names=['id', 'type'])
    
    dfs_test = load_metadata(background_fname)
    load_data(facts_fname_test, dfs_test)

    labels_df_test = dfs_test[target]
    labels_df_test.columns = [attr_name + "0", attr_name + "3"]

    true_links = pd.concat([labels_df_train, labels_df_test])
    true_links.drop(['Label'], axis=1, inplace=True)

    cols = true_links.columns.tolist()
    inv_true_links = true_links[[cols[1], cols[0]]].copy()
    inv_true_links.columns = [cols[0], cols[1]]
    
    yhat = score(proj, 1024)
    yhat[['new_' + attr_name + "3"]] = yhat[[attr_name + "3"]].apply(pd.to_numeric)
    yhat = pd.merge(yhat, dest2type, \
                    left_on=['new_' + attr_name + "3"], \
                    right_on=['id'], \
                    how='left')
    yhat.drop(['new_' + attr_name + "3", "id"], axis=1, inplace=True)
    yhat.fillna("null", inplace=True)
    print("done evaluation (" + str(time.time()-begtime) + "s)")

    mrr = 0.0
    hits10 = 0.0
    hits1 = 0.0
    hits3 = 0.0
    ranks = []

    scores = pd.merge(labels_df_test, yhat,
                      left_on=[attr_name + "0", attr_name + "3"],
                      right_on=[attr_name + "0", attr_name + "3"],
                      how='left')
    scores.fillna(0, inplace=True)
    scores = scores \
        .groupby([attr_name + "0", attr_name + "3"]) \
        .agg({'prediction': 'max'}) \
        .reset_index()
    print("done retrieving scores (" + str(time.time()-begtime) + "s)")

    false_links = None
    if range_name == 'null':
        false_links = pd.merge(yhat, true_links,
                               left_on=[attr_name + "0", attr_name + "3"],
                               right_on=[attr_name + "0", attr_name + "3"],
                               how='left', indicator=True) \
                        .query("_merge != 'both'") \
                        .drop('_merge', axis=1)
    else:
        false_links = pd.merge(yhat, true_links,
                               left_on=[attr_name + "0", attr_name + "3"],
                               right_on=[attr_name + "0", attr_name + "3"],
                               how='left', indicator=True) \
                        .query("_merge != 'both' and (type == '" + range_name + "' or type == 'null')") \
                        .drop('_merge', axis=1)
    false_links = false_links \
        .groupby([attr_name + "0"])['prediction'] \
        .agg(list)
    print("done join for false links (" + str(time.time()-begtime) + "s)")
    src2false = {}
    for index, value in false_links.iteritems():
        src2false[index] = value
    print("done groupby for false links (" + str(time.time()-begtime) + "s)")

    inv_false_links = None
    if range_name == 'null':
        inv_false_links = pd.merge(yhat, inv_true_links,
                                   left_on=[attr_name + "0", attr_name + "3"],
                                   right_on=[attr_name + "3", attr_name + "0"],
                                   suffixes = ['_left', '_right'],
                                   how='left', indicator=True) \
                            .query("_merge != 'both'") \
                            .drop('_merge', axis=1) \
                            .groupby([attr_name + "0_left"])['prediction'] \
                            .agg(list)
    else:        
        inv_false_links = pd.merge(yhat, inv_true_links,
                                   left_on=[attr_name + "0", attr_name + "3"],
                                   right_on=[attr_name + "3", attr_name + "0"],
                                   suffixes = ['_left', '_right'],
                                   how='left', indicator=True) \
                            .query("_merge != 'both' and (type == '" + range_name + "' or type == 'null')") \
                            .drop('_merge', axis=1) \
                            .groupby([attr_name + "0_left"])['prediction'] \
                            .agg(list)
    print("done join for inv_false links (" + str(time.time()-begtime) + "s)")
    src2inv_false = {}
    for index, value in inv_false_links.iteritems():
        src2inv_false[index] = value
    print("done groubby for inv_false links (" + str(time.time()-begtime) + "s)")
    
    for index, row in scores.iterrows():
        src = row[attr_name + "0"]
        dest = row[attr_name + "3"]
        s = row['prediction']

        rank4dest = 1
        if src in src2false:
            rank4dest = 1 + len([i for i in src2false[src] if i > s])

        rank4src_inv = 1
        if dest in src2inv_false:
            rank4src_inv = 1 + len([i for i in src2inv_false[dest] if i > s])
        
        rank = min(rank4dest, rank4src_inv)
        
        mrr += 1.0 / rank
        hits10 = hits10 + 1 if rank <= 10 else hits10
        hits1 = hits1 + 1 if rank <= 1 else hits1
        hits3 = hits3 + 1 if rank <= 3 else hits3
        ranks += [rank]    
    
    numqueries = labels_df_test.shape[0]
    print("queries=" + str(numqueries) + " (" + str(scores.shape[0]) + \
          ") mrr=" + str(mrr) + \
          " hits10=" + str(hits10) + \
          " hits3=" + str(hits3) + \
          " hits1=" + str(hits1) +  \
          " (" + str(time.time()-begtime) + "s)")
