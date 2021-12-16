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

if __name__ == "__main__":
    begtime = time.time()
    alpha = 0.99
    
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
            
    rel_names_train.append(target)
    relations_train.append(labels_df_train.copy())

    labels_df_train.columns = [attr_name + "0", attr_name + "3"]
    labels_df_train['Label'] = 1.0
            
    body0 = BaseMetaPredicate(relations_train)
    print("done body0 (" + str(time.time()-begtime) + "s)")
    body2 = copy.deepcopy(body0)
    body2.df.columns = [attr_name + "11", attr_name + "22"]
    body1 = copy.deepcopy(body0)
    body1.df.columns = [attr_name + "2", attr_name + "3"]
    join = MetaRule([body0, body2, body1], [[[attr_name + "1"], [attr_name + "11"]], [[attr_name + "22"], [attr_name + "2"]]], alpha, False)
    print("done join (" + str(time.time()-begtime) + "s)")
    proj = Project(join, [attr_name + "0", attr_name + "3"])
    print("done project (" + str(time.time()-begtime) + "s)")
    
    meta = proj
    label = 'Label'
    step = 1e-1
    batch_size = 32
    epochs = 1000
    y = align_labels(meta, labels_df_train, label)    
    print("done label alignment (" + str(time.time()-begtime) + "s)")

    pos_idx = np.nonzero(y.numpy())[0].tolist()
    pos_loader = DataLoader(TensorDataset(torch.LongTensor(pos_idx)), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(meta.parameters(), lr=step)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    #basemetapredicate alpha: is predicates1, project: sum
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
        
    print("done training")

    print(rel_names_train)
    
    np.set_printoptions(precision=3, suppress=True)
    #params = body0.alpha['default'].get_params().detach().numpy()
    #print("1st predicate: " + str(params))
    beta, weights = body0.alpha['default'].get_params()
    print("1st predicate: " + str(beta.item()) + " " + str(weights.detach().numpy()))
    np.set_printoptions(precision=3, suppress=True)
    #params = body2.alpha['default'].get_params().detach().numpy()
    #print("2nd predicate: " + str(params))
    beta, weights = body2.alpha['default'].get_params()
    print("2nd predicate: " + str(beta.item()) + " " + str(weights.detach().numpy()))
    np.set_printoptions(precision=3, suppress=True)
    #params = body1.alpha['default'].get_params().detach().numpy()
    #print("3nd predicate: " + str(params))
    beta, weights = body1.alpha['default'].get_params()
    print("3nd predicate: " + str(beta.item()) + " " + str(weights.detach().numpy()))

    lnn_beta, lnn_wts, slacks = join.AND['default'].cdd()
    np.set_printoptions(precision=3, suppress=True)
    print("LNN beta, weights: " + \
          str(np.around(lnn_beta.item(), decimals=3)) + " " + str(lnn_wts.detach().numpy()))

    dfs_test = load_metadata(background_fname)
    load_data(facts_fname_test, dfs_test)

    labels_df_test = dfs_test[target]
    labels_df_test.columns = [attr_name + "0", attr_name + "3"]
    print("read test data")

    yhat = score(proj, batch_size)
    print("done evaluation (" + str(time.time()-begtime) + "s)")

    test_countries = list(set(labels_df_test[[attr_name + "0"]].to_numpy().transpose()[0].tolist()))
    test_regions = list(set(labels_df_test[[attr_name + "3"]].to_numpy().transpose()[0].tolist()))

    fout = open('auc.csv', 'w')
    for test_c in test_countries:
        for test_r in test_regions:
            check = labels_df_test.loc[(labels_df_test[attr_name + "0"] == test_c) & (labels_df_test[attr_name + "3"] == test_r)]
            ground_prob = None
            if check.shape[0] == 0:
                ground_prob = 0.0
            else:
                ground_prob = 1.0

            check = yhat.loc[(yhat[attr_name + "0"] == test_c) & (yhat[attr_name + "3"] == test_r)]
            pred_prob = None
            if check.shape[0] == 0:
                pred_prob = 0.0
            else:
                for idx, r in check.iterrows():
                    pred_prob = r['prediction']

            fout.write(str(test_c) + " " + str(test_r) + " " + str(pred_prob) + " " + str(ground_prob) + "\n")
    fout.close()
                    
#neighborOf, locatedIn
#1st predicate: 0.0 [[1.034 0.042]]
#2nd predicate: 0.0 [[1.033 0.042]]
#3nd predicate: 0.0 [[0.001 1.075]]
#LNN beta, weights: 1.119 [1.125 1.125 1.125]
