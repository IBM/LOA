import sys
sys.path.append('../src/meta_rule/')
sys.path.append('../dd_lnn/')

import time
import argparse
from read import load_metadata, load_data, load_labels_kbc
from meta_interpretive import BaseMetaPredicate, MetaRule, Project, DisjunctionRule
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import dill

if __name__ == "__main__":
    begtime = time.time()
    alpha = 0.95
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-bk", "--background", required=True, help="file containing schema information")
    parser.add_argument("-ftr", "--facts_train", required=True, help="file containing facts for training")
    parser.add_argument("-m", "--model", required=True, help="file to write the model in")

    cl_args = parser.parse_args()
    background_fname = cl_args.background
    facts_fname_train = cl_args.facts_train
    fout = cl_args.model

    dfs_train = load_metadata(background_fname)
    load_data(facts_fname_train, dfs_train)

    attr_name = None
    relations_train = []
    rel_names_train = []
    choices = []
    for name, df in dfs_train.items():
        colnames = df.columns.values.tolist()
        attr_name = colnames[0]
        df.columns = [attr_name + "0", attr_name + "1"]

        choices.append(name)
        rel_names_train.append(name)
        relations_train.append(df)

        #creating inv
        name = "inv" + name
        cols = df.columns.tolist()
        inv_df = df[[cols[1], cols[0]]].copy()
        inv_df.columns = [cols[0], cols[1]]
        rel_names_train.append(name)
        relations_train.append(inv_df)

    colnames = [attr_name + "0", attr_name + "3"]
    labels_df_train = load_labels_kbc(facts_fname_train, colnames)
    print("done reading (" + str(time.time()-begtime) + ")")

    body0 = BaseMetaPredicate(relations_train, choices) #id0, id1
    print("done body0 (" + str(time.time()-begtime) + "s)")
    body1 = copy.deepcopy(body0)
    body1.df.columns = [attr_name + "2", attr_name + "3"] #id2, id3
    join = MetaRule([body0, body1], [[[attr_name + "1"], [attr_name + "2"]]], alpha, False, choices)
    print("done join (" + str(time.time()-begtime) + "s)")
    proj = Project(join, [attr_name + "0", attr_name + "3"])
    print("done project (" + str(time.time()-begtime) + "s)")
    metap = copy.deepcopy(body0)
    metap.df.columns = [attr_name + "0", attr_name + "3"]
    disj = DisjunctionRule([metap, proj], alpha, 0, choices)
    print("done disjunction (" + str(time.time()-begtime) + "s)")

    col_names = disj.df.columns.values.tolist()
    disj.df['id'] = list(disj.df.index)
    with_relations = labels_df_train.merge(disj.df, on=col_names, how='right')                               
    disj.df.drop(['id'], axis=1, inplace=True)

    with_relations = with_relations[with_relations['Relation'].apply(type) == str]
    print("+ve: " + str(len(with_relations.index)))
    grouped = with_relations.groupby(['Relation'])['id'].agg(list).reset_index(name='idlist')
    rel2id = {row['Relation']: row['idlist'] for index, row in grouped.iterrows()}
    print("done label alignment (" + str(time.time()-begtime) + "s)")
    
    loss_fn = nn.LogSigmoid()
    optimizer = optim.Adam(disj.parameters(), lr=1e-3)
    batch_size = 32
    iter = 0
    for epoch in range(30):
        pos_loss = 0.0
        for rel in rel2id:
            ids = rel2id[rel]
            mask = [1] * len(rel_names_train)
            pos = rel_names_train.index(rel)
            mask[pos:pos+2] = [0] * 2
            for i in range(0, len(ids), batch_size):
                batch = ids[i : i + batch_size]
                
                disj.train()
                optimizer.zero_grad()

                yhat, slacks = disj(batch, rel, mask)
                #loss = - torch.sum(yhat)
                loss = - torch.sum(loss_fn(yhat))
                pos_loss = pos_loss + loss.item()
                #print("Epoch " + str(epoch) + " (iteration=" + str(iter) + "): " + str(loss.item()))
                
                loss.backward()
                optimizer.step()
                iter = iter + 1

        print("Epoch=" + str(epoch) + " +ve.loss=" + str(pos_loss) + " (" + str(time.time()-begtime) + "s)")

    model = (attr_name, labels_df_train, disj, rel_names_train)
    dill.dump(model, open(fout, 'wb'))
