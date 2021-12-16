import sys
sys.path.append('../src/meta_rule/')
sys.path.append('../dd_lnn/')

import time
import copy
import argparse
import dill
from read import load_metadata, load_data
import pandas as pd
from train_test import score

if __name__ == "__main__":
    begtime = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-bk", "--background", required=True, help="file containing schema information")
    parser.add_argument("-fte", "--facts_test", required=True, help="file containing facts for testing")
    parser.add_argument("-t", "--target", required=True, help="the target predicate")    
    parser.add_argument("-m", "--model", required=True, help="file to read the model from")
    
    cl_args = parser.parse_args()
    background_fname = cl_args.background
    facts_fname_test = cl_args.facts_test
    target = cl_args.target
    fin = cl_args.model
        
    (attr_name, labels_df_train, disj, rel_names_train) = dill.load(open(fin, 'rb'))
    
    dfs_test = load_metadata(background_fname)
    load_data(facts_fname_test, dfs_test)

    labels_df_test = dfs_test[target]
    labels_df_test.columns = [attr_name + "0", attr_name + "3"]

    true_links = pd.concat([labels_df_train, labels_df_test])
    true_links = true_links.loc[true_links['Relation'] == target]
    true_links.drop(['Relation'], axis=1, inplace=True)

    cols = true_links.columns.tolist()
    inv_true_links = true_links[[cols[1], cols[0]]].copy()
    inv_true_links.columns = [cols[0], cols[1]]

    mask = [1] * len(rel_names_train)
    pos = rel_names_train.index(target)
    mask[pos:pos+2] = [0] * 2
    yhat = score(disj, 1024, target, mask)
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
    print("done retrieving scores (" + str(time.time()-begtime) + "s)")
    
    false_links = pd.merge(yhat, true_links,
                           left_on=[attr_name + "0", attr_name + "3"],
                           right_on=[attr_name + "0", attr_name + "3"],
                           how='left', indicator=True) \
                    .query("_merge != 'both'") \
                    .drop('_merge', axis=1) \
                    .groupby([attr_name + "0"])['prediction'] \
                    .agg(list)
    print("done join for false links (" + str(time.time()-begtime) + "s)")
    src2false = {}
    for index, value in false_links.iteritems():
        src2false[index] = value
    print("done groupby for false links (" + str(time.time()-begtime) + "s)")
        
    inv_false_links = pd.merge(yhat, inv_true_links,
                               left_on=[attr_name + "0", attr_name + "3"],
                               right_on=[attr_name + "3", attr_name + "0"],
                               suffixes = ['_left', '_right'],
                               how='left', indicator=True) \
                        .query("_merge != 'both'") \
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
    print("queries=" + str(numqueries) + \
          " mrr=" + str(mrr) + \
          " hits10=" + str(hits10) + \
          " hits3=" + str(hits3) + \
          " hits1=" + str(hits1) +  \
          " (" + str(time.time()-begtime) + "s)")
