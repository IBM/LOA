import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchviz import make_dot
from third_party.dd_lnn.src.meta_rule.utils import evaluate, MyBatchSampler
from torch.utils.data import DataLoader, TensorDataset

def align_labels(meta, df, label):
    '''
    Joins df (containing ground truth labels) with meta.df to
    associate its tuples with a ground truth label. Assumption is that
    the column called label in df contains the labels. Labels are
    assumed to be binary class (0,1). If a tuple in meta.df is not
    found in df then its label is assumed to be 0 (member of negative
    class). Returns a vector of ground truth labels whose ith label
    corresponds to the ith tuple in meta.df.

    Parameters:
    ----------
    meta: the meta_intepretive object representing the model. Its df
    member variable should have already been populated.

    df: a dataframe which contains ground truth labels (binary-class, 0/1)
    in the label attribute.

    label: a string denoting the name of the column in df containing
    labels.

    Returns: a vector of labels
    '''
    col_names = meta.df.columns.values.tolist()
    meta.df['id'] = list(meta.df.index)
    y = torch.FloatTensor(df.merge(meta.df, on=col_names, how='right').drop_duplicates(ignore_index=True).sort_values('id')[[label]].values)
    meta.df.drop(['id'], axis=1, inplace=True)
    y = torch.where(y == y, y, torch.zeros(y.size()[0], 1))

    return y


def train(meta, df, label, step, batch, epochs, use_balanced):
    '''
    Training routine for meta-interpretive learning. Assumes meta's df
    and mat member variables have been populated, df's label column
    contains ground truth labels which are binary-class (0/1).

    Parameters:
    ----------

    meta: the meta-intepretive object representing the model

    df: dataframe whose label attribute contains the ground truth
    label (binary class, 0/1)

    label: the name of the attribute containing ground truth labels

    step: step size to use for learning

    batch: batch size to use for learning

    epochs: number of epochs to train for

    use_balanced: set to True to use balanced batch sampling, else set
    to False
    '''
    y = align_labels(meta, df, label)
    
    data = TensorDataset(torch.arange(y.size()[0]), y)
    loader = \
        DataLoader(data, sampler=MyBatchSampler(y), batch_size=batch, shuffle=False) if use_balanced \
        else DataLoader(data, batch_size=batch, shuffle=False)
    
    loss_fn = nn.BCEWithLogitsLoss() #nn.BCELoss()
    optimizer = optim.Adam(meta.parameters(), lr=step)

    #make_dot(meta(torch.arange(y.size()[0]))).render("attached", format="png")

    iter = 0
    for epoch in range(epochs):
        for idx, yb in loader:
            meta.train()
            optimizer.zero_grad()

            yhat, slacks = meta(idx)
            #print(yhat.t().data)
            
            loss = loss_fn(yhat, yb)
            
            #print(yb.t().data)
            print("Epoch " + str(epoch) + " (iteration=" + str(iter) + "): " + str(loss.item()))
            
            loss.backward()
            optimizer.step()
            iter = iter + 1
            
        #for name, param in meta.named_parameters():
            #print(name)
            #print(param.data)
            #print(param.grad)


def test(meta, df, label, batch):
    '''
    Testing routine for meta-interpretive learning. Assumes meta's df
    and mat member variables have been populated, df's label column
    contains ground truth labels which are binary-class (0/1).

    Parameters:
    ----------

    meta: the meta-intepretive object representing the model

    df: dataframe whose label attribute contains the ground truth
    label (binary class, 0/1) for comparison

    label: the name of the attribute containing ground truth labels

    Returns: Precision, Recall, and F1, in that order
    '''
    y = align_labels(meta, df, label)

    data = TensorDataset(torch.arange(y.size()[0]), y)
    loader = DataLoader(data, batch_size=batch, shuffle=False)

    cnt = 0
    yhat = []
    y = []
    with torch.no_grad():
        for idx, yb in loader:
            meta.eval()

            yhat_local, slacks = meta(idx)

            yhat += yhat_local[:,0].numpy().tolist()
            y += yb[:,0].numpy().tolist()

            cnt += yb.size()[0]
            print("Evaluated: " + cnt)

        yhat = np.asarray(yhat)
        y = np.asarray(y)
        
        max_f1 = -1.0
        best_thresh = 0.0
        step = (max(yhat) - min(yhat)) / 1000.0
        for thresh in np.arange(min(yhat), max(yhat), step):
            tp, fp, fn, precision, recall, f1 = \
                evaluate(np.greater_equal(np.asarray(yhat), thresh).astype(int), y)

            #print("Thresh=" + str(thresh) + " P/R/F1=" + str(precision) + "/" + str(recall) + "/" + str(f1))
            
            if(f1 >= max_f1):
                max_f1 = f1
                best_thresh = thresh

        #print("Best f1=" + str(max_f1))
        tp, fp, fn, precision, recall, f1 = \
            evaluate(np.greater_equal(np.asarray(yhat), best_thresh).astype(int), y)

        ret = meta.df.copy()
        ret[label] = y
        ret["prediction"] = yhat
        
        return precision, recall, f1, ret

def score(meta, batch, choice='default', mask=[]):
    data = TensorDataset(torch.reshape(torch.arange(meta.df.shape[0]), (-1, 1)))
    loader = DataLoader(data, batch_size=batch, shuffle=False)

    yhat = []
    pos = 0
    with torch.no_grad():
        meta.eval()
        for idx in loader:
            yhat_local, slacks = meta(idx[0], choice, mask)
            yhat += yhat_local[:,0].numpy().tolist()
            pos += idx[0].size()[0]
            if pos % 100000 == 0:
                print(str(pos) + "/" + str(meta.df.shape[0]))

        ret = meta.df.copy()
        ret["prediction"] = np.asarray(yhat)
        
        return ret
