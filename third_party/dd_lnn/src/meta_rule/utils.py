from torch.utils.data import Dataset, Sampler
import numpy as np

def evaluate(pred, target):
    '''
    Computes precision, recall, f1 given prediction and ground truth
    labels. Meant for binary-class inputs, labels are 0 and 1.

    Parameters:
    ----------
    pred: vector of predicted labels
    target: vector of ground truth labels

    Returns: True-positives, False-positives, False-negatives,
    Precision, Recall and F1, in that order.
    '''
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1-target))
    fn = np.sum((1-pred) * target)

    #0/0 for precision/recall is defined as 1
    precision = tp / (tp + fp) if tp+fp > 0 else 1
    recall = tp / (tp + fn) if tp+fn > 0 else 1
    f1 = 2 * tp / (2*tp + fp + fn)

    return tp, fp, fn, precision, recall, f1
    

class MyBatchSampler(Sampler):
    '''
    Balanced batch sampling. Assumes input consists of binary-class
    labels (0/1) and that the positive class (label=1) is the rarer
    class. Ensures that every batch consists of an equal number from
    the positive and negative class.    
    '''
    def __init__(self, labels):
        self.pos_idx = list(filter(lambda i: labels[i] == 1, range(len(labels))))
        self.neg_idx = list(filter(lambda i: labels[i] == 0, range(len(labels))))

        print("+ve: " + str(len(self.pos_idx)) + " -ve: " + str(len(self.neg_idx)))
        
        self.pos_idx = self.pos_idx * (len(self.neg_idx) // len(self.pos_idx)) #integer division
        fillin = len(self.neg_idx) - len(self.pos_idx)
        self.pos_idx = self.pos_idx if fillin == 0 else self.pos_idx + self.pos_idx[0:fillin] #pos_idx is now as long as neg_idx
        
        self.idx = [val for pair in zip(self.pos_idx, self.neg_idx) for val in pair] #interleaving pos_idx and neg_idx

        self.shuffle()

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

    #call this at the end of an epoch to shuffle
    def shuffle(self):
        np.random.shuffle(self.neg_idx)
        np.random.shuffle(self.pos_idx)
        self.idx = [val for pair in zip(self.pos_idx, self.neg_idx) for val in pair] #interleaving pos_idx and neg_idx


class MyData(Dataset):
    def __init__(self, data):
        target = 0
        
        self.X = data.drop(target, axis='columns').astype('float32')
        self.y = data[target].astype('float32').tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_idx = self.X.iloc[idx].to_numpy()
        y_idx = np.atleast_1d(self.y[idx])
        return [X_idx, y_idx]


