import sys
sys.path.append('../../src/meta_rule/')
sys.path.append('../../dd_lnn/')

import argparse
import pandas as pd
import numpy as np
from meta_interpretive import BasePredicate, MetaRule, Project, DisjunctionRule, Negation
from train_test import train, test
from read import load_data, load_metadata, load_labels

def main():
    alpha = 0.95
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-bktr", "--background_train", required=True, help="file containing background for training")
    parser.add_argument("-ftr", "--facts_train", required=True, help="file containing facts for training")
    parser.add_argument("-bkte", "--background_test", required=True, help="file containing background for testing")
    parser.add_argument("-fte", "--facts_test", required=True, help="file containing facts for testing")

    #Rule 1: Smokes(x) => Cancer(x)
    #Rule 2: Friends(x, y) => (Smokes(x) <=> Smokes(y))

    cl_args = parser.parse_args()
    background_fname_train = cl_args.background_train
    facts_fname_train = cl_args.facts_train
    background_fname_test = cl_args.background_test
    facts_fname_test = cl_args.facts_test

    dfs_train = load_metadata(background_fname_train)
    load_data(facts_fname_train, dfs_train)

    cancer = dfs_train["Cancer"] #this is the target
    person = cancer.columns.values.tolist()[0]
    #we need a column of labels
    label = "cancer"
    cancer[label] = [1.0]*len(cancer.index)
    
    smoker = BasePredicate(dfs_train["Smokes"])

    #pandas doesn't deal well with repeated col names, need to rename
    person1 = person + "1"
    friends = dfs_train["Friends"]
    friends.columns = [person, person1]
    friends = BasePredicate(friends)
    
    #create negation of smokes
    non_smoker = BasePredicate(pd.DataFrame(set(dfs_train["Friends"][person].values) \
                                                - set(dfs_train["Smokes"][person].values), \
                                                columns=[person]))
    
    mr1 = MetaRule([friends, smoker], [[[person], [person]]], alpha, False)
    proj1 = Project(mr1, [person1])
    proj1.rename(proj1.df, person1, person)
    
    mr2 = MetaRule([friends, non_smoker], [[[person], [person]]], alpha, False)
    proj2 = Project(mr2, [person1])
    nproj = Negation(proj2)
    nproj.rename(nproj.df, person1, person)

    disj = DisjunctionRule([proj1, nproj, smoker], alpha, 0.0)
    
    train(disj, cancer, label, 1e-10, 2, 10, True) #len(disj.df.index)

    or1 = disj.OR
    beta_or, argument_wts_or, slacks = or1.AND.cdd()
    and1 = mr1.AND
    beta_and1, argument_wts_and1, slacks = and1.cdd()
    and2 = mr2.AND
    beta_and2, argument_wts_and2, slacks = and2.cdd()

    np.set_printoptions(precision=3, suppress=True)
    print("OR (beta, argument weights): " \
          + str(np.around(beta_or.item(), decimals=3)) + " " \
          + str(argument_wts_or.detach().numpy()))
    print("AND1 (beta, argument weights): " \
          + str(np.around(beta_and1.item(), decimals=3)) + " " \
          + str(argument_wts_and1.detach().numpy()))
    print("AND2 (beta, argument weights): " \
          + str(np.around(beta_and2.item(), decimals=3)) + " " \
          + str(argument_wts_and2.detach().numpy()))

    dfs_test = load_metadata(background_fname_test)
    load_data(facts_fname_test, dfs_test)
    
    cancer = dfs_test["Cancer"] #this is the target
    cancer[label] = [1.0]*len(cancer.index)
    
    smoker = BasePredicate(dfs_test["Smokes"])

    friends = dfs_test["Friends"]
    friends.columns = [person, person1]
    friends = BasePredicate(friends)
    
    #create negation of smokes
    non_smoker = BasePredicate(pd.DataFrame(set(dfs_test["Friends"][person].values) \
                                                - set(dfs_test["Smokes"][person].values), \
                                                columns=[person]))
    
    mr1 = MetaRule([friends, smoker], [[[person], [person]]], alpha, False)
    mr1.AND = and1
    proj1 = Project(mr1, [person1])
    proj1.rename(proj1.df, person1, person)
    
    mr2 = MetaRule([friends, non_smoker], [[[person], [person]]], alpha, False)
    mr2.AND = and2
    proj2 = Project(mr2, [person1])
    nproj = Negation(proj2)
    nproj.rename(nproj.df, person1, person)

    disj = DisjunctionRule([proj1, nproj, smoker], alpha, 0.0)
    disj.OR = or1

    precision, recall, f1, result = test(disj, cancer, label)
    print(result[[person, "prediction"]].to_markdown())

        
if __name__ == "__main__":
    main()
