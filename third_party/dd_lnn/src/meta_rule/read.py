import pandas as pd
#from multiprocessing import Pool
#from functools import partial

#parses a line of ILP input
def parse(line):
    remainder = line
    pos = remainder.find('(')
    rel_name = remainder[0:pos]
    remainder = remainder[pos+1:]

    args = []
    pos = remainder.find(',')
    while pos != -1:
        arg_name = remainder[0:pos].strip()
        remainder = remainder[pos+1:]
        pos = remainder.find(',')
        args.append(arg_name)
        
    pos = remainder.find(')')
    arg_name = remainder[0:pos].strip()
    remainder = remainder[pos+1:]
    args.append(arg_name)

    return rel_name, args

#parses the background file
def load_metadata(fname):
    '''
    Creates a dictionary with all predicates and their schemas as
    described in the background file fname

    Parameter fname: name of file
    '''
    file = open(fname, 'r')

    ret = {}
    for line in file:
        line = line.strip()
        if not line:
            next
        else:
            rel_name, colnames = parse(line)
            df = pd.DataFrame(columns=colnames)
            ret[rel_name] = df

    return ret
        
#populates labels relation
def load_labels(fname, dfs, label):
    '''
    Appends to the predicate in the dictionary (dfs) with labels found
    in file fname

    Parameters:
    ----------
    fname: name of file with labels

    dfs: dictionary containing schemas for all predicates in the
    knowledge base

    label: the label to assign all tuples in fname. we assume
    binary-class input, so label is either 0 or 1
    '''
    file = open(fname, 'r')

    rel_name = None
    rows = []
    for line in file:
        line = line.strip()
        if not line:
            next
        else:            
            rel_name, vals = parse(line)
            colnames = dfs[rel_name].columns.values.tolist()
            if "Label" in colnames:
                colnames = colnames[0:len(colnames)-1]
            rows.append(pd.Series(vals, index=colnames))
            #print("added to " + rel_name + " " + str(len(rows[rel_name])) + "th row")

    df = dfs[rel_name]
    colnames = df.columns.values.tolist()
    tmp_df = pd.DataFrame(rows, columns=colnames)
    tmp_df["Label"] = [label] * len(tmp_df.index)
    dfs[rel_name] = tmp_df if "Label" not in colnames else df.append(tmp_df)

    return df

#returns 1 df with 'Relation' column
def load_labels_kbc(fname, colnames):
    file = open(fname, 'r')

    colnames.append("Relation")
    rows = []
    for line in file:
        line = line.strip()
        if not line:
            next
        else:            
            rel_name, vals = parse(line)
            vals.append(rel_name)
            rows.append(pd.Series(vals, index=colnames))
            #print("added to " + rel_name + " " + str(len(rows[rel_name])) + "th row")

    df = pd.DataFrame(rows, columns=colnames)
    return df

def add_facts(fname, dfs):
    file = open(fname, 'r')

    rel_name = None
    rows = []
    for line in file:
        line = line.strip()
        if not line:
            next
        else:            
            rel_name, vals = parse(line)
            colnames = dfs[rel_name].columns.values.tolist()
            rows.append(pd.Series(vals, index=colnames))

    df = dfs[rel_name]
    colnames = df.columns.values.tolist()
    tmp_df = pd.DataFrame(rows, columns=colnames)
    dfs[rel_name] = df.append(tmp_df)

    return df

#needed for multi-threaded reading of facts file
#def process_line(dfs, indices, line):
#    rel_name, vals = parse(line)
#    return rel_name, pd.Series(vals, index=indices[rel_name])

#reads facts file. turn on Pool to read with multiple threads. this
#function has been carefully profiled, do not change unless you know
#what's going on
def load_data(fname, dfs):
    '''
    Replaces the predicates in the dictionary (dfs) with predicates
    populated after reading the file fname

    Parameters:
    ----------
    fname: name of file

    dfs: dictionary containing schemas for all predicates in the
    knowledge base
    '''
    rows = {}
    indices = {}
    for rel_name in dfs:
        rows[rel_name] = []
        indices[rel_name] = pd.Index(dfs[rel_name].columns.values.tolist())

    #begin multi-threaded tuple creation
    #pool_read = Pool()
    #func = partial(process_line, dfs, indices)
    #with open(fname) as f:
    #    results = pool_read.map(func, f)
    #pool_read.close()
    #pool_read.join()
    #for rel_name, series in results:
    #    rows[rel_name].append(series)
    #end multi-threaded tuple creation

    #begin single-threaded tuple creation
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                next
            else:
                rel_name, vals = parse(line)
                if rel_name in rows:
                    rows[rel_name].append(pd.Series(vals, index=indices[rel_name]))
    #end single-threaded tuple creation

    #begin single-threaded df creation
    for rel_name, tuples in rows.items():
        dfs[rel_name] = pd.concat(tuples, axis=1, copy=False).T if len(tuples) > 0 else dfs[rel_name]
    #end single-threaded df creation
    
