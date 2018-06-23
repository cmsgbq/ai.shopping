# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def split_file(path,id_path,feature_path):
    all = pd.read_csv('../../data/corpus/'+path, header = None, index_col = None).as_matrix()
    all = [t[0].split('\t') for t in all]
    f = [x[:-1] for x in all[1:]]
    f = np.array(f)
    f = ['\t'.join(x.astype(str).tolist()) for x in f]
    pd.DataFrame(f).to_csv('../../data/corpus/'+feature_path, index = False)
    
    k = [x[-1] for x in all[1:]]
    k = np.array(k[:]).astype(str)
    pd.DataFrame(k).to_csv('../../data/corpus/'+id_path, header=['id'], index = False)

def read_feature(path):
    f = pd.read_csv('../../data/corpus/'+path, header = None, index_col = None).as_matrix()
    print(type(f))
    f = [t[0].split('\t') for t in f]
    f = [x[:] for x in f[1:]]
    f = np.array(f[:]).astype(float)
    return f
    
def read_id(path):
    f = pd.read_csv('../../data/corpus/'+path, header = None, index_col = None).as_matrix()
    f = [x[:] for x in f[1:]]
    f = np.array(f[:]).astype(str)
    return f

def read_log(path):
    f = pd.read_csv('../../data/corpus/'+path, header = None, index_col = None).as_matrix()
    f = [t[0].split('\t') for t in f]
    f = [x[:] for x in f[1:]]
    f = np.array(f[:]).astype(str)
    return f

def write_feartue(f, path):
    tofile = ['\t'.join(x.astype(str).tolist()) for x in f]
    pd.DataFrame(tofile).to_csv('../../data/for_add_feature/'+path, header=['feature'], index = False)


if __name__ == '__main__':
    split_file('test_agg.csv','test_id1.csv','test_feature.csv')
    #ff =  read_feature()
    #write_feartue(ff)