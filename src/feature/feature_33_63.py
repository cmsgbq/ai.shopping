# -*- coding: utf-8 -*-

import feature_io
import numpy as np
import pandas as pd



def feature_add(log_path, save_id_path, save_feature_path):
    
    f = feature_io.read_log(log_path)
    
    id_times = {}
    same_opration = set()
    id_same_opration_time = {}
    last_id = f[0][0]
    for line in f:
        if last_id != line[0]:
            same_opration.clear()
            last_id = line[0]
        if line[0] not in id_times:
            id_times[line[0]] = 1
        else:
            id_times[line[0]]+=1
            
        if line[1] not in same_opration:
            same_opration.add(line[1])
        else:
            id_same_opration_time[line[0]] = 1
           
    id_list = feature_io.read_id(save_id_path)
    feature1 = [[0,0]]
    id_list = [x[0] for x in id_list.tolist()]
    for id in id_list:
        l = []
        ll = []
        if id in id_times:
            ll.append(id_times[id])
        else :
            ll.append(0)
        if id in id_same_opration_time:
            ll.append(id_same_opration_time[id])
        else :
            ll.append(0)
        l.append(ll)
        feature1 = np.concatenate((np.array(feature1), np.array(l)), axis=0)
    feature_follow = feature1[1:]
    


    feature = np.array(feature_io.read_feature(save_feature_path))
    feature = np.concatenate((feature, feature_follow), axis=1)
    feature_io.write_feartue(feature, save_feature_path)
    
if __name__ == '__main__':
    feature_add('test_log.csv', 'test_id.csv','test_feature.csv')
