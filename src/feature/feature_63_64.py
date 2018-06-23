# -*- coding: utf-8 -*-

import feature_io
import numpy as np
import pandas as pd
import feature_io
import collections 

#添加同一用户点击 不同业务的业务种类数 特征
#添加同一用户点击 相同业务点击次数 最多的点击数 特征
def feature_63_64(read_log_path, read_id_path, read_feature_path, write_feartue_path):
    f = feature_io.read_log(read_log_path)
    id_list = feature_io.read_id(read_id_path)
    id_differernt_item_kind_dic = {}
    id_differernt_item_count_dic = {}
    id_same_item_count_dic = {}
    for line in f:
        if line[0] not in id_differernt_item_kind_dic:
            id_differernt_item_kind_dic[line[0]] = set()
            id_differernt_item_kind_dic[line[0]].add(line[1])
            id_differernt_item_count_dic[line[0]] = 1

            id_same_item_count_dic[line[0]] = {}
            id_same_item_count_dic[line[0]][line[1]] = 1
        else:
            if line[1] not in id_differernt_item_kind_dic[line[0]]:
                id_differernt_item_kind_dic[line[0]].add(line[1])
                id_differernt_item_count_dic[line[0]] += 1
                id_same_item_count_dic[line[0]][line[1]] = 1
            else:
                id_same_item_count_dic[line[0]][line[1]] += 1
    
    #提取最大点击次数
    id_same_item_MAX_count_dic = {}
    for (id_, dic_) in id_differernt_item_kind_dic.items():
        id_same_item_MAX_count_dic[id_] = max(list(id_same_item_count_dic[id_].value))



    #读取feature 添加这2列特征
    feature = feature_io.read_feature(read_feature_path)
    #遍历feature 无数据行补 [0] 构造新的list
    data_feature = [[0, 0], ]
    for i, line in enumerate(feature):
        l = []
        if line[0] in id_differernt_item_count_dic:
            l.append(id_differernt_item_count_dic[line[0]])
            l.append(id_same_item_MAX_count_dic[line[0]])
        else:
            l = [0, 0]

        data_feature.append(l)
    data_feature = data_feature[1:]

    #拼接并写入feature
    feature = np.concatenate((feature, np.array(data_feature)), axis=1)
    feature_io.write_feartue(feature, write_feartue_path)
    
if __name__ == '__main__':
    feature_63_64('train_log.csv', 'train_id.csv','train_feature.csv', 'train_feature_64.csv')
