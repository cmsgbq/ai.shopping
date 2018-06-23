# -*- coding: utf-8 -*-

import feature_io
import numpy as np
import pandas as pd
import feature_io

#添加用户日点击次数
def feature_33_62(read_log_path, read_id_path, read_feature_path, write_feartue_path):
    f = feature_io.read_log(read_log_path)
    id_list = feature_io.read_id(read_id_path)
    #key = id, Value = list[31]
    date_all_table = {}
    #保存存在操作日志的id
    #遍历操作日志 将操作时间按 天/每次+1 放入矩阵中
    for line in f:
        if int(line[0]) in date_all_table:
            date_all_table[int(line[0])][int( line[2].split()[0].split('-')[2])] += 1
        else:
            date_all_table[int(line[0])] = [0]*32
            date_all_table[int(line[0])][int( line[2].split()[0].split('-')[2])] += 1
    
    #读取feature 添加这[1:] 32列特征
    feature = feature_io.read_feature(read_feature_path)
    #遍历feature 无数据行补 [0]*32 构造新的list
    data_feature = [[0]*32]
    for i, line in enumerate(feature):
        if int(id_list[i]) in date_all_table:
            data_feature.append(date_all_table[int(id_list[i])][1:]) #从下标1开始 剪去无用数据（每月第0天）
        else:
            data_feature.append([0]*31)
    data_feature = data_feature[1:] #剪去第一空行
    
    #拼接并写入feature
    feature = np.concatenate((feature, np.array(data_feature)), axis=1)
    feature_io.write_feartue(feature, write_feartue_path)
    
if __name__ == '__main__':
    feature_33_62('train_log.csv', 'train_id.csv','train_feature.csv', 'train_feature_62.csv')

            
            