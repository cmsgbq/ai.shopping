# -*- coding: utf-8 -*-

import feature_io
import numpy as np
import pandas as pd
import feature_io




feature = feature_io.read_feature('../../data/for_add_feature/train_feature_64.csv')
add_f = pd.read_csv('../../data/for_add_feature/tea_feature.csv').as_matrix()[:,30:]
#拼接并写入feature
feature = np.concatenate((feature, np.array(add_f)), axis=1)
print('feature shape is ')
print(feature.shape)
feature_io.write_feartue(feature, 'final_feature0.csv')


feature = feature_io.read_feature('../../data/for_add_feature/test_feature_64.csv')
add_f = pd.read_csv('../../data/for_add_feature/tea_feature_test.csv').as_matrix()[:,30:]
#拼接并写入feature
feature = np.concatenate((feature, np.array(add_f)), axis=1)
print('test shape is ')
print(feature.shape)
feature_io.write_feartue(feature, 'final_feature0_test.csv')
