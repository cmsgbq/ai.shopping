# -*- coding: utf-8 -*-

import feature_io
import numpy as np
import pandas as pd
import feature_io




feature = feature_io.read_feature('train_feature_64.csv')
add_f = pd.read_csv('../../data/for_add_feature/else_add_feature.csv')
#拼接并写入feature
feature = np.concatenate((feature, np.array(data_feature)), axis=1)
feature_io.write_feartue(feature, 'final_feature.csv')