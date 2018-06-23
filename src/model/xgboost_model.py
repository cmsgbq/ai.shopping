import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


#读取train_agg的id分布，通过list[id] 在train_feature与train_flg中，构造训练集和测试集
Y_train_data = pd.read_csv('../../data/corpus/train_flg.csv', header = None, index_col = None).as_matrix()

train_agg = pd.read_csv('../../data/corpus/train_agg.csv', header = None, index_col = None).as_matrix()
train_agg = [t[0].split('\t') for t in train_agg]

train_feartue = pd.read_csv('../../data/corpus/train_feature_62.csv', header = None, index_col = None).as_matrix()
train_feartue = [t[0].split('\t') for t in train_feartue]

user_set = set()
user_flg_dic = {}
for id in Y_train_data:
    split = id[0].split('\t')
    user_set.add(split[0])
    user_flg_dic[split[0]] = split[1]

#得到全部训练数据
x_all = [x[:] for x in train_feartue[1:]]
y_all = [user_flg_dic[y] for y in [x[-1] for x in train_agg[1:]] ] #根据 训练集X的id，取出对应Y
#切割训练数据
train_start = 0 
train_end = 70000
test_start = 70000
test_end = 80000
x_train = np.array(x_all[train_start:train_end]).astype(float)
y_train = np.array(y_all[train_start:train_end]).astype(float)
x_test = np.array(x_all[test_start:test_end]).astype(float)
y_test = np.array(y_all[test_start:test_end]).astype(float)

#建立模型 计算模型得分
data_train=xgb.DMatrix(x_train,label=y_train)  
data_test=xgb.DMatrix(x_test,label=y_test)  
watch_list=[(data_test,'eval'),(data_train,'train')]  

xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.1,silent=False, objective='reg:logistic')
xgbr.fit(x_train, y_train)
xgb_y = xgbr.predict(x_test)
test_accuracy = roc_auc_score(y_test, xgb_y)
print('accuracy score is %.4f'%(test_accuracy))


'''
#格式化并生成提交文件
#id
submite_x_test_agg = pd.read_csv('../../data/corpus/test_agg.csv', header = None, index_col = None).as_matrix()
submite_x_test_agg = [t[0].split('\t') for t in submite_x_test_agg]
y_index = [x[-1] for x in submite_x_test_agg[1:]]
#feature
submite_x_test_feature = pd.read_csv('../../data/corpus/test_fill_opr_times_feature.csv', header = None, index_col = None).as_matrix()
submite_x_test_feature = [t[0].split('\t') for t in submite_x_test_feature]
submite_x_test_feature = [x[:] for x in submite_x_test_feature[1:]]


submite_x = np.array(submite_x_test_feature).astype(float)
submite_y = xgbr.predict(submite_x)

tofile = [str(x)+'\t'+str(y) for x,y in zip(y_index, submite_y)]
pd.DataFrame(tofile).to_csv('testY2.csv', header=['USRID\tRST'], index = False)
'''

