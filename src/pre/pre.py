import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score



test_agg = pd.read_csv('../../data/corpus/train_agg.csv', header = None, index_col = None).as_matrix()
#test_log = pd.read_csv('../../data/corpus/train_log.csv', header = None, index_col = None).as_matrix()
Y_t = pd.read_csv('../../data/corpus/train_flg.csv', header = None, index_col = None).as_matrix()

test_feartue = pd.read_csv('../../data/corpus/train_feature_62.csv', header = None, index_col = None).as_matrix()
test_feartue = [t[0].split('\t') for t in test_feartue]

test_agg = [t[0].split('\t') for t in test_agg]
#test_log = [t[0].split('\t') for t in test_log]
user_set = set()
user_flg_dic = {}
for id in Y_t:
    split = id[0].split('\t')
    user_set.add(split[0])
    user_flg_dic[split[0]] = split[1]

x_all = [x[:] for x in test_feartue[1:]]
y_all = [user_flg_dic[y] for y in [x[-1] for x in test_agg[1:]] ]

'''
id_agg = set([x[-1] for x in test_agg[1:]])
id_log = set([x[0] for x in test_log[1:]])
id_agg_sub_log = id_agg-id_log

content_id_had_log = [x for x in test_log[1:] if str(x[0]) not in id_agg_sub_log] 

param = {'max_depth':'5', 'eta':0.1, 'silent':0, 'objective':'multi:softmax','num_class':3 }
num_round = 2 
'''
train_start = 0 
train_end = 70000
test_start = 70000
test_end = 80000

x_train = np.array(x_all[train_start:train_end]).astype(float)
y_train = np.array(y_all[train_start:train_end]).astype(float)
x_test = np.array(x_all[test_start:test_end]).astype(float)
y_test = np.array(y_all[test_start:test_end]).astype(float)


data_train=xgb.DMatrix(x_train,label=y_train)  
data_test=xgb.DMatrix(x_test,label=y_test)  
watch_list=[(data_test,'eval'),(data_train,'train')]  

#bst = xgb.train(param, data_train, num_boost_round=num_round, evals=watch_list)
#xgb_y=bst.predict(data_test)  
xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.1,silent=False, objective='reg:logistic')
xgbr.fit(x_train, y_train)
xgb_y = xgbr.predict(x_test)


'''

print('start lr')
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_y = lr.predict(x_test)

plt.plot(y_train[0:50],label='True')
plt.plot(lr_y[0:50],label='LR')
# plt.plot(gnb_y[200:250],label='BYS')
# plt.plot(bnb_y[200:250],label='BYS')
plt.legend()
'''
'''
hit = 0
for a, b in zip(y_test, xgb_y):
    if a == int(round(b)):
        hit+=1
print("xgb accr is %.2f" %(hit/len(y_test)))
'''
'''
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
test_accuracy = roc_auc_score(y_test, xgb_y)
print('accuracy score is %.4f'%(test_accuracy))
