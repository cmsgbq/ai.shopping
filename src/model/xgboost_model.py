import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def get_data():
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
    train_end = 80000
    test_start = 30000
    test_end = 80000
    x_train = np.array(x_all[train_start:train_end]).astype(float)
    y_train = np.array(y_all[train_start:train_end]).astype(float)
    x_test = np.array(x_all[test_start:test_end]).astype(float)
    y_test = np.array(y_all[test_start:test_end]).astype(float)
    return (x_train, y_train, x_test, y_test )


def get_xgboost_model_score():
    x_train, y_train, x_test, y_test = get_data()
    #建立模型 计算模型得分
    xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.15, objective='reg:logistic')
    xgbr.fit(x_train, y_train)
    xgb_y = xgbr.predict(x_test)
    test_accuracy = roc_auc_score(y_test, xgb_y)
    print('accuracy score is %.4f'%(test_accuracy))
    return [xgbr, x_train, y_train, x_test, y_test]


def build_loadfile():
    #格式化并生成提交文件
    xgbr = get_xgboost_model_score()[0]
    #id
    submite_x_test_agg = pd.read_csv('../../data/corpus/test_agg.csv', header = None, index_col = None).as_matrix()
    submite_x_test_agg = [t[0].split('\t') for t in submite_x_test_agg]
    y_index = [x[-1] for x in submite_x_test_agg[1:]]
    #feature
    submite_x_test_feature = pd.read_csv('../../data/corpus/test_feature_62.csv', header = None, index_col = None).as_matrix()
    submite_x_test_feature = [t[0].split('\t') for t in submite_x_test_feature]
    submite_x_test_feature = [x[:] for x in submite_x_test_feature[1:]]


    submite_x = np.array(submite_x_test_feature).astype(float)
    submite_y = xgbr.predict(submite_x)

    tofile = [str(x)+'\t'+str(y) for x,y in zip(y_index, submite_y)]
    pd.DataFrame(tofile).to_csv('testY2.csv', header=['USRID\tRST'], index = False)


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
def change_param():
    x_train, y_train, x_test, y_test = get_data()
    model = XGBClassifier()
    
    n_estimators = [100, 300, 500, 800]
    learning_rate = [0.05,0.1, 0.17, 0.15,0.2,0.25,0.3,0.4,0.8,0.9,1.1]
    gamma = [0.2, 0.4, 0.5,0.6, 0.8]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8]
    
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    #build_loadfile()
    change_param()
    get_xgboost_model_score()
