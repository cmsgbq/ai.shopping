import xgboost as xgb
import numpy as np
import pandas as pd
import feature_io
from xgboost import XGBClassifier

def count_none0(s):
    count_none0 = 0
    for i in s:
        if i != 0:
            count_none0 += 1
    return count_none0

def count_0(s):
    count_0 = 0
    for i in s:
        if i == 0:
            count_0 += 1
    return count_0

def log_OCC_TIM(data,recenttime):
    data['day'] = data.OCC_TIM.map(lambda x:x.day)
    data['hour'] = data.OCC_TIM.map(lambda x:x.hour)

    recent_time = recenttime
    time = data.groupby(['USRID'],as_index=False)['OCC_TIM'].agg({'recenttime':max})
    time['time_gap'] = (recent_time-time['recenttime']).dt.total_seconds()
    
    df_log = train_log.sort_values(['USRID','OCC_TIM'])
    df_log['next_time'] = data.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
    df_log['next_time'] = df_log.next_time.dt.total_seconds()
    log = df_log.groupby(['USRID'],as_index=False)['next_time'].agg({
        'next_time_mean':np.mean,
        'next_time_std':np.std,
        'next_time_min':np.min,
        'next_time_max':np.max
    })
    
    log_temp = log.drop(['USRID'],axis=1)
    log['next_time_cuont0'] = log_temp.apply(count_0,axis=1)
    log['next_time_cuontnone0'] = log_temp.apply(count_none0,axis=1)
    time = pd.merge(time,log,on='USRID',how='left')
    '''
    data['dayofweek'] = data.OCC_TIM.dt.dayofweek
    df_dw = data.groupby(['USRID','dayofweek'])['USRID'].count().unstack()
    df_dw['dw_count0'] = df_dw.apply(count_0,axis=1)
    df_dw['dw_countnone0'] = df_dw.apply(count_none0,axis=1)
    df_dw.reset_index(inplace=True)
    time = pd.merge(time,df_dw,on='USRID',how='left')
    df_day = data.groupby(['USRID','day'])['USRID'].count().unstack()
    df_day['day_count0'] = df_day.apply(count_0,axis=1)
    df_day['day_countnone0'] = df_day.apply(count_none0,axis=1)
    df_day.reset_index(inplace=True)
    time = pd.merge(time,df_day,on='USRID',how='left')
    df_hour = data.groupby(['USRID','hour'])['USRID'].count().unstack()
    df_hour['hour_count0'] = df_hour.apply(count_0,axis=1)
    df_hour['hour_countnone0'] = df_hour.apply(count_none0,axis=1)
    df_hour.reset_index(inplace=True)
    
    time = pd.merge(time,df_hour,on='USRID',how='left')
    '''
    return time


def feture_imp(data_log,data_flag,n):
    df = data_log.groupby(['USRID','EVT_LBL'])['USRID'].count().unstack()
    df_c = pd.merge(df,data_flag,left_index=True,right_on='USRID',how='right')
    df_c.fillna(0,inplace=True)
    x = df_c.drop(['USRID','FLAG'],axis=1)
    y = df_c['FLAG']
    clf = XGBClassifier(n_estimators=30,max_depth=5)
    clf.fit(x,y)
    imp = clf.feature_importances_
    names = x.columns
    d={}
    for i in range(len(names)):
        d[names[i]] = imp[i]
    d = sorted(d.items(),key=lambda x:x[1],reverse=True)
    d = d[0:n]
    feture_list=[j[0] for j in d]
    return feture_list


def log_EVTLBL_STA(data,feature_list):
    df = data.groupby(['USRID','EVT_LBL'])['USRID'].count().unstack()
    df_new = pd.DataFrame()
    for i in feature_list:
        try:
             df_new[i] = df[i]
        except:
            df_new[i] = 0
    df_new.index = df.index
    df_new['df_new_count0'] = df_new.apply(count_0,axis=1)
    df_new['df_new_countnone0'] = df_new.apply(count_none0,axis=1)       
    return df_new



train_feature = feature_io.read_feature('../../data/for_add_feature/train_f_30_4.csv')

train_log = pd.read_csv('../../data/corpus/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'],engine='python')
train_flg = pd.read_csv('../../data/corpus/train_flg.csv',sep='\t',engine='python')
recenttime =  max(train_log.OCC_TIM)


time = log_OCC_TIM(train_log,recenttime)
print('feature shape is')
print(train_feature.shape)

feature_list = feture_imp(train_log,train_flg,25)
df_evtblb_sta = log_EVTLBL_STA(train_log,feature_list)

time_id = pd.merge(train_flg,time,on=['USRID'],how='left')

time_id = pd.merge(time_id,df_evtblb_sta,left_on='USRID',right_index=True,how='left')

time_id.time_gap.fillna(max(time_id.time_gap)+1,inplace=True)
time_id.fillna(0,inplace=True)

time_id = time_id.drop(['USRID', 'FLAG','recenttime'], axis=1).values

print('time shape is')
print(time_id.shape)
train_feature = np.concatenate((train_feature, np.array(time_id)), axis=1)


feature_io.write_feartue(train_feature, 'train_34_time.csv')



#################################

train_feature = feature_io.read_feature('../../data/for_add_feature/test_f_30_4.csv')

train_log = pd.read_csv('../../data/corpus/test_log.csv',sep='\t',parse_dates = ['OCC_TIM'],engine='python')
train_flg = pd.read_csv('../../data/corpus/test_id.csv',sep='\t',engine='python')
recenttime =  max(train_log.OCC_TIM)


time = log_OCC_TIM(train_log,recenttime)
print('feature shape is')
print(train_feature.shape)

df_evtblb_sta = log_EVTLBL_STA(train_log,feature_list)

time_id = pd.merge(train_flg,time,on=['USRID'],how='left')

time_id = pd.merge(time_id,df_evtblb_sta,left_on='USRID',right_index=True,how='left')

time_id.time_gap.fillna(max(time_id.time_gap)+1,inplace=True)
time_id.fillna(0,inplace=True)

time_id = time_id.drop(['USRID','recenttime'], axis=1).values

print('time shape is')
print(time_id.shape)
train_feature = np.concatenate((train_feature, np.array(time_id)), axis=1)


feature_io.write_feartue(train_feature, 'test_34_time.csv')