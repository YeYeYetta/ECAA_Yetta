# -*- coding: utf-8 -*-
import gc
import joblib
import warnings
import numpy as np
import pandas as pd
import catboost as cb
from tqdm import tqdm
from joblib import Parallel,delayed
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

# load data
tra = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# base fe
def fe_base(df):
    df['week'] = df['date']%7
    for c in tqdm(['comments','zhi','buzhi','favorite','orders']):
        df[c+'_21_diff'] = df[c+'_2h']-df[c+'_1h']
        df[c+'_21_rate'] = df[c+'_2h']/(df[c+'_1h']+0.001)
        df[c+'_21_sum'] = df[c+'_2h']+df[c+'_1h']
    df['author_l1'] = df['author'].astype(str)+'_'+df['level1'].astype(str)
    df['author_l1-2'] = df['author_l1']+'_'+df['level2'].astype(str)
    df['author_l1-3'] = df['author_l1-2']+'_'+df['level3'].astype(str)
    df['author_l1-4'] = df['author_l1-3']+'_'+df['level4'].astype(str)
    df['author_brand'] = df['author'].astype(str)+'_'+df['brand'].astype(str)
    df['author_mall'] = df['author'].astype(str)+'_'+df['mall'].astype(str)
    df['author_brand_mall'] = df['author_brand']+'_'+df['mall'].astype(str)
    
fe_base(tra)
fe_base(test)

# train val split
train = tra[tra.date<=109]
val = tra[tra.date>109]
train['set']='train';val['set']='val';test['set']='test'
all_df = pd.concat([train,val,test],ignore_index=True)
all_df['level_1'] = all_df.index

# get stat var
def fe_stat(df):
    for c1 in tqdm(['author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall','author_brand','author_brand_mall', 'author_l1', 'author_l1-2','author_l1-3', 'author_l1-4', 'author_mall','url', 'baike_id_1h','baike_id_2h','date','week']):
        tr0 = df.groupby(c1)[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate']].agg(['count','nunique','sum','mean','median','max','min','std'])
        tr0.columns = [f'{x}_gp_{c1}_{y}_0810' for x,y in tr0.columns]
        df = pd.merge(df,tr0,left_on=c1,right_index=True,how='left')
        if c1 not in ['date','week']:
            tr0 = df.groupby([c1,'date'])[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate']].agg(['count','nunique','sum','mean','median','max','min','std'])
            tr0.columns = [f'{x}_gp_{c1}date_{y}_0810' for x,y in tr0.columns]
            df = pd.merge(df,tr0,left_on=[c1,'date'],right_index=True,how='left')
    return df

all_df=fe_stat(all_df)

# get sft var
def get_sft(all_df,c,i,flag=''):
    tmp = all_df[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate']].groupby(all_df[c]).shift(i)
    tmp.columns=[x+'gp_{}_shift_{}_{}'.format(c,i,flag) for x in tmp.columns]
    tmp1 = all_df[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate']].groupby(all_df[c]).shift(-i)
    tmp1.columns=[x+'gp_{}_shift_{}_n1_{}'.format(c,i,flag) for x in tmp.columns]
    tmpf = pd.merge(tmp,tmp1,left_index=True,right_index=True)   
    del tmp,tmp1;gc.collect()
    return tmpf

rng = {}
rng_list=[]
for c in ['baike_id_2h','url','author','mall','author_brand','author_brand_mall']:
    rng[c]={}
    for i in range(1,5):
        rng[c][i]=all_df[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate',c]].copy()
        rng_list.append([c,i])

p_res = Parallel(n_jobs=-1)(delayed(get_sft)(rng[c][i],c,i) for c,i in tqdm(rng_list))
all_df = pd.concat([all_df]+p_res,axis=1)
all_df.sort_values(['article_id','date'],inplace=True)
all_df.index=range(len(all_df))
rng = {}
rng_list=[]
for c in ['baike_id_2h','url','author','mall','author_brand','author_brand_mall']:
    rng[c]={}
    for i in range(1,5):
        rng[c][i]=all_df[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate',c]].copy()
        rng_list.append([c,i])
p_res = Parallel(n_jobs=-1)(delayed(get_sft)(rng[c][i],c,i,'idx') for c,i in tqdm(rng_list))
all_df = pd.concat([all_df]+p_res,axis=1)
del rng,p_res;gc.collect()

# get rank var
def get_rank(all_df):
    all_df['week_f'] = all_df['date']//7
    for c in tqdm(['baike_id_2h','url','author','mall','author_brand','author_brand_mall']):
        tmp = all_df.groupby(c)['level_1','article_id'].rank()
        all_df[[f'level_1_gp_{c}_rank',f'article_id_gp_{c}_rank']] = tmp
        
        tmp1 = all_df.groupby([c,'date'])['level_1','article_id'].rank()
        all_df[[f'level_1_gp_{c}_daterank',f'article_id_gp_{c}_daterank']] = tmp1

        tmp2 = all_df.groupby([c,'week_f'])['level_1','article_id'].rank()
        all_df[[f'level_1_gp_{c}_weekrank',f'article_id_gp_{c}_weekrank']] = tmp2
all_df.index=range(len(all_df))
get_rank(all_df)

# get train val test
train = all_df[all_df.set=='train']
val = all_df[all_df.set=='val']
test = all_df[all_df.set=='test']
del all_df;gc.collect()
traf = pd.concat([train,val],ignore_index=True)
del train,val;gc.collect()

# get binary target
def get_ymean(df):
    for i in tqdm(range(10)):
        df[f'mean_{i}']=np.where(df['orders_3h_15h']==i,1,0)

get_ymean(traf)

# binary target encoder used in regression
def my_target_encoding(train, test, keys, k = 5):
    oof_train, oof_test = np.zeros((train.shape[0],10)), np.zeros((test.shape[0],10))
    skf = KFold(n_splits = k).split(train)
    t_val = {}
    for i, (train_idx, valid_idx) in enumerate(skf):
        df_train = train[keys+[f'mean_{x}' for x in range(10)]].loc[train_idx]
        df_valid = train[keys].loc[valid_idx]
        t_val[i] = df_valid
        df_map=df_train.groupby(keys).agg({f'mean_{x}':[(f'kfold_{x}_mean','mean')] for x in range(10)})
        t_cols = [f'{x}_{y}' for x,y in df_map.columns]
        df_map.columns = t_cols
        oof_train[valid_idx] = df_valid.merge(df_map, on = keys, how = 'left')[t_cols].fillna(-1).values
        oof_test += test[keys].merge(df_map, on = keys, how = 'left')[t_cols].fillna(-1).values / k
    return oof_train, oof_test, t_cols

for k in tqdm(['author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall','author_brand','author_brand_mall', 'author_l1', 'author_l1-2','author_l1-3', 'author_l1-4', 'author_mall','url', 'baike_id_1h','baike_id_2h','date','week']):
    keys=[k]
    oof_train, oof_test ,t_cols= my_target_encoding(traf, test, keys)
    t_cols1=[k+'_'+x for x in t_cols]
    for c in t_cols1:
        traf[c]=np.nan
        test[c]=np.nan
    traf[t_cols1] = oof_train
    test[t_cols1] = oof_test

for k in tqdm(['author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall','author_brand','author_brand_mall', 'author_l1', 'author_l1-2','author_l1-3', 'author_l1-4', 'author_mall','url', 'baike_id_1h','baike_id_2h']):
    traf[k+'_date'] = traf[k].astype(str)+'_'+traf['date'].astype(str)
    test[k+'_date'] = test[k].astype(str)+'_'+test['date'].astype(str)
    k=k+'_date'
    keys=[k]
    oof_train, oof_test ,t_cols= my_target_encoding(traf, test, keys)
    t_cols1=[k+'_'+x for x in t_cols]
    for c in t_cols1:
        traf[c]=np.nan
        test[c]=np.nan
    traf[t_cols1] = oof_train
    test[t_cols1] = oof_test

traf=joblib.load(r'E:\ecaa\traf_复现.pkl')
test=joblib.load(r'E:\ecaa\test_复现.pkl')

# train model and predict
use_cols = ['price','price_diff', 'author','level1', 'level2', 'level3', 'level4', 'brand', 'mall','comments_1h','zhi_1h', 'buzhi_1h', 'favorite_1h', 'orders_1h','comments_2h', 'zhi_2h', 'buzhi_2h', 'favorite_2h', 'orders_2h', 'week', 'comments_21_diff','url', 'baike_id_1h','baike_id_2h', 'comments_21_rate', 'zhi_21_diff', 'zhi_21_rate', 'buzhi_21_diff', 'buzhi_21_rate', 'favorite_21_diff', 'favorite_21_rate', 'orders_21_diff', 'orders_21_rate']+[x for x in traf.columns if '0810' in x]+[x for x in traf.columns if 'rank' in x]+[x for x in traf.columns if 'shift' in x]
params={'custom_metric':'RMSE',
        'eval_metric':'RMSE',
        'iterations':2818,
        'learning_rate':0.05,
        'random_state':666666,
        'l2_leaf_reg':0,
        'use_best_model':True,
        'depth':10,
        'task_type':'GPU',
        'verbose':100}
cbtf= cb.CatBoostRegressor(**params)
cbtf.fit(traf[use_cols],traf['orders_3h_15h'],eval_set=(traf[use_cols][traf.date>109],traf['orders_3h_15h'][traf.date>109]))
test['pred']=cbtf.predict(test[use_cols])
test['orders_3h_15h']=np.where(test['pred']<0,0,test['pred'])
test[['article_id','orders_3h_15h']].to_csv('./sub/submit.csv',index=False)
