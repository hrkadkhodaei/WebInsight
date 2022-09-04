import numpy as np
import pandas as pd
# import pkg_resources
# import itertools
# import shap
# import math
# import logging
# import multiprocessing
# from scipy.stats import norm
import matplotlib.pyplot as plt
# import plotnine
# from plotnine import *
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from ngboost.distns import Poisson, Normal
from ngboost import NGBRegressor
import definitions
from sklearn.preprocessing import StandardScaler

which_features = ['SP', 'SN'] + [f'DP{i}' for i in range(1, 9)] + [f'DN{i}' for i in range(1, 9)]
features = [f for fs in which_features for f in definitions.feature_sets[fs]]


def read_data(InEx, percent_zeros=0):
    print(datetime.now().strftime("%H:%M:%S"), "\n")
    # target = f'link{InEx}ChangeRate'
    target = f'diff{InEx}OutLinks'
    # atts += [f'{prefix}num{InEx}InLinks-{i}' for i in range(1, 9)]
    related = 'related_'
    a = [f'{related}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{related}diff{InEx}OutLinks']
    b = [f'{related}diffExternalOutLinks-{i}' for i in range(1, 9)] + [f'{related}diffExternalOutLinks']
    atts = [f'{related}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{related}diff{InEx}OutLinks']
    atts += [f'{related}linkInternalChangeRate'] + [f'{related}linkExternalChangeRate']
    atts += [f'related_avg_diff{InEx}OutLinks']
    # atts += [f'diff{InEx}OutLinks-{i}' for i in range(1, 9)]
    atts += [f'avg_diff{InEx}OutLinks']
    # atts += [f'diffInternalOutLinks', 'diffExternalOutLinks']
    atts += features
    atts = list(set(atts))

    path = r'dataset/'

    fn_orders = path + fr'orders\{InEx}_orders-NGB.csv'
    fn = path + r'url_all_data.pkl'

    df_orders = pd.read_csv(fn_orders)
    test_urls = list(df_orders['URL'])
    url_orders = set(test_urls)

    # df_orders_non_zero = df_orders.loc[df_orders['num_new_outlinks_week_10'] > 0]
    # df_orders_zero = df_orders.loc[df_orders['num_new_outlinks_week_10'] == 0].sample(frac=percent_zeros,
    #                                                                                   random_state=123)
    # url_orders_non_zero = list(df_orders_non_zero['URL']) + list(df_orders_zero['URL'])

    df = pd.read_pickle(fn)
    # df = pd.read_csv(fn)
    df[f'related_avg_diff{InEx}OutLinks'] = df[a].mean(axis=1)
    df[f'related_avg_diffExternalOutLinks'] = df[b].mean(axis=1)
    # df = df[atts + ['url'] + [target]]
    url_all = set(df['url'])

    url_train = url_all - url_orders
    df.set_index('url', inplace=True)
    df_train = df.loc[url_train]

    path = 'temp/'

    a = df_train.reset_index()
    a.to_csv(path + f'{InEx}_train.csv', index=False, header=True)
    # df_train = df_train.loc[df_train[target] > 0]
    # df_train = pd.concat([df_train, df_train_zeros], axis=0)
    X_train = df_train[atts]
    y_train = df_train[target]
    y_train = y_train.apply(lambda x: 1 if x > 0 else 0)

    # df_result = pd.DataFrame()
    # df_result['url'] = test_urls

    df_test = df.loc[url_orders]
    b = df_test.reset_index()
    b.to_csv(path + f'{InEx}_test.csv', index=False, header=True)

    X_test = df_test[atts]
    y_test = df_test[target]
    y_test = y_test.apply(lambda x: 1 if x > 0 else 0)
    return X_train, y_train, X_test, y_test


def read_data2(InEx):
    print(datetime.now().strftime("%H:%M:%S"), "\n")
    # target = f'link{InEx}ChangeRate'
    target = f'diff{InEx}OutLinks'
    # atts += [f'{prefix}num{InEx}InLinks-{i}' for i in range(1, 9)]
    related = 'related_'
    a = [f'{related}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{related}diff{InEx}OutLinks']
    atts = [f'{related}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{related}diff{InEx}OutLinks']
    atts += [f'{related}linkInternalChangeRate'] + [f'{related}linkExternalChangeRate']
    atts += [f'related_avg_diff{InEx}OutLinks']
    # atts += [f'diff{InEx}OutLinks-{i}' for i in range(1, 9)]
    atts += [f'avg_diff{InEx}OutLinks']
    # atts += [f'diffInternalOutLinks', 'diffExternalOutLinks']
    atts += features
    atts = set(atts)
    atts = list(atts)

    fn_orders = path + fr'{InEx}_orders-NGBCC.csv'
    fn = path + r'1M_Final.pkl'

    df_orders = pd.read_csv(fn_orders)
    test_urls = list(df_orders['URL'])
    url_orders = set(test_urls)

    # df_orders_non_zero = df_orders.loc[df_orders['num_new_outlinks_week_10'] > 0]
    # df_orders_zero = df_orders.loc[df_orders['num_new_outlinks_week_10'] == 0].sample(frac=percent_zeros,
    #                                                                                   random_state=123)
    # url_orders_non_zero = list(df_orders_non_zero['URL']) + list(df_orders_zero['URL'])

    df = pd.read_pickle(fn)
    # df = df.loc[df[f'diff{InEx}OutLinks'] > 1].sample(n=1000, replace=False)
    # df = pd.read_csv(path + 'test2.csv')
    df[f'related_avg_diff{InEx}OutLinks'] = df[a].mean(axis=1)
    df = df[atts + ['url'] + [target]]
    url_all = set(df['url'])

    url_train = url_all - url_orders
    df.set_index('url', inplace=True)
    df_train = df.loc[url_train]
    # df_train_zeros = df_train.loc[df_train[target] == 0].sample(frac=percent_zeros, random_state=123)
    # df_train = df_train.loc[df_train[target] > 0]
    # df_train = pd.concat([df_train, df_train_zeros], axis=0)
    X_train = df_train[atts]
    y_train = df_train[target]

    df_result['url'] = test_urls

    df_test = df.loc[test_urls]
    X_test = df_test[atts]
    y_test = df_test[target]
    df_result['y_true'] = y_test.values
    return X_train, y_train, X_test, y_test


state = 0
path = r'output/'

InExs = ['Internal', 'External']
# InExs = ['External']
print("start reading data")

for InEx in InExs:
    df_result = pd.DataFrame()

    X_train, y_train, X_test, y_test = read_data2(InEx)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # X_dev, _, y_dev, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=state)

    ngb = NGBRegressor(Dist=Normal, n_estimators=500, verbose=True, verbose_eval=5)
    ngb.fit(X=X_train, Y=y_train)
    # preds = ngb.predict(X_test)

    aa = ngb.pred_dist(X_test)
    df_result.loc[:, 'loc'] = aa.loc
    df_result.loc[:, 'scale'] = aa.scale
    df_result.to_csv(path + f'{InEx}_ngb_normal_pred_dist.csv', index=False, header=True)
    print(f"{InEx} done")
print('-' * 50, '\n', "done")
