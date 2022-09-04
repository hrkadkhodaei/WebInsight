import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import explained_variance_score, median_absolute_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, \
    fbeta_score, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split
import seaborn as sns
import time

start_time = time.time()

static_page_features = ['contentLength', 'textSize', 'textQuality', 'pathDepth', 'domainDepth', 'numInternalOutLinks',
                        'numExternalOutLinks']  #
static_page_semantics = ["SV" + str(i) for i in range(192)]
static_network_features = ['numInternalInLinks', 'numExternalInLinks', 'trustRank']
static_network_features += ['related_' + f for f in static_page_features + static_network_features]
dynamic_network_features = ['numInternalInLinks-', 'numExternalInLinks-', 'trustRank-']
dynamic_page_features = ['contentLength-', 'textSize-', 'textQuality-', 'diffInternalOutLinks-',
                         'diffExternalOutLinks-']

feature_sets = {
    'SP': static_page_features,
    'v': static_page_semantics,
    'SN': static_network_features,
}

feature_sets.update(dict([
    ('DP' + str(i + 1), [f + str(j + 1) for j in range(i + 1) for f in dynamic_page_features])
    for i in range(8)
]))

feature_sets['DPRate'] = ['related_linkExternalChangeRate', 'related_linkInternalChangeRate']

feature_sets.update(dict([
    ('DN' + str(i + 1), [f + str(j + 1) for j in range(i + 1) for f in dynamic_network_features])
    for i in range(8)
]))  # 'DP/N8' will contain all -1, ... -8 dynamic page features

fig_path = 'figures_spearman_correlation_orders/'


def int_to_categorical(x):
    if x > 0:
        return 1
    else:
        return 0


def calc_content_change_rate(row):
    s = 0
    for i in range(7):
        if row['contentLength-' + str(i + 1)] != row['contentLength-' + str(i + 2)]:
            s += 1
    if row['contentLength-8'] != row['contentLength']:
        s += 1
    return s / 8


def hyperparameter_tuning(model, params, X_tune, y_tune, cv=5, n_jobs=-1):
    print("\nHyperparameter tuning on", len(y_tune), "samples")

    my_scorer = make_scorer(r2_score)
    tuned_model = GridSearchCV(model, params, cv=cv, scoring=my_scorer, n_jobs=n_jobs)
    tuned_model.fit(X_tune, y_tune)

    print("\n\tScores on the development set:\n")
    means = tuned_model.cv_results_['mean_test_score']
    stds = tuned_model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, tuned_model.cv_results_['params']):
        print("\tmean %0.5f, stdev %0.05f for parameters %r" % (mean, std, params))

    print("\n\tBest parameters on the development set:", tuned_model.best_params_)
    model = tuned_model.best_estimator_

    return model


def corr_matrix(data, labels, method, filename):
    correlations = data.corr(method=method)  # method : {'pearson', 'kendall', 'spearman'}
    # print(correlations)
    num_features = len(data.columns.values)
    fig, ax = plt.subplots(figsize=(2.1 + num_features / 2, 1.1 + num_features / 4))
    mask = np.triu(np.ones_like(correlations, dtype=np.bool_))  # masks the top half
    h = sns.heatmap(correlations, yticklabels=labels, mask=mask, square=False, cmap="RdYlGn", annot=True, fmt='.2f',
                    linewidths=1, vmin=-1, vmax=1, cbar=False, ax=ax)
    h.set_xticklabels(labels, rotation=25, ha='right')

    plt.tight_layout(pad=0.5)

    fig.savefig(filename, dpi=500, bbox_inches='tight')

    return correlations


total_target = 'Internal'
target_feature = 'diff' + total_target + 'OutLinks'
seed = 0

if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

which_features = ['SP', 'SN', 'DN8', 'DP8', 'DPRate']
which_features = ['SP', 'SN']
which_features = ['SP', 'SN', 'DP8', 'DPRate']
features = [f for fs in which_features for f in feature_sets[fs]]
untuned_models = {
    'ET': [ExtraTreesRegressor(n_jobs=-1, random_state=seed),
           {'n_estimators': [50, 200, 300, 400, 500], 'min_samples_leaf': [2, 5, 10]}],
    'ETC': [ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=seed),
            {'n_estimators': [50, 200, 300, 400, 500],
             'min_samples_leaf': [2, 5, 10, 15]}],
}

pretuned_models = {
    'ET': ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    'ETC': ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed)
}

model1 = untuned_models['ET'][0]
params1 = untuned_models['ET'][1]

fn = r'dataset/1M_all_with_diffs_avg_linkChangeRate.csv'
tuning_fraction, test_fraction = 1. / 4, 1. / 4
Xy = pd.read_csv(fn)

Xy = Xy[Xy['isValid'] == True]
Xy = Xy.set_index('url')
y = Xy[target_feature]  # pd.Series
# y = y.to_numpy()  # np.array
X = Xy.drop([target_feature], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True, random_state=seed)

df = pd.DataFrame()
labels = []

df['link_change_rate'] = X_test['link' + total_target + 'ChangeRate']
labels.append('link_change_rate')

ytr1 = X_train['link' + total_target + 'ChangeRate']
# ytest1 = X_test['link' + total_target + 'ChangeRate']
X_train = X_train[features]
X_test = X_test[features]
# cr = df.corr(method='spearman')  # pearson or spearman
# X_tune, _, y_tune, _ = train_test_split(X_train, ytr1, train_size=tuning_fraction, shuffle=True,
#                                         random_state=seed)  # for lack of a simpler split function
print(X_train.columns.values)
# tuned_model = hyperparameter_tuning(model1, params1, X_tune, y_tune, cv=4, n_jobs=-1)
tuned_model = pretuned_models['ET']

print("training first model")
# model = ExtraTreesRegressor(n_jobs=-1, random_state=seed, n_estimators=100, min_samples_leaf=2)
tuned_model.fit(X_train, ytr1)

y_pred = tuned_model.predict(X_test)
df['link_change_rate_ET'] = y_pred
labels.append('link_change_rate_ET')

# X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=tuning_fraction, shuffle=True,
#                                         random_state=seed)  # for lack of a simpler split function
#
# # tuned_model = hyperparameter_tuning(model1, params1, X_tune, y_tune, cv=4, n_jobs=-1)
# tuned_model = pretuned_models['ET']
#
# # tuned_model = ExtraTreesRegressor(n_jobs=-1, random_state=seed, n_estimators=100, min_samples_leaf=2)
# print("training 2nd model")
# tuned_model.fit(X_train, y_train)
# elapsed_time = time.time() - start_time
# print("until now, Total time: ", elapsed_time)
#
# y_pred = tuned_model.predict(X_test)
#
# y_pred1 = np.rint(y_pred)
#
# df['num_new_outlinks_ET'] = y_pred1
#
# labels.append('num_new_outlinks_ET')
#
# y_train = np.array([int_to_categorical(yi) for yi in y_train])
# X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=tuning_fraction, shuffle=True,
#                                         random_state=seed)  # for lack of a simpler split function
#
# model2 = untuned_models['ETC'][0]
# params2 = untuned_models['ETC'][1]
#
# # tuned_model = hyperparameter_tuning(model2, params2, X_tune, y_tune, cv=4, n_jobs=-1)
# tuned_model = pretuned_models['ETC']
#
# # tuned_model = ExtraTreesClassifier(n_jobs=-1, random_state=seed, n_estimators=100, min_samples_leaf=2)
# print("training 3rd model")
# tuned_model.fit(X_train, y_train)
# elapsed_time = time.time() - start_time
#
# print("until now, Total time: ", elapsed_time)
#
# y_pred = tuned_model.predict_proba(X_test)
# df['prob_of_new_link'] = y_pred[:, 1]
# labels.append('prob_of_new_link')
# df['new_link_using_ET'] = np.rint(df['prob_of_new_link'])
# labels.append('new_link_using_ET')
#
# df['prob_of_new_link'] = round(df['prob_of_new_link'], 2)
# df['new_link_in_week_10'] = np.array([int_to_categorical(yi) for yi in y_test])
# labels.append('new_link_in_week_10')
# labels = ['num_new_outlinks_avg_history', 'num_new_outlinks_ET', 'link_change_rate', 'related_link_change_rate',
#           'link_change_rate_ET',
#           'content_change_rate', 'prob_of_new_link', 'new_link_using_ET', 'new_link_in_week_10',
#           'num_new_outlinks_week_10']
# labels2 = ['num_new_outlinks_week_10', 'num_new_outlinks_avg_history', 'num_new_outlinks_ET', 'link_change_rate',
#            'related_link_change_rate',
#            'link_change_rate_ET', 'content_change_rate', 'prob_of_new_link', 'new_link_using_ET', 'new_link_in_week_10']
df1 = df[labels]
df1.to_csv(fig_path + 'LCR_orders_' + total_target + '_' + '_'.join(which_features) + '.csv', header=True, index=False)
# labels = ['Order-' + str(i + 1) for i in range(10)]
cr = corr_matrix(data=df1, labels=labels, method='spearman',
                 filename=fig_path + total_target + '_different_orders_' + '_'.join(which_features) + '.png')

elapsed_time = time.time() - start_time

print("finished. Total time: ", elapsed_time)
