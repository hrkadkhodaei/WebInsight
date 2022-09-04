import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

fig_path = 'figures_spearman_correlation_orders/'


def int_to_categorical(x):
    if x == 0:
        return 0
    else:
        return 1


def plot_bar(avg_prediction, avg_prediction_stddev, regressor_prediction, regressor_prediction_stddev):
    labels = ['Internal OutLinks', 'External OutLinks']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    # axes = plt.gca()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylim([.40, 1])
    rects1 = ax.bar(x - width / 2, avg_prediction, width, color='xkcd:coral pink',
                    label='Using average of history')
    rects2 = ax.bar(x + width / 2, regressor_prediction, width,
                    color='xkcd:soft green', label='ExtraTreeClassifier')
    # rects1 = ax.bar(x - width / 2, avg_prediction, width, yerr=avg_prediction_stddev, label='Avg of history')
    # rects2 = ax.bar(x + width / 2, regressor_prediction, width, yerr=regressor_prediction_stddev,
    #                 label='ExtraTreeClassifier')
    # rects1 = ax.bar(x - width/2, a, width, label='avg of history',color='xkcd:coral pink')
    # rects2 = ax.bar(x + width/2, b, width, label='ExtraTreeClassifier',color='xkcd:soft green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Spearman correlation')
    # ax.set_title('Spearman correlation of estimated number \n of new outlinks and the real value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=0)
    ax.bar_label(rects2, padding=0)

    fig.tight_layout()
    plt.savefig(fig_path + 'spearman_corr_orders_numOfOutlinks.png', dpi=500)
    plt.close()
    # plt.show()


if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

fn = r'1M_all_with_diffs_avg.csv'
tuning_fraction, test_fraction = 1. / 3, 1. / 4
Xy = pd.read_csv(fn)
Xy = Xy.set_index('url')
targets = ['diffInternalOutLinks', 'diffExternalOutLinks']
seeds = [0, 1, 2]

avg_estimation_mean = []
regressor_estimation_mean = []
avg_estimation_stddev = []
regressor_estimation_stddev = []
for target in targets:
    avg_estimation_temp = []
    regressor_estimation_temp = []
    for seed in seeds:
        y = Xy[target]  # pd.Series
        # y = y.to_numpy()  # np.array
        X = Xy.drop([target, 'language'], axis='columns')
        # X = X.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                            random_state=seed)
        # X_train_outlinks = X_train[target].to_numpy()
        X_train_avg_outlinks = np.rint(X_test['avg_' + target].to_numpy())
        avg_estimation_temp.append(stats.spearmanr(y_test, X_train_avg_outlinks)[0])

        print("\nRetraining on", len(y_train), "samples")
        model = ExtraTreesRegressor(n_jobs=-1, random_state=seed, n_estimators=400, min_samples_leaf=2)
        model.fit(X_train, y_train)
        #
        y_pred = np.rint(model.predict(X_test))
        regressor_estimation_temp.append(stats.spearmanr(y_test, y_pred)[0])
        # y_true = np.array([int_to_categorical(yi) for yi in y_true])  # binarize the target
        #
        #
        # y_true = np.array([int_to_categorical(yi) for yi in y_true])  # binarize the target
        #
        # y_pred = df['avg' + target].to_numpy()
        # y_pred = np.array([int_to_categorical(round(yi)) for yi in y_pred])
    avg_estimation_mean.append(np.mean(avg_estimation_temp))
    regressor_estimation_mean.append(np.mean(regressor_estimation_temp))
    avg_estimation_stddev.append(np.std(avg_estimation_temp))
    regressor_estimation_stddev.append(np.std(regressor_estimation_temp))
plot_bar(avg_estimation_mean, avg_estimation_stddev, regressor_estimation_mean, regressor_estimation_stddev)

print("avg_history",avg_estimation_mean,avg_estimation_stddev)
print("ETRegressor",regressor_estimation_mean, regressor_estimation_stddev)

print("finished")
