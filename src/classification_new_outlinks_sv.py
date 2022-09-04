from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sys import exit

from sklearn import __version__
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, \
    fbeta_score, make_scorer, confusion_matrix, classification_report
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

# from data_exploration import read_dataset, plot_categorical_distribution, corr_matrix
# from feature_selection import filter_RFECV
import definitions
import os

fig_path = 'figures-newOutlinks/'


def int_to_categorical(x):
    if x == 0:
        return 0
    # elif x >= 1 and x <= 5:
    # 	return 1
    else:
        return 1


def hyperparameter_tuning(model, params, X_tune, y_tune, cv=4, n_jobs=-1):
    print("\nHyperparameter tuning on", len(y_tune), "samples")

    my_scorer = make_scorer(balanced_accuracy_score)
    # my_scorer = make_scorer(f1_score)
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


def export_confusion_matrix(y, y_pred, filename):
    # assumption: all classes are represented in the test set
    class_names = sorted(list(set(y)))  # used to explicitly order the CM

    # CM[i,j] = samples in true class i, predicted to be in class j
    CM = confusion_matrix(y, y_pred, labels=class_names)

    # normalise CM to percentages
    CM_norm = 100 * CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]  # row=100
    # CM_norm = 100 * CM.astype('float') / CM.sum(axis=0) # col=100

    size = len(class_names)

    plt.figure(figsize=(0.9 + 0.35 * size, 0.7 + 0.35 * size))
    plt.imshow(CM_norm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=100)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = list(range(size))
    plt.xticks(tick_marks, [definitions.pretty_class_names[c] for c in class_names])  # , rotation=70)
    plt.yticks(tick_marks, [definitions.pretty_class_names[c] for c in class_names])
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "black"
            if CM_norm[i][j] >= 70:
                color = "white"
            plt.text(j, i, int(round(CM_norm[i][j])), color=color, ha='center', va='center')

    plt.tight_layout(pad=0.5)

    plt.savefig(filename, dpi=500)
    plt.close()

    return CM


def test_and_score(model, X_test, y_test, title, y_pred=None):
    print("\tTesting on", len(y_test), "samples")
    if not y_pred and model:
        y_pred = model.predict(X_test)

    export_confusion_matrix(y_test, y_pred, fig_path + "confusion_matrix-" + title + ".png")

    scoring = {
        'Recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'Precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'F1 score': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        'Accuracy (b)': balanced_accuracy_score(y_test, y_pred)
    }

    for s in scoring:
        print("\t\tTest", s, scoring[s])

    return scoring


def export_scores(score_values, score_names, title):
    fig, ax = plt.subplots(figsize=(2.5, 0.58 + len(score_values) / 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    # ax.spines['left'].set_visible(False)

    y_range = range(len(score_values), 0, -1)
    plt.barh(y_range, width=score_values, height=0.8,
             color=["xkcd:soft green", "xkcd:greyish green", "#98b898", "#b0d0b0"])
    plt.yticks(y_range, score_names)
    for i in range(len(score_values)):
        if score_values[i] >= 0.3:
            plt.text(score_values[i], y_range[i], "%.2f " % score_values[i], ha='right', va='center')
        else:
            plt.text(score_values[i], y_range[i], " %.2f" % score_values[i], ha='left', va='center')

    plt.tick_params(left=False)
    plt.xlim((0, 1))
    plt.tight_layout()

    plt.savefig(fig_path + "scores-" + title + ".png", dpi=500, bbox_inches='tight')
    plt.close()


def report_performance(y_true, y_pred):
    score_names = ["F1-score", "Recall", "Accuracy (b)", "Precision"]
    score_values = [
        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        balanced_accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    ]

    print()
    for i in range(len(score_names)):
        print(score_names[i], ":", score_values[i])

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    """ CM[i,j] = samples in true class i, predicted to be in class j """
    class_names = sorted(list(set(y_true)))  # used to explicitly order the CM
    CM = confusion_matrix(y_true, y_pred, labels=class_names)
    print("Confusion matrix:")
    print(CM)

    return score_values, score_names


def plot_permutation_feature_importance(model, title, features, target, X_test, y_test, random_state, max_to_plot=10):
    # Source: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=random_state)
    sorted_idx = result.importances_mean.argsort()
    print("\nTop features in ascending order of importance:\n\t", list(np.array(features)[sorted_idx]), "or",
          list(sorted_idx))

    pretty_features = [definitions.pretty_label_dict.get(f, f) for f in features]
    title_suffix = ""
    num_features = len(features)
    if max_to_plot:
        if max_to_plot < num_features:
            sorted_idx = sorted_idx[-max_to_plot:]
            title_suffix = " (top 10 features)"
            num_features = max_to_plot

    fig = plt.figure(figsize=(4.1, 0.75 + num_features / 6))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(pretty_features)[sorted_idx])
    plt.title("Feature permutation importance" + title_suffix + "\ntarget: " + definitions.pretty_label_dict.get(target,
                                                                                                                 target),
              fontsize=10)
    fig.tight_layout()
    plt.savefig(fig_path + "permutation_importance-" + title + ".png", dpi=500, bbox_inches='tight')

    return list(sorted_idx)  # top feature indices in ascending order


def plot_two_top_features(model, score, score_name, title, X, y, X_names, y_name):
    from matplotlib import gridspec
    fig = plt.figure(figsize=(6.2, 2.8))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[3.53 / 7, 4.47 / 7])
    plt.subplots_adjust(hspace=0.0)

    colours = plt.cm.bwr
    # colours = plt.cm.RdYlGn_r

    x_min, x_max = 0.9 * X[:, 0].min() - 0.25, X[:, 0].max() + 0.2 * (X[:, 0].max() - X[:, 0].min())
    y_min, y_max = 0.9 * X[:, 1].min() - 0.25, X[:, 1].max() + 0.2 * (X[:, 1].max() - X[:, 1].min())
    if definitions.scale_dict.get(X_names[0], 'linear') not in ['log', 'symlog']:
        x_min, x_max = X[:, 0].min() - .02, X[:, 0].max() + .02
    if definitions.scale_dict.get(X_names[1], 'linear') not in ['log', 'symlog']:
        y_min, y_max = X[:, 1].min() - .02, X[:, 1].max() + .02

    # ____________________________________________________________
    # The first figure: a scatterplot of raw data points

    ax = fig.add_subplot(spec[0])

    ax.set_xscale(definitions.scale_dict.get(X_names[0], 'linear'))
    ax.set_yscale(definitions.scale_dict.get(X_names[1], 'linear'))

    sc = ax.scatter(X[:, 0], X[:, 1], marker='.', s=2, c=y, cmap=colours)

    ax.tick_params(which='minor', length=0)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    ax.set_xlabel(definitions.pretty_label_dict.get(X_names[0], X_names[0]))
    ax.set_ylabel(definitions.pretty_label_dict.get(X_names[1], X_names[1]))

    # ____________________________________________________________
    # The second figure: the statistical model trained on the data
    ax = fig.add_subplot(spec[1])

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, (x_max - x_min) / 2500),
        np.arange(y_min, y_max, (y_max - y_min) / 2500))

    ax.set_xscale(definitions.scale_dict.get(X_names[0], 'linear'))
    ax.set_yscale(definitions.scale_dict.get(X_names[1], 'linear'))

    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    cn = ax.contourf(xx, yy, Z, levels=np.linspace(0.0, 1.0, 11), cmap=colours)
    cbar = plt.colorbar(cn, label="\n" + y_name, ticks=np.linspace(0.0, 1.0, 11))

    ax.tick_params(which='minor', length=0)

    ax.set_xlabel(definitions.pretty_label_dict.get(X_names[0], X_names[0]))
    # ax.set_ylabel(X_names[1])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.yaxis.set_tick_params(left=False)

    ax.text(xx.max(), yy.min(),
            score_name + ': %.2f  \n' % score, size=10, horizontalalignment='right')

    # ____________________________________________________________
    # Save the composite figure
    fig.tight_layout(pad=0)
    plt.savefig(fig_path + "two_features-" + title + '-' + ' '.join(X_names) + ".png", dpi=500,
                bbox_inches='tight')


# _____________________________________________________________________________
if __name__ == '__main__':
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    n_jobs = -1
    which_model = 'ET'  # HGB cannot be configured with balanced class weights (doesn't have the option);
    # HGB thus replaced by LGB, a more complete implementation of the same alg.
    num_SV_clusters = 20
    tuning_fraction, test_fraction = 1. / 4, 1. / 4

    target, new_target = ['diffExternalOutLinks'], 'diffExternalOutLinks'
    # target, new_target = ['diffInternalOutLinks'], 'diffInternalOutLinks'
    path = r"dataset/"
    file_name = 'All_data_avg_and_DP_atts.pkl'
    mainXy = pd.read_pickle(path + file_name)
    # Xy = read_dataset(path + file_name, ['url'] + features + target)  # Xy = pd.DataFrame with url as index

    for dur in range(8):
    # for dur in [7]:
        which_features = ['SP', 'SN'] + ['DP' + str(dur + 1)] + ['DN' + str(dur + 1)] + ['DPRateE', 'DPRateI']
        # which_features = ['SP'] + ['DN' + str(dur + 1)]
        # which_features = ['SP', 'SN']
        features = [f for fs in which_features for f in definitions.feature_sets[fs]]

        Xy = mainXy.filter(items=['url'] + features + target)
        Xy = Xy.set_index('url')
        print("------------------------------------------------------------")
        print(new_target + "  -->  " + 'DP' + str(dur + 1))
        model_is_tuned = False
        first_run = True
        tuned_model = None
        for rand_state in range(3):
            untuned_models = {
                'ET': [ExtraTreesClassifier(class_weight="balanced", n_jobs=n_jobs, random_state=rand_state),
                       {'n_estimators': [200, 300, 400, 500],
                        'min_samples_leaf': [5, 10, 15]}],
                'LGB': [LGBMClassifier(objective='binary', is_unbalance=True, max_depth=None, num_leaves=None,
                                       n_jobs=n_jobs, random_state=rand_state),
                        {'n_estimators': [2000, 3000, 4000, 5000, 6000, 7000],
                         'min_child_samples': [10, 15, 20, 25],
                         'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]}],
                'HGB': [HistGradientBoostingClassifier(max_depth=None, max_leaf_nodes=None, random_state=rand_state),
                        {'max_iter': [200, 300, 400, 500],
                         'min_samples_leaf': [10, 15, 20, 25],
                         'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]}],
            }
            pretuned_models = {
                'ET': ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, class_weight="balanced",
                                           n_jobs=n_jobs, random_state=rand_state),
            }

            title = "target " + new_target + "-features " + "_".join(
                which_features) + "-model " + which_model + "-seed " + str(rand_state)
            print("\n" + "_" * 80 + "\nRandom state:", rand_state)

            if 'v' in which_features:
                agglo = FeatureAgglomeration(affinity='cosine', linkage='complete',
                                             n_clusters=num_SV_clusters)  # n_clusters should be hypertuned
                X_SV = agglo.fit_transform(X[definitions.feature_sets['v']])  # X_SV = np.array
                print("Features reduced to", X_SV.shape)
                X = X.drop(definitions.feature_sets['v'], axis='columns')  # X = pd.DataFrame
                new_columns = ["SVCluster" + str(i) for i in range(num_SV_clusters)]
                X_SV = pd.DataFrame(X_SV, index=X.index.values, columns=new_columns)
                X[new_columns] = X_SV[new_columns]

            y = Xy[new_target]  # pd.Series
            y = np.array([int_to_categorical(yi) for yi in y])  # np.array
            now = datetime.now()
            print(now.strftime("%Y/%m/%d %H:%M:%S"))
            print("\nTarget:", new_target)
            X = Xy.drop(target,
                        axis='columns')  # X = pd.DataFrame with url as index; still needed to separate SV features
            feature_names = X.columns.values
            feature_index_to_name = dict(enumerate(X.columns.values))
            print("\nFinal enumerated feature set (" + "_".join(which_features) + "):\n\t", feature_index_to_name, "\n")
            X = X.to_numpy()  # np.ndarray; X column names are lost from X itself now

            model = untuned_models[which_model][0]
            params = untuned_models[which_model][1]
            print("Model:", which_model)

            X_dev, X_test, y_dev, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                            random_state=rand_state)

            X_tune, _, y_tune, _ = train_test_split(X_dev, y_dev, train_size=tuning_fraction, shuffle=True,
                                                    random_state=rand_state)  # for lack of a simpler split function
            if not model_is_tuned:
                tuned_model = hyperparameter_tuning(model, params, X_tune, y_tune, cv=5, n_jobs=n_jobs)
                model_is_tuned = True

            print("\nRetraining on", len(y_dev), "samples")
            tuned_model.fit(X_dev, y_dev)

            scoring = test_and_score(tuned_model, X_test, y_test, title)  # includes confusion matrix
            export_scores(list(scoring.values()), list(scoring.keys()), title)

            if first_run:
                first_run = False
                top_feature_indices = plot_permutation_feature_importance(tuned_model,
                                                                          title, feature_names, new_target, X_test,
                                                                          y_test,
                                                                          rand_state)

                print("\nRefitting on two features for decision boundaries")
                top_feature_pairs = itertools.combinations(top_feature_indices[-4:], r=2)
                top_feature_pairs = [list(t) for t in top_feature_pairs]  # tuple to list
                for top_feature_indices in top_feature_pairs:
                    top_feature_names = np.array(feature_names)[top_feature_indices]
                    tuned_model.fit(X_dev[:, top_feature_indices], y_dev)
                    y_pred = tuned_model.predict(X_test[:, top_feature_indices])
                    score, score_name = f1_score(y_test, y_pred, pos_label=1, zero_division=0), 'F1 score'
                    print("\t", top_feature_indices, score_name + ':', score)
                    plot_two_top_features(tuned_model, score, score_name,
                                          title, X[:, top_feature_indices], y, top_feature_names,
                                          definitions.pretty_label_dict.get(new_target, new_target))
