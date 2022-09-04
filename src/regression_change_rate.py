import itertools
from sys import exit
from sklearn.experimental import enable_hist_gradient_boosting  # it is needed for the HistGradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, median_absolute_error, mean_absolute_error, r2_score, make_scorer

from lightgbm import LGBMRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split, ShuffleSplit, learning_curve
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from matplotlib.patches import Rectangle
import seaborn as sns
import definitions
import os

fig_path = 'figures_regression/'


def plot_permutation_feature_importance(model, title, features, target, X_test, y_test, random_state,
                                        max_to_plot=10):
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
    plt.title(
        "Feature permutation importance" + title_suffix + "\ntarget: " + definitions.pretty_label_dict.get(target,
                                                                                                           target),
        fontsize=10)
    fig.tight_layout()
    plt.savefig(fig_path + r"permutation_importance-" + title + ".png",
                dpi=500,
                bbox_inches='tight')

    return list(sorted_idx)  # top feature indices in ascending order


def plot_two_top_features(model, score, score_name, title, X, y, X_names, y_name):
    from matplotlib import gridspec
    fig = plt.figure(figsize=(6.2, 2.8))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[3.53 / 7, 4.47 / 7])
    plt.subplots_adjust(hspace=0.0)

    # colours = plt.cm.bwr
    colours = plt.cm.RdYlGn_r

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

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
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
    plt.savefig(
        fig_path + r"two_features-" + title + '-' + ' '.join(
            X_names) + ".png",
        dpi=500,
        bbox_inches='tight')


def scatterplot_2D(X, y, X_names, y_name):
    """ Expects X dataframe, y array. """

    fig = plt.figure(figsize=(11, 3.75))  # , facecolor='black'
    ax = plt.axes()
    ax.set_facecolor("#909090")  # light grey

    # colours = plt.cm.bwr
    colours = plt.cm.RdYlGn_r

    factor_x = 20
    factor_y = 1.75

    def forward_x(x, factor=factor_x):
        return x ** (1 / factor)

    def inverse_x(x, factor=factor_x):
        return x ** factor

    def forward_y(x, factor=factor_y):
        return x ** (1 / factor)

    def inverse_y(x, factor=factor_y):
        return x ** factor

    plt.xscale('function', functions=(forward_x, inverse_x))
    plt.yscale('function', functions=(forward_y, inverse_y))  # scale_dict.get(X_names[1], 'linear'

    sc = plt.scatter(X.loc[:, X_names[0]], X.loc[:, X_names[1]], marker='o', s=1.25, c=y, cmap=colours)
    cbar = plt.colorbar(sc, label="\n" + y_name, fraction=0.04, aspect=25, pad=0.02)

    # annotate domains for X = ['textSize', 'numInternalOutLinks']
    # each annotation: ((xmin, xmax), (ymin, ymax), "name")
    annotations = [ \
        ((66, 161), (143, 154), "1", "cooltext.com"),  # 94 pages with change rate 1; +2 from other domains
        ((235, 247), (292, 486), "2", "cooltext.com"),  # 21 pages with change rate 1
        ((3954, 6632), (908, 992), "3", "shropshire.gov.uk"),
        # 190 pages with change rate 1; +1 from another domain
        ((3766, 6094), (390, 489), "4", "icpdas-usa.com"),
        # 259 pages, with maybe other domains added on top of these
        ((7230, 8455), (755, 810), "5", "siliconvalleycf.org"),  # 35 pages with change rate 0.(8)
        ((52, 315), (195, 201), "6", "buffettworld.com"),  # 67 pages with change rate 0.(3); +1 from another domain
        ((808, 1062), (552, 725), "7", "picturesof.net"),  # 98 pages with change rate 0.(1); +1 from another domain
        # ((8709, 8830), (1983, 1998), "6", "rimmerbros.com"), # 150 pages with change rate 0.(7)
    ]

    margin = 8
    for a in annotations:
        ax.add_patch(Rectangle((a[0][0] - margin, a[1][0] - margin), a[0][1] - a[0][0] + 2 * margin,
                               a[1][1] - a[1][0] + 2 * margin,
                               edgecolor='white',
                               fill=False,
                               lw=0.75))
        ax.text(a[0][0] - margin, a[1][0] - margin, a[2], size=9,
                horizontalalignment='right', verticalalignment='bottom',
                color="black",
                bbox=dict(facecolor='white', edgecolor="white", pad=0.2, alpha=0.9))

    plt.xlim((40, 13000))
    plt.ylim((.5, 1000))

    # ax.minorticks_on()

    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_yticks([1, 10, 20, 50, 100, 250, 500, 750, 1000])

    plt.xlabel(definitions.pretty_label_dict.get(X_names[0], X_names[0]))
    plt.ylabel(definitions.pretty_label_dict.get(X_names[1], X_names[1]))

    plt.tight_layout(pad=0.5)
    plt.savefig(fig_path + r"annotated-scatterplot-" + ' '.join(X_names) + "-" + y_name + ".png", dpi=200,
                bbox_inches='tight')


def export_scores(score_values, score_names, title):
    fig, ax = plt.subplots(figsize=(2.5, 0.58 + len(score_values) / 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    # ax.spines['left'].set_visible(False)

    y_range = range(len(score_values), 0, -1)
    plt.barh(y_range, width=score_values, height=0.8,
             color=["xkcd:soft green", "xkcd:coral pink", "xkcd:pastel red"])  # "xkcd:sage"
    plt.yticks(y_range, score_names)
    for i in range(len(score_values)):
        if score_values[i] >= 0.3:
            plt.text(score_values[i], y_range[i], "%.2f " % score_values[i], ha='right', va='center')
        else:
            plt.text(score_values[i], y_range[i], " %.2f" % score_values[i], ha='left', va='center')

    plt.tick_params(left=False)
    plt.xlim((0, 1))
    plt.tight_layout()

    plt.savefig(fig_path + r"scores-" + title + ".png", dpi=500,
                bbox_inches='tight')
    plt.close()


def plot_learning_curve(model, title, X, y, ylim=None, cv=3, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10)):
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    plt.figure(figsize=(4, 3))
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Training set size")
    plt.ylabel("MAE score")

    from matplotlib.ticker import FormatStrFormatter
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_ticks_position("both")

    my_scorer = make_scorer(mean_absolute_error)
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=my_scorer, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.2,
                     facecolor='#448CBE')
    plt.plot(train_sizes, train_scores_mean, 's-', markersize=4, color='#448CBE', label="Training")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     facecolor='#B72633')
    plt.plot(train_sizes, test_scores_mean, 'o-', markersize=4, color='#B72633', label="Cross-validation")

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.legend(loc="lower right", markerfirst=True, fontsize=9, frameon=False, labelspacing=0.2)
    plt.savefig(fig_path + "learning_curve-" + title + ".png", dpi=500, bbox_inches='tight')


def corr_matrix(data, labels, method, outfile):
    correlations = data.corr(method=method)  # method : {â€کpearsonâ€™, â€کkendallâ€™, â€کspearmanâ€™}

    num_features = len(data.columns.values)
    fig, ax = plt.subplots(figsize=(2.1 + num_features / 2, 1.1 + num_features / 4))
    mask = np.triu(np.ones_like(correlations, dtype=np.bool))  # masks the top half
    h = sns.heatmap(correlations, yticklabels=labels, mask=mask, square=False, cmap="RdYlGn", annot=True, fmt='.2f',
                    linewidths=1, vmin=-1, vmax=1, cbar=False, ax=ax)
    h.set_xticklabels(labels, rotation=25, ha='right')

    plt.tight_layout(pad=0.5)

    fig.savefig(fig_path + outfile, dpi=500, bbox_inches='tight')

    return correlations


def test_and_score(model, X_test, y_test):
    print("\tTesting on", len(y_test), "samples")
    y_pred = model.predict(X_test)

    scoring = {'R2': r2_score(y_test, y_pred),
               'MAE': mean_absolute_error(y_test, y_pred),
               'MedAE': median_absolute_error(y_test, y_pred)}
    # 'ExVar': explained_variance_score(y_test, y_pred),

    for s in scoring:
        print("\t\tTest", s, scoring[s])

    return scoring


def baseline_score(y_pred, y_test):
    scoring = {'R2': r2_score(y_test, y_pred),
               'MAE': mean_absolute_error(y_test, y_pred),
               'MedAE': median_absolute_error(y_test, y_pred)}
    # 'ExVar': explained_variance_score(y_test, y_pred),

    for s in scoring:
        print("\t\tBaseline", s, scoring[s])

    return scoring


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


if __name__ == '__main__':
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    random_state = 0
    n_jobs = -1
    which_model = 'HGB'  # ET and HGB are most competitive, quick, and configurable; GB far too slow on large data
    # HGB ~ LGB (same alg.); HGB is experimental, so its configuration may need porting in the future
    which_features = ['SP', 'SN']
    # which_features = ['SP']
    num_SV_clusters = 20
    tuning_fraction, test_fraction = 1. / 3, 1. / 4

    # for _content change rate_; don't need this for this manuscript (it used to be Sec. 4.1, but I commented it out)
    # target, new_target = ['changeCount', 'fetchCount'], 'changeRate'
    # dataset_filename = "datasets/changeRate_dataset-SVflat.pkl"

    # for _link change rate_
    target, new_target = ['linkInternalChangeRate'], 'linkInternalChangeRate'
    # target, new_target = ['linkExternalChangeRate'], 'linkExternalChangeRate'
    dataset_filename = r"dataset/1M_all_with_avg_atts.pkl"
    # dataset_filename = r"d:/WebInsight/datasets/1M_all_with_avg_atts.pkl"
    # dataset_filename = r"~/dataset/1M/Pickle/1M_all_with_avg_atts.pkl"

    untuned_models = {
        'ET': [ExtraTreesRegressor(n_jobs=n_jobs, random_state=random_state),
               {'n_estimators': [200, 300, 400, 500],
                'min_samples_leaf': [2, 5, 10, 15, 20, 25]}],
        'HGB': [HistGradientBoostingRegressor(max_depth=None, max_leaf_nodes=None, random_state=random_state),
                {'max_iter': [200, 300, 400, 500],
                 'min_samples_leaf': [10, 15, 20, 25],
                 'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]}],
        # 'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]}],
        'LGB': [LGBMRegressor(max_depth=-1, random_state=random_state, n_jobs=n_jobs),
                {'n_estimators': [3000, 4000, 5000],
                 'min_child_samples': [2, 5, 7],
                 'learning_rate': [0.1]}],
        'GB': [GradientBoostingRegressor(max_depth=-1, random_state=random_state),
               {'n_estimators': [400, 600, 800, 1000],
                'min_samples_leaf': [2, 5],
                'learning_rate': [0.1]}],
        'ETtest': [ExtraTreesRegressor(n_jobs=n_jobs, random_state=random_state),
                   {'n_estimators': [400], 'min_samples_leaf': [2]}],
    }
    pretuned_models = {
        'ET': ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=n_jobs, random_state=random_state),
        'HGB': HistGradientBoostingRegressor(max_iter=400, min_samples_leaf=20, learning_rate=0.02, max_depth=None,
                                             max_leaf_nodes=None, random_state=random_state)
    }

    title = "target " + new_target + "-features " + "_".join(which_features) + "-model " + which_model + "-seed " + str(
        random_state)
    print("\n" + "_" * 80 + "\nRandom state:", random_state)

    # export_scores([0.68, 0.16, 0.11], ['R2', 'MAE', 'MedAE'], title)
    # exit()

    # _________________________________________________________________________________________________
    # Read data, form the target variable
    features = [f for fs in which_features for f in definitions.feature_sets[fs]]
    # Xy = read_dataset(dataset_filename, ['url'] + features + target)  # Xy = pd.DataFrame with url as index
    Xy = pd.read_pickle(dataset_filename)  # Xy = pd.DataFrame with url as index
    # some rows are invalid. They have marked with False value in their 'isValid' attribute
    Xy = Xy[Xy['isValid'] == True]
    Xy = Xy[['url'] + features + target]
    Xy = Xy.set_index('url')

    # 	-> content change rate
    # y = Xy['changeCount'] / (Xy['fetchCount'] - 1) # pd.Series
    # 	-> link change rate
    y = Xy[new_target]  # pd.Series

    y = y.to_numpy()  # np.array
    print("\nTarget:", new_target)
    X = Xy.drop(target, axis='columns')  # X = pd.DataFrame with url as index; still needed to separate SV features

    # _________________________________________________________________________________________________
    # (Optional, not ML) Run the dynamic baseline: y_pred = 0 (or any other value)
    # baseline_y_pred = [0 for i in range(len(y))]
    # _ = baseline_score(baseline_y_pred, y)

    # _________________________________________________________________________________________________
    # (Optional) Dimensionality reduction for the semantic vector
    if 'v' in which_features:
        agglo = FeatureAgglomeration(affinity='cosine', linkage='complete',
                                     n_clusters=num_SV_clusters)  # n_clusters should be hypertuned
        X_SV = agglo.fit_transform(X[definitions.feature_sets['v']])  # X_SV = np.array
        print("Features reduced to", X_SV.shape)

        # merge back into a pd.DataFrame X[without SV features] with the now reduced X_SV
        X = X.drop(definitions.feature_sets['v'], axis='columns')  # X = pd.DataFrame
        new_columns = ["SVCluster" + str(i) for i in range(num_SV_clusters)]
        X_SV = pd.DataFrame(X_SV, index=X.index.values, columns=new_columns)
        X[new_columns] = X_SV[new_columns]

    # _________________________________________________________________________________________________
    # (Optional) Any stats?
    # plot_univariate_distribution(y, 9, (0, 1), pretty_label_dict.get(new_target, new_target), "figures-changeRate/distribution_"+new_target+".png")
    # scatterplot_2D(X, y, ['textSize', 'numInternalOutLinks'], pretty_label_dict.get(new_target, new_target))

    if not 'v' in which_features:
        corr_matrix(X, [definitions.pretty_label_dict.get(f, f) for f in X.columns.values],
                    'spearman', "corrmatrix-target_" + new_target + "-model " + which_model + "-features " + "_".join(
                which_features) + ".png")
    # exit()  # remove this when needed

    # _________________________________________________________________________________________________
    # Build a dictionary of feature names, for later reference
    feature_names = X.columns.values
    feature_index_to_name = dict(enumerate(X.columns.values))
    print("\nFinal enumerated feature set (" + "_".join(which_features) + "):\n\t", feature_index_to_name, "\n")
    X = X.to_numpy()  # np.ndarray; X column names are lost from X itself now

    # _________________________________________________________________________________________________
    # Define the model, with feature scaling; the semantic vector needs no scaling, but it's hard to separate it
    # model = Pipeline(steps=[	('scaler', PowerTransformer()),
    # 							('model', untuned_models[which_model])])
    # ...or without if no scaling needed
    model = untuned_models[which_model][0]
    params = untuned_models[which_model][1]
    print("Model:", which_model)

    # (Alternative) transformed target, but not effective on this changeRate distribution (too discrete)
    # model = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='uniform'))

    # _________________________________________________________________________________________________
    # (Step 0) Set aside a fraction of test data
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                    random_state=random_state)

    # _________________________________________________________________________________________________
    # (Step 1) Tune hyperparameters on a fraction of the development data
    X_tune, _, y_tune, _ = train_test_split(X_dev, y_dev, train_size=tuning_fraction, shuffle=True,
                                            random_state=random_state)  # for lack of a simpler split function
    tuned_model = hyperparameter_tuning(model, params, X_tune, y_tune, cv=5, n_jobs=n_jobs)
    #
    # (Alternative) preturned model, based on prior runs with tuning
    # tuned_model = pretuned_models[which_model]

    # _________________________________________________________________________________________________
    # (Step 2) Refit a single model on all development data
    print("\nRetraining on", len(y_dev), "samples")
    tuned_model.fit(X_dev, y_dev)

    # _________________________________________________________________________________________________
    # (Step 3) Test it on the test data
    scoring = test_and_score(tuned_model, X_test, y_test)
    export_scores(list(scoring.values()), list(scoring.keys()), title)

    # exit()
    # _________________________________________________________________________________________________
    # (Step 4, optional) Get permutation feature importance scores, or paste it from a previous run
    # top_feature_indices = [4, 3, 6, 2, 1, 5, 0] # manual
    top_feature_indices = plot_permutation_feature_importance(tuned_model,
                                                              title, feature_names, new_target, X_test, y_test,
                                                              random_state)
    # top_feature_indices = [3, 0, 5, 21]
    # _________________________________________________________________________________________________
    # (Optional) Refit a model on 2 top features only, to visualise the decision boundaries
    print("\nRefitting on two features for decision boundaries")
    # top_feature_pairs = itertools.permutations(top_feature_indices[-4:], r=2)
    top_feature_pairs = itertools.combinations(top_feature_indices[-4:], r=2)  # (0,3) & (3,0) are equal
    top_feature_pairs = [list(t) for t in top_feature_pairs]  # tuple to list
    # b = [(min(r), max(r)) for r in top_feature_pairs]
    # c = set(b)
    # d = [list(r) for r in c]
    # top_feature_pairs = d
    for top_feature_indices in top_feature_pairs:
        top_feature_names = np.array(feature_names)[top_feature_indices]
        tuned_model.fit(X_dev[:, top_feature_indices], y_dev)
        y_pred = tuned_model.predict(X_test[:, top_feature_indices])
        score, score_name = mean_absolute_error(y_test, y_pred), 'MAE'
        print("\t", top_feature_indices, score_name + ':', score)
        plot_two_top_features(tuned_model, score, score_name,
                              title, X[:, top_feature_indices], y, top_feature_names,
                              definitions.pretty_label_dict.get(new_target, new_target))
