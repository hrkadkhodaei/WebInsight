import pandas as pd

from sklearn.feature_selection import VarianceThreshold, RFECV, RFE

from data_exploration import corr_matrix

def filter_na(X):
	X = X.dropna(axis='index') # drop rows with any missing value
	return X

def filter_variance(X):
	selector = VarianceThreshold(0.001)
	X = selector.fit_transform(X)
	# removed = [ X.columns.values[i] for i in range(len(X.columns.values)) if not selector.get_support()[i] ]
	# print("Features removed by low variance:", removed)
	print("Features selected:", selector.get_support(indices=True))

	return X

def filter_correlation(X):
	method = 'spearman'
	correlations = corr_matrix(X, method, "correlations_"+method+".pdf")
	# print(correlations)

	names = X.columns.values.tolist()
	removed = []

	for col_name, col_series in correlations.items(): # col_name: string, col_series: pd.Series
		for row_name, value in col_series.iteritems(): # items: (string, value)
			if col_name != row_name and abs(value) >= 0.90:
				col_ctg = feature_category(col_name)
				row_ctg = feature_category(row_name)
				print("Spearman correlation:", col_name, col_ctg, row_name, row_ctg, value)
				if col_ctg == row_ctg:
					to_remove = 'F' + str(max(int(row_name[1:]), int(col_name[1:])))
					if to_remove in names:
						names.remove(to_remove)
						removed.append(to_remove)

	print("Features removed by high Spearman correlation:", removed)

	X = X.filter(items=names)
	return X

def filter_RFECV(model, X, y, feature_set):
	# selector = RFECV(model, step=10, cv=3, n_jobs=4)
	selector = RFE(model, step=25)
	X = selector.fit_transform(X, y)

	# removed = [feature_set[i] for i in range(len(feature_set)) if not selector.support_[i]]
	# print("Features removed by recursive feature elimination:", removed)

	indices_selected = selector.get_support(indices=True)
	print("\nFeatures selected:", len(indices_selected))

	return X

"""
def feature_dependence(X):
	D = feature_dependence_matrix(X)
	viz = plot_dependence_heatmap(D, figsize=(11,10))
	viz.save("fdep/"+tnet+"_fdep.pdf")
"""
