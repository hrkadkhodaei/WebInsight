import numpy as np
import pandas as pd
import seaborn as sns
import time
import json
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from sys import exit
# from rfpimp import feature_dependence_matrix, plot_dependence_heatmap

def flatten_json_to_json(original_filename, nrows, new_filename):
	df = pd.read_json(original_filename, lines=True, nrows=nrows)
	print("\nOriginal dataset \n#rows:", len(df), "\ncolumns:\n", list(df.columns.values))

	def list_of_coords(str_json_array):
		lst = json.loads(str_json_array)
		return dict( [ ("SV"+str(i), lst[i]) for i in range(len(lst)) ] )

	# Flatten an array of 192 floats, encoded as a one-column string
	if 'semanticVector' in df.columns.values:
		df = df[df['semanticVector'] != 'not set']
		df = df.reset_index(drop=True) # re-index from zero after dropping the rows above
		df_flat = pd.json_normalize(df['semanticVector'].map(list_of_coords).tolist()) # a new 1920column df from the vector column  
		df = df.drop('semanticVector', axis='columns').join(df_flat) # merge the two, which have the same indexes

	print("\nNew dataset \n#rows:", len(df), "\ncolumns:\n", list(df.columns.values))

	with open(new_filename, 'w') as fout:
		for json_dict in df.to_dict(orient='records'): # df.to_json(new_filename, orient='records')
			fout.write(json.dumps(json_dict) + "\n")

def compute_link_change_rate(df):

	cols_in = ['diffInternalOutLinks+' + str(i+1) for i in range(9)]
	cols_ex = ['diffExternalOutLinks+' + str(i+1) for i in range(9)]
	print(df[cols_ex])

	def to_bool (x):
		if x > 0:
			return 1
		return 0

	for c in cols_in + cols_ex:
		df[c] = df[c].apply(to_bool)

	df['linkInternalChangeRate'] = (df[cols_in].sum(axis=1)) / len(cols_in)
	df['linkExternalChangeRate'] = (df[cols_ex].sum(axis=1)) / len(cols_ex)
	print(df['linkExternalChangeRate'])

	df = df.drop(cols_in + cols_ex, axis=1)

def json_to_pickled_df(filename_json, filename_pkl, nrows=None, processing_function=None):
	df = pd.read_json(filename_json, lines=True, nrows=nrows)

	if processing_function:
		processing_function(df)

	df.to_pickle(filename_pkl)

def df_from_pickle(filename_pkl):
	return pd.read_pickle(filename_pkl)

def read_dataset(filename, columns):
	# was: df = pd.read_json(filename, lines=True, nrows=nrows)
	df = df_from_pickle(filename)

	# print("\nOriginal dataset \n#rows:", len(df), "\ncolumns:\n", list(df.columns.values))

	# Some statistics
	# print(np.mean(df['fetchCount']), np.std(df['fetchCount']), min(df['fetchCount']), max(df['fetchCount']))
	# print(df['fetchMon'].value_counts())
	# print(df['fetchDay'].value_counts())

	df = df.filter(items=columns)
	# print("Raw dataset:", len(df), "rows, columns", df.columns.values)

	df = df.set_index("url")
	print("\nDataset \n\t# rows:", len(df), "\n\tcolumns:\n\t", list(df.columns.values))
	return df

def plot_univariate_distribution(data, bins, x_markers, xlabel, outfile):
	fig, ax = plt.subplots(figsize=(1.1 + bins*0.1, 1.25))
	sns_plot = sns.histplot(data, bins=bins, ax=ax, stat="probability", color="xkcd:slate grey", edgecolor="white", linewidth=1)

	plt.xticks(x_markers)
	plt.xlabel(xlabel)
	plt.ylabel('probability')

	sns_plot.get_figure().savefig(outfile, dpi=500, bbox_inches='tight')
	plt.clf()

def plot_categorical_distribution(data, classes, x_markers, x_marker_labels, xlabel, outfile):
	fig, ax = plt.subplots(figsize=(1 + classes*0.1, 1.8))
	sns_plot = sns.histplot(data, bins=classes, ax=ax, stat="probability", color="xkcd:grey", edgecolor="white", linewidth=1)

	bin_w = (max(x_markers) - min(x_markers)) / len(x_markers)
	plt.xticks(np.arange(min(x_markers)+bin_w/2, max(x_markers), bin_w), x_marker_labels)

	# upper bound hardcoded to keep it fixed across datasets
	plt.ylim((0, 1))
	plt.yticks(np.arange(0, 1.1, 0.1))

	words = xlabel.split(' ')
	plt.xlabel(' '.join(words[:-1]) + '\n' + words[-1])
	plt.ylabel('probability')

	sns_plot.get_figure().savefig(outfile, dpi=500, bbox_inches='tight')
	plt.clf()

def plot_bivariate_distributions(data, outfile):
	sns_plot = sns.pairplot(data, kind='hist', corner=True)

	sns_plot.savefig(outfile, bbox_inches='tight')
	plt.clf()

def corr_matrix(data, labels, method, outfile):
	correlations = data.corr(method=method) # method : {‘pearson’, ‘kendall’, ‘spearman’}

	num_features = len(data.columns.values)
	fig, ax = plt.subplots(figsize=(2.1+num_features/2, 1.1+num_features/4))
	mask = np.triu(np.ones_like(correlations, dtype=np.bool)) # masks the top half
	h = sns.heatmap(correlations, yticklabels=labels, mask=mask, square=False, cmap="RdYlGn", annot=True, fmt='.2f', linewidths=1, vmin=-1, vmax=1, cbar=False, ax=ax)
	h.set_xticklabels(labels, rotation=25, ha= 'right')

	plt.tight_layout(pad=0.5)

	fig.savefig(outfile, dpi=500, bbox_inches='tight')

	return correlations

"""
def plot_distribution(X, X_title, filename):
	plt.figure(figsize=(5, 3)) # in inches

	n, bins, patches = plt.hist(X, 20, weights=np.ones(len(X))/len(X), density=False, color='xkcd:faded blue')
	plt.xlabel(X_title)
	plt.ylabel('Prob. mass function')

	plt.tight_layout(pad=1)
	plt.savefig("data_distributions/"+filename)

def feature_dependence(X, N):
	from sklearn.ensemble import RandomForestRegressor
	D = feature_dependence_matrix(X, sort_by_dependence=True, n_samples=100000, 
		rfmodel=RandomForestRegressor(n_estimators=50, min_samples_leaf=0.01, oob_score=True))
	viz = plot_dependence_heatmap(D, figsize=(11,10))
	viz.save("feature_dependence/"+str(N)+"_fdep.pdf")
"""

if __name__ == '__main__':
	fname = "linkChangeRate_dataset"
	# flatten_json_to_json("../"+fname+".json", 
	# 					 1000000, 
	# 					 "../"+fname+"-SVflat.json")

	json_to_pickled_df("../"+fname+"-SVflat.json", 
						"datasets/"+fname+"-SVflat.pkl", 
						None, 
						compute_link_change_rate)

	# json_to_pickled_df("datasets/numNewOutlinks_dataset-between_09-07_and_09-14-SVflat-with_history_8.json", 
	# 	"datasets/numNewOutlinks_dataset-between_09-07_and_09-14-SVflat-with_history_8-sample.pkl", 10000)

