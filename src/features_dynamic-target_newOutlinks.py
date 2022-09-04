from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf

from re import sub
from urlparse import urlparse
from time import strptime

"""
Executed with:

	time spark-submit --master yarn --deploy-mode cluster --executor-cores 2 --conf spark.dynamicAllocation.minExecutors=25 --conf spark.dynamicAllocation.initialExecutors=25 --driver-memory 6G --executor-memory 6G features_dynamic-target_newOutlinks.py
	(will use more than 25 executors at read and write time)
	(runtime: 3 minutes for the longest history)

The parts in the output can be merged with:

	hdfs dfs -text WebInsight/numNewOutlinks_dataset-between_09-07_and_09-14-SVflat-with_history_8/part*.json | hdfs dfs -put - WebInsight/numNewOutlinks_dataset-between_09-07_and_09-14-SVflat-with_history_8.json
"""

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# crawl_present is a dataset containing page features at Date1, and the target variable:
#     the diff***OutLinks fields computed between Date1 and Date2 (Date2 > Date1).
# crawl_past is 1+ dataset(s) containing similar data from previous days, from which 
#     only some fields will be selected 

crawl_present = "between_09-07_and_09-14-SVflat"
crawl_past = ["between_09-01_and_09-07-SVflat", 
              "between_08-27_and_09-01-SVflat",
              "between_08-18_and_08-27-SVflat",
              "between_08-11_and_08-18-SVflat",
              "between_08-07_and_08-11-SVflat",
              "between_07-28_and_08-07-SVflat",
              "between_07-21_and_07-28-SVflat",
              "between_07-13_and_07-21-SVflat"] # in decreasing date order, so most recent first!

# From crawl_present, keep everything. These are the static page features, plus the target variable.

dataset = spark.read.json("WebInsight/numNewOutlinks_dataset-" + crawl_present + ".json") \
	.cache()

# From crawl_past datasets, keep the 'diff***OutLinks' fields (the same fields as the target variable,
#    but from the past, to be used as historical features). Keep the other static page and network features
#    that are likely to change in time (so, not pathDepth, domainDepth; skip the semantic vector). 
#    These are the dynamic features.

for i in range(len(crawl_past)):

	dynamic_features = spark.read.json("WebInsight/numNewOutlinks_dataset-" + crawl_past[i] + ".json") \
		.select(
			'url', 
			col('diffExternalOutLinks').alias('diffExternalOutLinks-'+str(i+1)),
			col('diffInternalOutLinks').alias('diffInternalOutLinks-'+str(i+1)),
			col('contentLength').alias('contentLength-'+str(i+1)),
			col('textSize').alias('textSize-'+str(i+1)),
			col('textQuality').alias('textQuality-'+str(i+1)),
			col('numInternalInLinks').alias('numInternalInLinks-'+str(i+1)),
			col('numExternalInLinks').alias('numExternalInLinks-'+str(i+1)),
			col('trustRank').alias('trustRank-'+str(i+1))) \
		.cache()

	dataset = dataset \
		.join(dynamic_features, on='url', how='inner')

dataset.write.json("dataset/numNewOutlinks_dataset-" + crawl_present + "-with_history_" + str(len(crawl_past)), 
	mode="overwrite")
