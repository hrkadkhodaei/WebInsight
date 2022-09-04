from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, regexp_replace, udf  # , to_timestamp, lag, unix_timestamp, lit
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType

from re import sub
from urlparse import urlparse
from time import strptime

"""
To ensure a large minimum number of executors and a large memory-to-core ratio:

	time spark-submit --master yarn --deploy-mode cluster --executor-cores 2 --conf spark.dynamicAllocation.minExecutors=25 --conf spark.dynamicAllocation.initialExecutors=25 --driver-memory 6G --executor-memory 6G features_static-target_numNewOutlinks.py 

The parts in the output can be merged with:

	hdfs dfs -text WebInsight/numNewOutlinks_dataset-between_07-13_and_07-21/part*.json | hdfs dfs -put - WebInsight/numNewOutlinks_dataset-between_07-13_and_07-21.json
"""

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# integer tuples (month, day, pretty_string)
crawl_list = [(7, 13, "07-13"), (7, 21, "07-21"), (7, 28, "07-28"), (8, 7, "08-07"), (8, 11, "08-11"),
              (8, 18, "08-18"), (8, 27, "08-27"), (9, 1, "09-01"), (9, 7, "09-07"), (9, 14, "09-14")]
crawl_initial, crawl_final = 5,6  # indices in crawl_list, 0..9


def getPathDepth(url):
    # /en/news/article123456/ has pathDepth 3
    path = urlparse(url).path
    # NB: urlparse("http://www.drive.google.com/en/news/article123456/")
    #     ParseResult(scheme='http', netloc='www.drive.google.com', path='/en/news/article123456/', params='', query='', fragment='')
    path = sub("/$", "", path)  # $ matches the end of line
    return path.count("/")


udf_getPathDepth = udf(getPathDepth, IntegerType())


def getDomainDepth(url):
    # drive.google.com has domainDepth 3
    netloc = sub(r"^www.", "", urlparse(url).netloc)
    return len(netloc.split("."))


udf_getDomainDepth = udf(getDomainDepth, IntegerType())


def getTrustRank(struct):
    if not struct:
        return None
    for item in struct:
        if item['id'] == 'TrustValue':
            return item['value']
    return None


udf_getTrustRank = udf(getTrustRank, StringType())


def reduce_to_mon(string_date):
    ts = strptime(string_date, "%Y-%m-%d %H:%M")
    return ts.tm_mon


def reduce_to_day(string_date):
    ts = strptime(string_date, "%Y-%m-%d %H:%M")
    return ts.tm_mday


udf_mon = udf(reduce_to_mon, IntegerType())
udf_day = udf(reduce_to_day, IntegerType())


def numExternalOutLinks(this_url, out_url_array_external):
    if not out_url_array_external:
        return 0
    #
    # This corrects any misclassified instances; it isn't crucial to do though
    # corrected_out_url_array = []
    # this_netloc = sub(r"^www.", "", urlparse(this_url).netloc)
    # for u in out_url_array_external:
    # 	out_netloc = sub(r"^www.", "", urlparse(u['targetUrl']).netloc)
    # 	if out_netloc != this_netloc:
    # 		corrected_out_url_array.append(u['targetUrl'])
    # return len(corrected_out_url_array)
    #
    return len(out_url_array_external)


udf_numExternalOutLinks = udf(numExternalOutLinks, IntegerType())


def numInternalOutLinks(this_url, out_url_array_internal, out_url_array_external):
    if not out_url_array_internal:
        num_internal = 0
    else:
        num_internal = len(out_url_array_internal)
    #
    # This corrects any misclassified instances; it isn't crucial to do though
    # if not out_url_array_external:
    # 	return num_internal
    # extra_out_url_array = []
    # this_netloc = sub(r"^www.", "", urlparse(this_url).netloc)
    # for u in out_url_array_external:
    # 	out_netloc = sub(r"^www.", "", urlparse(u['targetUrl']).netloc)
    # 	if out_netloc == this_netloc:
    # 		extra_out_url_array.append(u['targetUrl'])
    # return len(extra_out_url_array) + num_internal
    #
    return num_internal


udf_numInternalOutLinks = udf(numInternalOutLinks, IntegerType())

udf_set_difference = udf(lambda x, y: len(list(set(x) - set(y))), IntegerType())

# From the first crawl, simply grab the static page and static network features.
# 	For the (later) computation of the target variable, also add here the two sets of outlinks (internal and external).
# 	These caches will occupy memory, because the link sets are quite some data.

features = spark.read.json("/data/doina/WebInsight/2020-" + crawl_list[crawl_initial][2] + "/*.gz") \
    .where(
    (col('url').isNotNull()) &
    (col('fetch.contentLength') > 0)) \
    .select(
    'url',
    udf_getPathDepth(col('url')).alias('pathDepth'),
    udf_getDomainDepth(col('url')).alias('domainDepth'),
    'fetch.language',
    'fetch.contentLength',
    'fetch.textSize',
    'fetch.textQuality',
    'fetch.semanticVector',
    udf_numInternalOutLinks(col('url'), col('fetch.internalLinks'), col('fetch.externalLinks')).alias(
        'numInternalOutLinks'),
    udf_numExternalOutLinks(col('url'), col('fetch.externalLinks')).alias('numExternalOutLinks'),
    col('urlViewInfo.numInLinksInt').alias('numInternalInLinks'),
    col('urlViewInfo.numInLinksExt').alias('numExternalInLinks'),
    udf_getTrustRank(col('urlViewInfo.metrics.entries')[0]['metrics']['entries']).alias('trustRank'),
    col('fetch.externalLinks.targetUrl').alias('initialSetExternalOutLinks'),
    col('fetch.internalLinks.targetUrl').alias('initialSetInternalOutLinks')) \
    .cache()

# From the final crawl, grab the two sets of outlinks (internal and external).
argets = spark.read.json("/data/doina/WebInsight/2020-" + crawl_list[crawl_final][2] + "/*.gz") \
    .where(
    (col('url').isNotNull()) &
    (col('fetch.contentLength') > 0)) \
    .select(
    'url',
    col('fetch.externalLinks.targetUrl').alias('finalSetExternalOutLinks'),
    col('fetch.internalLinks.targetUrl').alias('finalSetInternalOutLinks')) \
    .cache()

# Compute the difference between the sets of outlinks, and save the size of the difference (# new outlinks).
# This join should be cheap in memory: one assumes there's only one instance of each URL in each crawl.
# Runtime for the first 2 crawls: 3 minutes w. 25 executors; the resulting file is 1.3 GB uncompressed, with 742492 pages.

dataset = features \
    .join(targets, on='url', how='inner') \
    .withColumn('diffExternalOutLinks', udf_set_difference('finalSetExternalOutLinks', 'initialSetExternalOutLinks')) \
    .drop('initialSetExternalOutLinks') \
    .drop('finalSetExternalOutLinks') \
    .withColumn('diffInternalOutLinks', udf_set_difference('finalSetInternalOutLinks', 'initialSetInternalOutLinks')) \
    .drop('initialSetInternalOutLinks') \
    .drop('finalSetInternalOutLinks') \
    .write.option("header", "true").csv(
    "dataset/numNewOutlinks_dataset-between_" + crawl_list[crawl_initial][2] + "_and_" + crawl_list[crawl_final][2]+".csv",
    mode='overwrite')
