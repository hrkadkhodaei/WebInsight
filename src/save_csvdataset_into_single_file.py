import findspark

findspark.init()
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, udf, concat_ws, row_number, lit, \
    collect_set  # , to_timestamp, lag, unix_timestamp, lit
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType

from re import sub
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from urllib.parse import urlparse
from time import strptime

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
import heapq
import numpy as np
from pyspark.sql.window import Window
from scipy.spatial import distance

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")


def getPathDepth(url):
    # /en/news/article123456/ has pathDepth 3
    path = urlparse(url).path
    # NB: urlparse("http://www.drive.google.com/en/news/article123456/") ParseResult(scheme='http',
    # netloc='www.drive.google.com', path='/en/news/article123456/', params='', query='', fragment='')
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


def getPageRank(struct):
    if not struct:
        return None
    for item in struct:
        if item['id'] == 'ElementValue':
            return item['value']
    return None


udf_getPageRank = udf(getPageRank, StringType())


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
    return len(out_url_array_external)


udf_numExternalOutLinks = udf(numExternalOutLinks, IntegerType())


def numInternalOutLinks(this_url, out_url_array_internal, out_url_array_external):
    if not out_url_array_internal:
        num_internal = 0
    else:
        num_internal = len(out_url_array_internal)
    return num_internal


udf_numInternalOutLinks = udf(numInternalOutLinks, IntegerType())

udf_set_difference = udf(lambda x, y: len(list(set(x) - set(y))), IntegerType())


def containsNotSet(line):
    if "not set" in line:
        return 1
    else:
        return 0


udf_set_notSet = udf(containsNotSet, IntegerType())


class DistributedCosineKnn:
    def __init__(self, k=3):
        self.k = k

    def fit(self, input_data, n_bucket=1):
        idxs = []
        dists = []
        buckets = np.array_split(input_data, n_bucket)
        for b in range(n_bucket):
            cosim = cosine_distances(buckets[b], input_data)
            idx0 = [(heapq.nsmallest((self.k + 1), range(len(i)), i.take)) for i in cosim]
            idxs.extend(idx0)
            dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
        return idxs, dists


class HelperFunctions:
    @staticmethod
    def set_difference(list1, list2):
        if list1 is None and list2 is None:
            return None
        if list1 is None:
            return None
        if list2 is None:
            return list1
        return list(set(list1).difference(list2))

    @staticmethod
    def cos_dist(list1, list2):
        dist = distance.cosine(list1, list2)
        return float(dist)

    @staticmethod
    def weighted_average(weights, amounts):
        res = np.average(amounts, weights=weights)
        return float(res)

    @staticmethod
    def create_internal_inlinks(d_set):
        temp1 = d_set.withColumn("internalLinks1", explode(d_set["internalLinks1"]))
        internal_inlinks1 = temp1.groupby('internalLinks1').agg(collect_set(temp1['url'])).toDF('url',
                                                                                                'internalInLinks1')
        temp2 = d_set.withColumn("internalLinks2", explode(d_set["internalLinks2"]))
        internal_inlinks2 = temp2.groupby('internalLinks2').agg(collect_set(temp2['url'])).toDF('url',
                                                                                                'internalInLinks2')
        joined = internal_inlinks1.join(internal_inlinks2, 'url', how='outer')
        return joined.withColumn('new_internal_inlinks',
                                 difference(joined['internalInLinks2'], joined['internalInLinks1']))

    @staticmethod
    def create_external_inlinks(d_set):
        temp1 = d_set.withColumn("externalLinks1", explode(d_set["externalLinks1"]))
        external_inlinks1 = temp1.groupby('externalLinks1').agg(collect_set(temp1['url'])).toDF('url',
                                                                                                'externalInLinks1')
        temp2 = d_set.withColumn("externalLinks2", explode(d_set["externalLinks2"]))
        external_inlinks2 = temp2.groupby('externalLinks2').agg(collect_set(temp2['url'])).toDF('url',
                                                                                                'externalInLinks2')
        joined = external_inlinks1.join(external_inlinks2, 'url', how='outer')
        return joined.withColumn('new_external_inlinks',
                                 difference(joined['externalInLinks2'], joined['externalInLinks1']))

    @staticmethod
    def remove_itself(el, list1):
        if el in list1:
            list1.remove(el)
        else:
            del list1[0]
        return list1

    @staticmethod
    def remove_degree(el, index_list, deg_list):
        if el in index_list:
            del deg_list[index_list.index(el)]
        else:
            del deg_list[0]
        return deg_list


rem_degree = udf(HelperFunctions.remove_degree, ArrayType(FloatType()))
rem_neighbor = udf(HelperFunctions.remove_itself, ArrayType(IntegerType()))
difference = udf(HelperFunctions.set_difference, ArrayType(StringType()))
w = Window().orderBy(lit('A'))

fileNameList = [r"/data/doina/WebInsight/2020-07-13/1M.2020-07-13-a" + chr(i) + ".gz" for i in range(97, 113)]

features = spark.read.json(fileNameList) \
    .where(
    (col('url').isNotNull()) &
    (col('fetch.contentLength') > 0) & ~col('fetch.semanticVector').contains('not set')) \
    .select(
    'url',
    'fetch.contentLength',
    'fetch.textSize',
    'fetch.textQuality',
    udf_numInternalOutLinks(col('url'), col('fetch.internalLinks'), col('fetch.externalLinks')).alias(
        'numInternalOutLinks'),
    udf_numExternalOutLinks(col('url'), col('fetch.externalLinks')).alias('numExternalOutLinks'),
    col('urlViewInfo.numInLinksInt').alias('numInternalInLinks'),
    col('urlViewInfo.numInLinksExt').alias('numExternalInLinks')) \
    .cache()

features = features.repartition(1)
path = 'dataset/'
features.write.json(path + 'ALL_DATA_0713_SINGLE_FILE.josn')

print("finished")
