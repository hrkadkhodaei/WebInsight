from csv import reader
from sklearn.neighbors import NearestNeighbors
import numpy as np
import csv
from datetime import datetime
import logging

LOG_FORMAT = "%(levelname)s %(asctime)s in %(funcName)s line %(lineno)d- %(message)s"
logging.basicConfig(filename="log/log.log", level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

logger.critical("")
logger.critical("---------------------------------------------------------------------")


def related_pages_content_similarity(d_set, k, method='cosine'):
    start_time = datetime.now()
    if method == 'cosine':
        print("Start trainig KNN")
        logger.debug("start line NearestNeighbors")
        try:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean', n_jobs=-1).fit(d_set)
            logger.debug("start kneighbors")
            distance, ind = nbrs.kneighbors(d_set)
            logger.debug("finished kneighbors")
        except Exception as e:
            err = getattr(e, 'message', repr(e))
            logger.critical(err)
            print(err)
        end_time = datetime.now()
        print("Learning time: {}".format(end_time - start_time))
        ind = ind.tolist()
        logger.debug("finished ind")
    else:
        pass
        # knn1 = DistributedCosineKnn(k)
        # ind, distances = knn1.fit(input_data=semantic, n_bucket=8)
    index = [[x + 1 for x in arr] for arr in ind]
    degrees = [[float(1 / x) if x != 0 else float(np.finfo(np.float64).max / 100) for x in d] for d in distance]
    return index, distance, degrees


path = "dataset/"
fileName = "Semantic Vector.csv"
logger.debug("start reading file: " + fileName)
with open(path + fileName, 'r') as f:
    reader1 = reader(f)
    data = list(reader1)
logger.debug("file loaded into memory")
x3 = [list(map(float, lst)) for lst in data]
logger.debug("file converted to list[list[float]]")
print(len(x3))
logger.debug("dataframe is ready. size is: " + str(len(x3)))
logger.debug("start finding similarity pages")

indices, distances, degree = related_pages_content_similarity(x3, 30)

logger.debug("finished everything. degree has " + str(len(degree)) +
             " rows, indices: " + str(len(indices)) + ", distabces: " + str(len(distances)))

print("start writing indices")
logger.debug("wirting to file indices.txt")
try:
    with open(path + "indices.txt", "w+") as of1:
        write = csv.writer(of1)
        write.writerows(indices)
    print("wirting to file distances.txt")
    logger.debug("wirting to file distances.txt")
    with open(path + "distances.txt", "w+") as of2:
        write = csv.writer(of2)
        write.writerows(distances)
    print("wirting to file degree.txt")
    logger.debug("wirting to file degree.txt")
    with open(path + "degree.txt", "w+") as of3:
        write = csv.writer(of3)
        write.writerows(degree)
except Exception as e:
    err = getattr(e, 'message', repr(e))
    logger.critical(err)
    print(err)
print("finished successfully")
