from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.rdd import RDD
from pyspark.sql.functions import col, countDistinct

from pyspark.sql import functions as F
import time

from google.cloud import language_v1beta2
from google.cloud.language_v1beta2 import enums
from google.cloud.language_v1beta2 import types
import os

start_time = time.time()

conf = SparkConf().setAppName("Classify text based on content")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName("Classification").getOrCreate()

#fnames = sc.textFile("file:///home/tnguyen/BigData/project5/small.csv")
fnames = sc.textFile("file:///home/tnguyen/CREU/CREU/HedgeDetection/unzipped_data/newsexport/metadata.csv")
m=fnames.map(lambda line: line.split(","))

header = m.first()
m = m.filter(lambda line: line!=header)
#print(m.collect())
def getCDID(rows):
    s=[]
    language_client = language_v1beta2.LanguageServiceClient()

    def getTag(content_input):
        document = types.Document(content=content_input, type=enums.Document.Type.PLAIN_TEXT)
        result = language_client.classify_text(document)
        return result

    if rows is not None:
        name = "/home/tnguyen/CREU/CREU/HedgeDetection/parse_data_articles/fulltext/"+rows[0]+'.txt'
        print(name)
        st = ''
        conf=''
        print(name)
        with open(name,'r') as myfile:
            if os.stat(name).st_size!=0:
                article = myfile.read()
                try:
                    results = getTag(article)
                    if results is not None:
                        for result in  results.categories:
                            if result is not None:
                                st = result.name
                                print(st)
                                s.append((rows[0],rows[1],rows[2],rows[3], rows[4], st))
                except:
                    print("too few words")
            else:
                s.append((rows[0],rows[1],rows[2],rows[3], rows[4], 0))
    return s

m2 = m.flatMap(getCDID)
m2 = m2.filter(lambda line: line[5]!=0)
df = m2.toDF(['CDID','Date','Displ','Genre','Industry','Cateogory'])
#df.show()
df.toPandas().to_csv("trainingset.csv")
df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).show()

