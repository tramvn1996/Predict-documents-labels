from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.rdd import RDD
from pyspark.sql.functions import col, countDistinct
from pyspark.sql.functions import desc

from pyspark.sql import functions as F
import time

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
import os
import sys

start_time = time.time()

conf = SparkConf().setAppName("train text classification")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName("Classification").getOrCreate()


#Group the same categories together
fnames = sc.textFile("file:///home/tnguyen/BigData/cinf401-project5/splitted.csv")
m=fnames.map(lambda line: line.split(','))
header = fnames.first()
rows = m.filter(lambda line: line!=header)
def getRow(rows):
    s=[]
    if rows is not None:
        if rows!=header:
            s.append((rows[1],rows[6]))
    return s
m2=rows.flatMap(getRow)
df = m2.toDF(['CDID','Category'])
count = df.groupby('Category').count()
top = count.sort(desc("count"))
top.toPandas().to_csv("top100short.csv")

#Split the category column
#fnames = sc.textFile("file:///home/tnguyen/BigData/cinf401-project5/trainingset.csv")
#m = fnames.map(lambda line: line.split(','))
#header = fnames.first()
#print("HEADER"+header)
#rows = m.filter(lambda line: line!=header)
#print("NEW")
#h= rows.first()

#def getCate(rows):
#    s=[]
#    if rows is not None:
#        if rows!=h:
#            sl = rows[6].count('/')
#            rows6 = rows[6].rsplit('/',sl)[1]
#            print(sl)
#            print(rows6)
#            s.append((rows[1],rows[2],rows[3],rows[4],rows[5],rows6))
#    return s



#m2 = rows.flatMap(getCate)
#df = m2.toDF(['CDID','Date','Displ','Genre','Industry','Category'])
#df.show()
#df.toPandas().to_csv("splitted.csv")


#Get top 100 of all the categories

#fnames = sc.textFile("file:///home/tnguyen/BigData/cinf401-project5/fullset.csv")
#def getRow(rows):
#    s=[]
#    if rows is not None:
#        s.append((rows[1],int(rows[2])))
#    return s

#m=fnames.map(lambda line: line.split(","))
#header = m.first()
#m = m.filter(lambda line: line!=header)
#m2 = m.flatMap(getRow)
#df  = m2.toDF(['Cate','Count'])



#top10 = m2.sortBy(lambda x: x[1],ascending=False)
#top10=sc.parallelize(top10.take(100))
#
