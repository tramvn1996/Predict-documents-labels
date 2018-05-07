from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.rdd import RDD
from pyspark.sql.functions import col, countDistinct

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

fnames = sc.textFile("file:///home/tnguyen/BigData/project5/cinf401-finalproject/trainsmall.csv")
#fnames = sc.textFile("file:///home/tnguyen/CREU/CREU/HedgeDetection/unzipped_data/newsexport/metadata.csv")
m=fnames.map(lambda line: line.split(","))
header = m.first()
m = m.filter(lambda line: line!=header)

def getRow(rows):
    s=[]
    if rows is not None:
        s.append((rows[1],rows[2],rows[3],rows[4],rows[5],rows[6]))
    return s

m2 = m.flatMap(getRow)
df  = m2.toDF(['CDID','Date','Displ','Genre','Industry','Category'])
df = df.toPandas()
df =df.drop_duplicates(subset=['CDID'],keep='first')
y=df['Category']
#df.groupBy('Category').count().show()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

#CountVectorizer
vectorizer = CountVectorizer(analyzer="word", stop_words="english",ngram_range=(1,2),lowercase=True)

#tfidf 
print("extracting td-idf features")
tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df =2, max_features=1000, stop_words = 'english')



def getData(rows):
    article =[]
    for row in rows:
        name = "/home/tnguyen/CREU/CREU/HedgeDetection/parse_data_articles/fulltext/" +row+'.txt'
        #print(name)
        with open(name,'r') as myfile:
            if os.stat(name).st_size!=0:
                article.append(myfile.read())
    return article

train_data_feature = vectorizer.fit_transform(getData(X_train['CDID']))
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_feature, y_train)

tfidf =tfidf_vectorizer.fit_transform(getData(X_train['CDID']))

import pickle
filename = 'content_classification.sav'
pickle.dump(forest, open(filename,'wb'))
from sklearn.externals import joblib
joblib.dump(vectorizer, 'vocab.pkl')
joblib.dump(forest, 'randomeforest.pkl')

test_data_features = vectorizer.transform(getData(X_test['CDID']))
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
#output = pd.DataFrame(data={"text":X_test["CDID"],"original":X_test["Category"],"content":result})
#output.to_csv("no.csv", index=False, encoding="latin-1")
forest2 = RandomForestClassifier(n_estimators =100)
forest2 = forest2.fit(tfidf, y_train)
test2= tfidf_vectorizer.transform(getData(X_test['CDID']))
test2 = test2.toarray()
result2 = forest2.predict(test2)

accuracy = accuracy_score(y_test,result)
print(accuracy)

accuracy2 = accuracy_score(y_test, result2)
print("TFIDF")
print(accuracy2)
# Perform 3-fold cross validation
#scores = cross_val_score(forest, df, y, cv=3)
#print("Cross-validated score:" + scores)

