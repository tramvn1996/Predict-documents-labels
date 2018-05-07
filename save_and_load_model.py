


import pandas as pd
from sklearn.externals import joblib
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os



forest1 = joblib.load('randomeforest.pkl') 
vectorizer = joblib.load('vocab.pkl')



with open(sys.argv[1], 'r') as myfile:
    if os.stat(sys.argv[1]).st_size != 0:
        article=[]
        article.append(myfile.read())
        article_data_feature = vectorizer.transform(article)
        article_data_feature = article_data_feature.toarray()
        result = forest1.predict(article_data_feature)
        print(result)

# In[ ]:



