#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis Of Amazon Food Reviews

# # Import Libraries

# In[1]:


import re
import nltk
import string
import operator
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict, Counter
matplotlib.style.use('ggplot')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.ml.feature import NGram
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel,LogisticRegressionWithSGD


# # Prepare Dataset

# In[2]:


sqlContext = SQLContext(SparkContext.getOrCreate())
#convert data csv file to text file
data = SparkContext.getOrCreate()
sqlContext = SQLContext(data)
# convert to spark sql dataframe
data = data.textFile('Reviews.txt')
data.take(5)


# In[3]:


#Count total number of recods
data.count()


# # Converting to spark sql dataframe

# In[4]:


data = data.map(lambda x: x.split('\t')).toDF()  
data = data.selectExpr("_1 as Id", "_2 as ProductId", "_3 as UserId", "_4 as ProfileName", "_5 as HelpfulnessNumerator", 
                       "_6 as HelpfulnessDenominator","_7 as Score","_8 as Time","_9 as Summary","_10 as Text")
data.printSchema()


# In[5]:


data.head(5)


# # Labeling Records

# In[6]:


filter_data = data.filter((data.Score == "1") | (data.Score=="5")).select('Score','Text')
filter_data.show(10)


# In[7]:


filter_data.count()


# # Positivity Plot

# In[8]:


dataDF = pd.read_csv('Reviews.csv')
dataDF['Positivity'] = np.where(dataDF['Score'] > 3, 1, 0)
dataDF.head()
sns.countplot(dataDF['Positivity'])
plt.show()


# # Negativity Plot

# In[9]:


dataDF['Negativity'] = np.where(dataDF['Score'] < 3, 0, 1)
dataDF.head()
sns.countplot(dataDF['Negativity'])
plt.show()


# # Lowercase The Text

# In[10]:


def lower_text(line):
    word_list=re.findall('[\w_]+', line.lower())
    return ' '.join(map(str, word_list))

filter_data_withColumn = filter_data.withColumn("text_lower", udf(lower_text, StringType())("Text")).select('text_lower','Score')

#Showing the result
filter_data_withColumn.show(15)


# # Tokenize

# In[11]:


tokenize = Tokenizer(inputCol="text_lower", outputCol="words")
words_Data_Frame = tokenize.transform(filter_data_withColumn)
words_Data_Frame.take(5)


# # Remove Stopword

# In[12]:


remove = StopWordsRemover(inputCol="words", outputCol="filtered_words")
words_Data_Frame1 = remove.transform(words_Data_Frame).select("filtered_words","Score")
words_Data_Frame1.show(5)


# # Stemming

# In[13]:


def stem_tokens(tokens):
    return [PorterStemmer().stem(item) for item in tokens]


# In[14]:


def stem_text(tokens):
    return ' '.join(stem_tokens(tokens))


# In[15]:


words_Data_Frame2 = words_Data_Frame1.withColumn("final_text", udf(stem_text, StringType())("filtered_words")).select('final_text','Score')


# In[16]:


words_Data_Frame2.cache()
training = words_Data_Frame2.selectExpr("final_text as text", "Score as label")
training = training.withColumn("label", training["label"].cast(DoubleType()))
training.take(5)


# # Logistic Regression Model

# In[13]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="hashing")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])


# # Train The Logistic Regression Model

# In[21]:


model = pipeline.fit(training)
model.transform(training).printSchema()


# In[ ]:


print("Percentage of error rate is: {0}".format(model.transform(training).rdd.map(lambda line: abs(line[1] - line[7])).reduce(lambda x,y:x+y) 
                                  // float(model.transform(training).count())))


# # Decision Tree Classifier Model

# In[ ]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="hashing")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
dt = DecisionTreeClassifier(maxDepth=2)
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, dt])


# # Train The Decision Tree Classifier Model

# In[ ]:


paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [2,3]).build()
Cross_Validator = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=5)
model = Cross_Validator.fit(training)
model.avgMetrics


# # Word Cloud Data

# In[17]:


word_cloud_data = words_Data_Frame1.rdd.map(lambda x: (x[1],x[0])).toDF()
word_cloud_data = word_cloud_data.selectExpr("_1 as Score","_2 as word")
word_cloud_data.show(5)
word_cloud_data.createOrReplaceTempView("words")


# # Word Cloud Data For Positive Words

# In[18]:


positive_words_list = sqlContext.sql(" SELECT word from words where Score = 5 ").take(1000)
positive_words=''
positive_feats=defaultdict(lambda:0)
for each in positive_words_list:
    for eachwords in each[0]:
        if eachwords.isalpha() and eachwords!='br':
            positive_words+=eachwords+' ' 
            positive_feats[eachwords]+=1
        

sorted_positive_feats = dict(sorted(positive_feats.items(), key=operator.itemgetter(1),reverse=True))
sorted_positive_feats


# # Word Cloud Data For Positive Reviews

# In[19]:


positive_image = np.array(Image.open("bdPos.jpg"))
positive_wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black', width=800, height=650,mask=positive_image,).generate(positive_words)
plt.imshow(positive_wordcloud)
plt.axis('off')
plt.show()


# # Word Cloud Data For Negative Words

# In[20]:


negative_words_list = sqlContext.sql(" SELECT word from words where Score = 5 ").take(1000)
negative_words=''
negative_feats=defaultdict(lambda:0)
for each in negative_words_list:
    for eachwords in each[0]:
        if eachwords.isalpha() and eachwords!='br':
            negative_words+=eachwords+' ' 
            negative_feats[eachwords]+=1
        

sorted_negative_feats = dict(sorted(negative_feats.items(), key=operator.itemgetter(1),reverse=True))
sorted_negative_feats


# In[21]:


negative_image = np.array(Image.open("bdNeg.jpg"))
negative_wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black', width=800, height=650,mask=negative_image,).generate(negative_words)
plt.imshow(negative_wordcloud)
plt.axis('off')
plt.show()


# # Positive words count

# In[23]:


positive_word_count = 0
selected_positive_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'loves','like','wonderful','tasty','fresh','organic','happy'])
for word in positive_feats.keys():
    if word in selected_positive_words:
        positive_word_count+=1
print("Positive words count=",positive_feats['great'])


# # Negative words count 

# In[25]:


negative_word_count=0
selected_negative_words = set(['bad', 'hate', 'horrible', 'terrible','worst', 'dislike','disappointed','disappointing','never','waste','awful'])
for word in negative_feats.keys():
    if word in selected_negative_words:
        negative_word_count+=1

print("Negative words count = ", negative_feats['bad'])

