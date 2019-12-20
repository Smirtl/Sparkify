# Sparkify - Customer Churn Prediction

## Description and Motivation
For the ficticious music streaming service Sparkify, we will use PySpark to analyze a small data set of customers and predict churn based on usage features of the service and cancelation of the service as a target variable.

## Data
The data (mini-set) is provided directly in the Udacity environment and will not be uploaded on GitHub.  

## Python Libraries
General Imports  
* `import numpy as np`  
* `import pandas as pd`  
* `import matplotlib.pyplot as plt`  
* `import seaborn as sns`
* `import datetime`  

PySpark SQL Imports  
* `from pyspark.sql import SparkSession`  
* `from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, avg`  
* `from pyspark.sql.functions import concat, explode, lit, min, max, split, datediff, to_date, countDistinct`  
* `from pyspark.sql.functions import sum as Fsum`  
* `from pyspark.sql.window import Window`  
* `from pyspark.sql.types import IntegerType, DateType`  

PySpark Machine Learning Imports  
* `from pyspark.ml import Pipeline`  
* `from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, DecisionTreeClassificationModel`  
* `from pyspark.ml.classification import RandomForestClassificationModel, RandomForestClassifier, NaiveBayes, GBTClassifier`  
* `from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator`  
* `from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, RegexTokenizer, StandardScaler, StopWordsRemover`  
* `from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer, StandardScaler, Normalizer, MinMaxScaler`  
* `from pyspark.ml.regression import LinearRegression`  
* `from pyspark.ml.tuning import CrossValidator, ParamGridBuilder`  
* `from pyspark.mllib.evaluation import MulticlassMetrics`  
* `from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score`

## Files
* `Sparkify.ipynb` - Jupyter Notebook for with exploratory analysis and modeling  
* `README.md` - This readme file  

## Result Summary
The results can be viewed in this summary blog post on MEDIUM:  
https://medium.com/@smirtl/wouldnt-you-want-to-know-in-advance-if-a-subscriber-is-about-to-leave-you-729e06c763db
