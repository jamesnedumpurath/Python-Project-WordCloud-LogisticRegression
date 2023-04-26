#Regression
#Importing necessary Libraries
import random
import string
import re as re
import random as ran

import plotly
import matplotlib
import pandas as pd
import nltk as nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

#To avoid the truncation problem while importing csv files by showing all the records

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Load csv files into  into reviews and text_details.

reviews =  pd.read_csv("D:\\TSoM-SQL Python\\Python Project\\Books_rating.csv")
text_details = pd.read_csv("D:\\TSoM-SQL Python\\Python Project\\books_data-1.csv")

#printing the top 5 rows to get a look at data and analyse

print(reviews.head())
print(text_details.head())

#Joining both reviews and Text_deatils together to form a combined dataframe merged_df using inner join and with tittle

merged_df = text_details.merge(reviews, how = 'inner',on='Title')

#printing the merged_df top 5 rows only

print(merged_df.head())

#Treat missing values of the variables that may be used in the prediction model.
#using dropna function the NA's in these three columns "review/score","review/summary","review/text"
#was taken out. these are the columns which matter to thhe analysis

merged_df.dropna(subset=["review/score","review/summary","review/text"], inplace= True)

#checking the total number of rows after the Na removal
print("Number of rows=",len(merged_df))
print()

#Select only the 'Title','categories','review/score','review/summary','review/text' variables that would matter to the
#analysis. Creating a new DataFrame called merged_df_sub and renaming the column names for clarity

merged_df_sub = merged_df[['Title','categories','review/score','review/summary','review/text']]
merged_df_sub.columns = ['Title','categories','review_score','review_summary','review_text']


#Data Cleaning
#A function is created which can take one text argument and return a string which can
#remove all the punctuations and stop words from the given data.
#that can also perform duplicate handling

def remove_punc_stopwords(text):
    text = re.sub(f"[{string.punctuation}]"," ",text)
    text_tokens = set(text.split())
    stops = set(stopwords.words('english'))
    text_tokens = text_tokens.difference(stops)
    return " ".join(text_tokens)

#Removing punctuations and stopwords from the text data in review_summary
merged_df_sub['review_summary'] = merged_df_sub['review_summary'].apply(remove_punc_stopwords)

#Adding a new column called sentiment
#If review_score is greater than 3, then sentiment = 1, else sentiment = -1
#This is to categorize the rating positive and negative.
#recusrsive list is used to perform this task.
merged_df_sub['sentiment'] = [1 if x>3 else -1 for x in merged_df_sub.review_score]

#spliting the dataset into two data sets to perform and evaluvate the model
## a  train data set with 85% of the obs. and a test data set with 15% of the obs.
#to perform this a new column named random index is added and that is calculated using the
#uniform function
merged_df_sub['random_index'] = [ random.uniform(0,1) for x in range(len(merged_df_sub))]
#the test and train data frames were found by cheking the random index
merged_df_sub_train = merged_df_sub[merged_df_sub.random_index < 0.85]
merged_df_sub_test = merged_df_sub[merged_df_sub.random_index >= 0.85]


#Printing the top 5 rows of both train and test set.
print(merged_df_sub_train.head())
print(merged_df_sub_test.head())

#count vectorizer:

#count vectorizer from the Scikit-learn library is used to transform the text in our data frame into a bag of words
#which will contain a sparse matrix of integers. Each word will be counted and the number of occurances will be printed.
#For logistic regression algoritm we will need bag of words mode.

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
#both test and train dataframes review_summary is vectorized and stored in in their respective variable matrices.
train_matrix = vectorizer.fit_transform(merged_df_sub_train['review_summary'])
test_matrix = vectorizer.transform(merged_df_sub_test['review_summary'])

#Performing Logistic Regression
#assign the logistic regreesion function to lr.


lr = LogisticRegression()
#X and Y components of regression is defined

X_train = train_matrix
X_test = test_matrix
y_train = merged_df_sub_train['sentiment']
y_test = merged_df_sub_test['sentiment']

#Performing the model for sentiment
lr.fit(X_train,y_train)

#Generating the predictions for the test dataset
predictions = lr.predict(X_test)
merged_df_sub_test['predictions'] = predictions
print(merged_df_sub_test.head(30))

#Calculating the prediction accuracy
#Created a new column match with true or false values by comparing each rows senitment with prediction
merged_df_sub_test['match'] = merged_df_sub_test['sentiment'] == merged_df_sub_test['predictions']
#printing the percentage of accuracy of the prediction
#sum of match divided by the total number of rows
print(sum(merged_df_sub_test['match'])/len(merged_df_sub_test))


#performing the presdictions for review_score
lr2 = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
y_train2 = merged_df_sub_train['review_score']
y_test2 = merged_df_sub_test['review_score']


lr2.fit(X_train,y_train2)

#Generate the predictions for the test dataset
predictions2 = lr2.predict(X_test)
merged_df_sub_test['predictions_rating'] = predictions2
print(merged_df_sub_test.head(30))


#Calculating the prediction accuracy to ensure the model worked.
#Created a new column match_rating with true or false values by comparing each rows review score with prediction rating
merged_df_sub_test['match_rating'] = merged_df_sub_test['review_score'] == merged_df_sub_test['predictions_rating']
#printing the percentage of accuracy of the prediction
#sum of match rating divided by the total number of rows
print(sum(merged_df_sub_test['match_rating'])/len(merged_df_sub_test))