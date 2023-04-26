#importing necessary libraries
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

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


#printing the merged_df top 5 rows only to understand the structure

print(merged_df.head())

#print the summary of columns to understand the data
print(merged_df.describe(include='all'))

#Printing the number of rows in the data set
print("Number of Rows=",len(merged_df))
print()

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

#spliting the dataset into two data sets to perform and evaluvate the model
# a  train data set with 85% of the obs. and a test data set with 15% of the obs.
#sample function is used with .85 and .15 fraction to split the data frame
merged_df_sub_train = merged_df_sub.sample(frac =.85)
print(merged_df_sub_train.head(10))
merged_df_sub_test = merged_df_sub.sample(frac =.15)
print(merged_df_sub_test.head(10))

#Adding a new column called sentiment
#If review_score is greater than 3, then sentiment is positive else negative
#This is to categorize the rating positive and negative.
#recusrsive list is used to perform this task.
merged_df_sub_train['sentiment'] = ['positive' if x > 3 else 'negative' for x in merged_df_sub_train.review_score]
print(merged_df_sub_train.head(10))

#Histogram is plotted using the review_score.
import plotly.express as px
fig = px.histogram(merged_df, x= "review/score")
fig.show()

#Importing nltk and stopwords
import nltk

#Create list stop words to filter the review summary
stops= set(STOPWORDS)

#Combine all the review summary into one string so that the word cloud can be performed.
#All the review summary is joined together with a space character
textt = " ".join(x for x in merged_df_sub_train.review_summary)
print(textt)


import matplotlib.pyplot as plt

#creating a word cloud of the review summary
wordcloud = WordCloud(stopwords=stops).generate(textt)
#saving the word cloud to computer drive
wordcloud.to_file("D:\\first_review.png")

#creating a word cloud of the positive reviews
#subsetting the merged_df_sub_train to contain only the positive reviews
merged_df_sub_train_p = merged_df_sub_train[merged_df_sub_train.sentiment == 'positive']
#Combine all the review summary into one string so that the word cloud can be performed
textt_p = " ".join(x for x in merged_df_sub_train_p.review_summary)
wordcloud = WordCloud(stopwords=stops).generate(textt_p)
#saving the word cloud to computer drive
wordcloud.to_file("D:\\first_review1_p.png")


#creating a word cloud of the negative reviews
#subsetting the merged_df_sub_train to contain only the negative reviews
merged_df_sub_train_n = merged_df_sub_train[merged_df_sub_train.sentiment == 'negative']
#Combine all the review summary into one string so that the word cloud can be performed
textt_n = " ".join(x for x in merged_df_sub_train_n.review_summary)
wordcloud = WordCloud(stopwords=stops).generate(textt_n)
#saving the word cloud to computer drive
wordcloud.to_file("D:\\first_review2_n.png")