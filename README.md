# Movie_Review_Prediction
In the dataset each record represents a movie-review pair with movie title, description, genres, duration, director, actors, users' ratings, review text, reviewer name, etc. We build various ML model to predict sentiment of the review text.

# Libraries Used :
- pandas
- numpy
- scikitlearn
- matplotlib
- re


# About the Dataset:
 There are three csv files named; movies; train and test datasets that contain the various columns we need to work on for predicting the sentiment of a movie based on various parameters. 
 
 Follow the link below to know more about the dataset:  
  https://www.kaggle.com/competitions/sentiment-prediction-on-movie-reviews/data 

 # Pre-processing steps:
 - Removed  duplicates
 - removed null values
 - merged train and movies dataset to have a bigger sample of data for training purposes.
 - pre-processed the **reviewText** column by removing stopwords, URLs, HTML tags, numbers or apostrophes using regex to have a better quality of text corpora to work with and for faster convergence.
 - Imputed missing values in columns which had <30% missing data in various columns.

# Feature Engineering steps:
- Scaled the numerical columns in the dataset using MinMaxScaler.
- Vectorized the text using **TF-IDFVectorizer**
