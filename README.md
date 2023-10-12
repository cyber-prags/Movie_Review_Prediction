# Movie Review Sentiment Prediction

![Movie Review]([link_to_image.jpg](https://nkpremices.com/content/images/2021/08/mih10uhu1464fx1kr0by-1.jpg))

This project aims to predict the sentiment of movie reviews using machine learning models. The dataset consists of movie-title, description, genres, duration, director, actors, users' ratings, review text, reviewer name, and more. We use various machine learning models to analyze and predict the sentiment of movie reviews.

## Libraries Used

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [re](https://docs.python.org/3/library/re.html)

## About the Dataset

The dataset is comprised of three CSV files: `movies`, `train`, and `test`. These files contain a variety of columns that we leverage to predict the sentiment of movie reviews. For a detailed understanding of the dataset, you can visit the [Kaggle competition page](https://www.kaggle.com/competitions/sentiment-prediction-on-movie-reviews/data).

## Pre-processing Steps

1. **Removing Duplicates:** Duplicate records are removed to ensure data quality.

2. **Handling Null Values:** Null values in the dataset are handled appropriately to avoid issues during modeling.

3. **Data Merging:** The `train` and `movies` datasets are merged to create a larger, comprehensive dataset for training purposes.

4. **Text Pre-processing:** The `reviewText` column is pre-processed to enhance text quality by removing stopwords, URLs, HTML tags, numbers, or apostrophes using regular expressions (regex). This ensures a better quality of text data for analysis and faster convergence during modeling.

5. **Missing Data Imputation:** Missing values in columns with less than 30% missing data are imputed using appropriate strategies.

## Feature Engineering Steps

1. **Numerical Column Scaling:** Numerical columns in the dataset are scaled using Min-Max scaling, ensuring that all features have the same scale for modeling.

2. **Text Vectorization:** The text data is vectorized using TF-IDF Vectorizer, transforming the review text into numerical features that can be used for machine learning models.


## 
































## Repository Structure

- `/data`: Contains the dataset files (`movies.csv`, `train.csv`, and `test.csv`).
- `/notebooks`: Jupyter notebooks for data analysis, preprocessing, and model building.
- `/scripts`: Custom scripts for data cleaning and preprocessing.
- `/models`: Saved machine learning models.
- `/images`: Image files used in the project description.
- `README.md`: This project's main README file, which contains an overview of the project, data, preprocessing, and feature engineering steps.

## Getting Started

To get started with this project, you can follow these steps:

1. Clone this repository to your local machine.
2. Set up a Python environment with the required libraries listed in the `requirements.txt` file.
3. Explore the Jupyter notebooks in the `/notebooks` directory for detailed analysis and model development.

Feel free to contribute, provide feedback, or report issues if you find this project interesting. Happy coding!
