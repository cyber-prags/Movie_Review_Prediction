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

5. **Missing Data Imputation:** Missing values in columns with less than 30% missing data are imputed using appropriate strategies like `most frequent` ,`median`, `mean`.

6. **Encoding Data:** Data was encoded using **LabelEncoder**.

## Feature Engineering Steps

1. **Numerical Column Scaling:** Numerical columns in the dataset are scaled using Min-Max scaling, ensuring that all features have the same scale for modeling.

2. **Text Vectorization:** The text data is vectorized using TF-IDF Vectorizer, transforming the review text into numerical features that can be used for machine learning models.


# Key Insights

Our in-depth Exploratory Data Analysis (EDA) of the dataset has revealed a wealth of fascinating insights, shedding light on various aspects of movie reviews and audience sentiments. These discoveries provide essential context for our movie sentiment prediction project, making it more informative and engaging.

**1. Distinguished Reviewers:**
   - Among the reviewers, John Luna holds the top position, closely followed by the insightful Bryan Phillips.

   ![Reviewer Ranking](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/120cb58c-3eef-4b50-b986-2ea96e0d8031)

**2. Audience Score Spectrum:**
   - The spectrum of audience scores assigned by reviewers spans a wide range, reflecting diverse opinions and tastes.

   ![Audience Score Distribution](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/8719c644-c224-46d1-8db2-3d1ba9e855d3)

**3. Runtime Diversity:**
   - Our dataset showcases a fascinating diversity in movie runtimes, echoing the broad spectrum of cinematic experiences.

   ![Runtime Distribution](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/031831f7-50a5-498b-9245-c044c1a42dc4)

**4. Movie Language Mosaic:**
   - English emerges as the dominant language for the movies, followed by French and British English, forming a rich linguistic tapestry.

   ![Movie Language Distribution](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/9d58dbd5-0b20-470f-bafd-54fcee2282cb)

**5. Genre Galore:**
   - The cinematic landscape is dominated by Drama, closely accompanied by the genres of Comedy and Documentary, offering a captivating blend of storytelling.

   ![Genre Distribution](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/58ff9ff1-c479-4c46-bb98-3edfc0322c6b)

**6. Remarkable Release Dates:**
   - The cinematic calendar highlights that May 22, 2017, was a momentous day with the release of an impressive 653 movies, closely followed by the cinematic offerings on August 27, 2019.

   ![Release Date Distribution](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/6337a6d3-c0d3-4b87-b199-baf314d48d89)

**7. Audience Score and Sentiment Correlation:**
   - A captivating connection between audience scores and movie sentiment comes into view. Movies with audience scores exceeding 60 often carry a positive sentiment, uncovering an interesting trend.

   ![Audience Score vs. Sentiment](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/950c0494-e874-4a78-b450-25d52e59b5de)

**8. Runtime's Impact on Sentiment:**
   - A compelling correlation between the duration of movies and their sentiment surfaces, hinting at the influence of runtime on audience feelings.

   ![Runtime vs. Sentiment](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/c0813436-a3cc-4638-9ef7-46acb6fb27bd)

**9. Comprehensive Correlation Matrix:**
   - Our meticulously constructed correlation matrix validates our initial observations, reaffirming the links between audience score, runtime minutes, and movie sentiment.

   ![Correlation Matrix](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/5306d9a6-203e-4c6b-9d34-304a9b8c5595)

These captivating insights provide the groundwork for our movie review sentiment prediction project. They not only offer valuable context but also fuel our project with meaningful features and trends, promising an engaging journey ahead.






































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
