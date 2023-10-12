# Movie Review Sentiment Prediction

![mih10uhu1464fx1kr0by-1](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/75ce6d67-4657-48ba-86b7-e59b6bacade8)

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


## Key Insights

Our in-depth Exploratory Data Analysis (EDA) of the dataset has revealed a wealth of fascinating insights, shedding light on various aspects of movie reviews and audience sentiments. These discoveries provide essential context for our movie sentiment prediction project, making it more informative and engaging.

**Data Imbalance:**
We have an imbalanced dataset in our hands, with a majority of Positive sentimented classes.
![sentiment imbalance](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/2a40147d-99bd-489e-8278-945146698314)

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

## Feature Selection

Our feature selection process is guided by insights obtained during the Exploratory Data Analysis. We have identified key features that exhibit a strong correlation with movie sentiment. These features will form the basis of our predictive models:

- `reviewText`: The content of the movie reviews.
- `audienceScore`: The audience scores provided by reviewers.
- `runtimeMinutes`: The duration of the movie in minutes.

These features were chosen based on their strong positive correlation with movie sentiment.

```python
X = merged_train_data[['reviewText', 'audienceScore', 'runtimeMinutes']]
y = merged_train_data['sentiment']
```


## Evaluation metrics:

In our model evaluation, we recognize the slight class imbalance in our dataset. To ensure a comprehensive assessment, we have selected the following evaluation metrics:

`F1-Micro Score`: This metric offers a balanced evaluation of precision and recall and is particularly effective for datasets with class imbalances.

In addition to the F1-Micro Score, we will employ `Precision-Recall (PR)` curves to assess and compare the performance of our models.

# Pipelines and Transformers

To streamline our workflow and reduce code complexity, we've harnessed the power of Pipelines and ColumnTransformers. These tools enable us to efficiently preprocess data and apply transformations with ease, enhancing the overall efficiency of our project.

![Pipeline Example](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/700205c4-c16b-44fb-a2f1-c5dd986516dc)

## Model Selection

Our journey towards effective sentiment prediction is guided by a carefully curated selection of machine learning models. Our model lineup boasts diversity and includes the following six robust contenders:

1. **Logistic Regression**: A fundamental yet versatile linear model, well-suited for classification tasks.
2. **Naive Bayes**: Embracing probabilistic approaches for simplicity and efficiency in text classification.
3. **Stochastic Gradient Descent Regressor (SGD)**: A versatile optimization technique adaptable to various machine learning challenges.
4. **Support Vector Classifier (SVC)**: Leveraging the power of support vector machines, a popular choice for binary and multiclass classification.
5. **Light Gradient Boosting Model**: A highly efficient gradient boosting framework, adept at handling complex datasets.
6. **XGBoost**: A leading gradient boosting algorithm renowned for its performance and scalability in both classification and regression tasks.

These models bring their unique strengths to the table and will undergo rigorous evaluation and comparison to identify the most effective approach for predicting movie sentiment.

Stay with us as we delve deeper into the realm of machine learning, and discover which model rises to the challenge of predicting movie sentiment with precision and accuracy.

## Comparative Analysis of Models

### PR-Curves

#### For the Training Set
In the cumulative PR-curves below, we evaluate the performance of our models on the training dataset. The Area Under the Curve (AUC) in a PR curve is a vital metric, with a higher AUC indicating superior model performance. Our observations reveal:

- **SVC** leads the way with an AUC of 0.98.
- It is closely followed by **Logistic Regression** and **Naive Bayes** with an AUC of 0.95.

![PR-Curves for Training Set](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/d41b7777-21c5-44f9-b3e8-797e1d1b69bb)

#### For the Test Set
We observe similar trends in the test dataset:

- **Logistic Regression**, **SVC**, and **Naive Bayes** excel with an AUC of 0.92.
- **SGD** follows closely with an AUC of 0.91.

![PR-Curves for Test Set](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/cebf8adb-313c-494f-a26a-a306dfa1c8fd)

These insights suggest that for PR curves, **Logistic Regression**, **SVC**, and **Naive Bayes** exhibit strong performance on the dataset.

### F1-Micro Scores

In the comparative analysis of F1-scores, we observe:

- **SVC** achieves the highest F1-score on the training data, closely followed by **Logistic Regression**.
- On the test dataset, both **Logistic Regression** and **SVC** exhibit similar F1-micro scores, reinforcing their strong performance as indicated by the PR-curves.

![F1-Score Comparison](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/81d142e2-9fa8-4a5e-9c32-b0c832999deb)

For further insights, including accuracy, precision, and recall scores, please refer to our notebook: [Movie_Review_Sentiments_Notebook](https://github.com/cyber-prags/Movie_Review_Prediction/blob/main/Movie_Review_Sentiments_Notebook.ipynb).

## Hyper-Tuning

Our journey to optimize model performance involved hyper-parameter tuning. However, the best-performing models did not exhibit significant improvements through this process. As a result, we opted for baseline models to ensure faster code convergence and maintain efficient results.

# Final Conclusion

In terms of accuracy, our top-performing models without hyper-parameter tuning are **Logistic Regression**, **Calibrated Linear SVC**, and **Complement Naive Bayes**, all achieving an accuracy of 79%.

## Model Analysis

### Logistic Regression (Logit)

**Pros:** Balances the detection of both positive and negative sentiments.

**Cons:** Potential misclassification may impact decision-making in marketing or product development.

**Business Implication:** Ideal for obtaining a general overview of customer sentiment. Suitable for applications like brand monitoring, where both positive and negative sentiments are equally important.

### Naive Bayes (NB)

**Pros:** High detection of negative sentiments (low False Negatives).

**Cons:** The tendency to miss positive sentiments may impact strategies for brand promotion and loyalty programs.

**Business Implication:** Useful when capturing negative feedback is crucial, such as in quality control or customer service improvement.

### Stochastic Gradient Descent (SGD)

**Pros:** Balanced detection of positive and negative sentiments.

**Cons:** Moderate incorrect classifications could create challenges in precisely targeting customer segments or tailoring personalized marketing strategies.

**Business Implication:** A versatile option that might require further tuning for specific use cases, such as targeted marketing or product enhancement.

### LightGBM (LGBM)

**Pros:** Reasonable detection of positive sentiments.

**Cons:** The potential misclassification of both positive and negative sentiments might lead to misguided business strategies, such as incorrect product improvements or inefficient allocation of resources.

**Business Implication:** May need tuning for applications like assessing customer satisfaction or promoting positive reviews.

### Support Vector Classifier (SVC)

**Pros:** Highest detection of positive sentiments.

**Cons:** Overlooking negative feedback might lead to missed opportunities for addressing customer grievances, potentially harming brand reputation or customer retention.

**Business Implication:** Suitable for highlighting and leveraging positive feedback, such as in advertising or enhancing a positive brand image.

### XGBoost (XGBoost)

**Pros:** Balanced detection of positive and negative sentiments.

**Cons:** Some misclassifications may reduce the effectiveness of competitive analysis or market segmentation, leading to suboptimal business decisions.

**Business Implication:** A flexible option that might need more tuning for applications like market segmentation or competitive analysis.

#### Decision

**Best Model:** Both Logistic Regression and Support Vector Classifier (SVC) exhibit strong performance.

- If the goal is to obtain a **balanced view of customer sentiments**, <code style="background:red;color:white">Logistic Regression</code> might be the preferred choice.

- If the focus is on **leveraging positive feedback for marketing or brand enhancement**, <code style="background:red;color:white">SVC</code> might be more suitable.

**Business Considerations:**
The choice of model should align with the specific goals of sentiment analysis:

- **Customer Service Improvement:** Focus on models that detect negative sentiments effectively (e.g., nb).

- **Brand Promotion:** Consider models that highlight positive sentiments (e.g., svc).

- **Overall Market Analysis:** Choose a model that provides a balanced view (e.g., logit).


Our journey through the world of sentiment analysis in the realm of movie reviews has led us to a fascinating array of models and performance evaluations. Through PR-curves and F1-micro scores, we've identified standout candidates such as Logistic Regression and Support Vector Classifier. These models exhibit the potential to predict movie sentiment with precision and accuracy, while our business considerations allow us to tailor our choice to the specific objectives of the analysis.

Despite efforts in hyper-parameter tuning, our baseline models stood strong as the top performers, ensuring faster convergence and maintaining efficiency.

Explore the notebook, delve into the models, and uncover the insights that matter most to you.

Thank you for being with me in this journey through data-driven decision-making and the exciting world of movie sentiment prediction.



## Repository Structure

- `/data`: Contains the dataset files (`movies.csv`, `train.csv`, and `test.csv`).
- `/notebooks`: Jupyter notebooks for data analysis, preprocessing, and model building.
- `/images`: Image files used in the project description.
- `README.md`: This project's main README file, which contains an overview of the project, data, preprocessing, and feature engineering steps.

## Getting Started

To get started with this project, you can follow these steps:

1. Clone this repository to your local machine.
2. Set up a Python environment with the required libraries listed in the `requirements.txt` file.
3. Explore the Jupyter notebooks in the `/notebooks` directory for detailed analysis and model development.

Feel free to contribute, provide feedback, or report issues if you find this project interesting. Happy coding!
