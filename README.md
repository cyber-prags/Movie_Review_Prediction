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

## Pipelines and Transformers:
We made use of Pipelines and ColumnTransformers to streamline our task and reduce the code complexity of the notebook.

![pipeline eg](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/700205c4-c16b-44fb-a2f1-c5dd986516dc)



## Model Selection

To harness the full potential of our dataset and effectively predict movie sentiment, we have carefully selected a diverse range of machine learning models. Our model selection includes the following six powerful candidates:

1. **Logistic Regression**: A fundamental yet robust linear model, widely used in classification tasks.
2. **Naive Bayes**: Leveraging probabilistic approaches, Naive Bayes is known for its simplicity and effectiveness in text classification.
3. **Stochastic Gradient Descent Regressor (SGD)**: A versatile optimization technique that adapts well to various machine learning tasks.
4. **Support Vector Classifier (SVC)**: Harnessing the power of support vector machines, SVC is a popular choice for binary and multiclass classification.
5. **Light Gradient Boosting Model**: A highly efficient gradient boosting framework that excels in handling complex datasets.
6. **XGBoost**: A leading gradient boosting algorithm renowned for its performance and scalability in both classification and regression tasks.

Each of these models brings its unique strengths to the table, and we will rigorously evaluate and compare their performance to determine the most effective approach for sentiment prediction.

Stay tuned as we delve into the world of machine learning to discover which model outshines the rest in our quest to predict movie sentiment with precision and accuracy.

## Comparitive Analysis of models:

1. **PR-curves:**
   - **For Training Set:**
     
     In the cumulative PR-curves below we can see how the various models perform on the test dataset. In a PR curve; a model with a higher Area Under Curve(AUC) is generally deemed to perform better. In the graph below we see that :
     
     > SVC performs the best with an AUC of 0.98.
     
     > It is followed by logit(LogisticRegression) and NaiveBayes with 0.95.
     
     >![PR-train](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/d41b7777-21c5-44f9-b3e8-797e1d1b69bb)

- **For Test Set:**
  
     We can see similar observations in the Test set as follows:
  
     > LogisticRegression,SVC and NaiveBayes performs the best with an AUC of 0.92.
     
     > It is followed by SGD with 0.91.
     
     >![PR-test](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/cebf8adb-313c-494f-a26a-a306dfa1c8fd)
     

This suggests that for the PR curves; LogisticRegression, SVC and NaiveBayes perform quite well on the dataset.

2. **F1-micro scores:**
   
   In the comparitive analysis of the F1-scores; we see that:

   > SVC has the highest F1-score on the training data; followed by Logistic Regression.
   
   > LogisticRegression and SVC has similar f1-micro scores for the test dataset which suggests that these models will perform the best on our dataset as also shown by the PR-curves.
 
   ![F1-comparision](https://github.com/cyber-prags/Movie_Review_Prediction/assets/74003758/81d142e2-9fa8-4a5e-9c32-b0c832999deb)


3. To have a look at the other scores like `Accuracy`, `Precision`,`Recall` please do refer the following notebook: https://github.com/cyber-prags/Movie_Review_Prediction/blob/main/Movie_Review_Sentiments_Notebook.ipynb

## Hyper-tuning:

The best performing  models were further hypertuned but they did not show significant improvement in performance and hence the baseline models were chosen for faster convergence of code.
   


# Final Conclusion:

## Model summary without hyper-parameter tuning
 
> In terms of accuracy,the top performing models without hyper-parameter tuning are: **LogisticRegressor,Calibrated LinearSVC with 80%, Complement Naive Bayes** with an accuracy of **79%**

## Model Analysis
### Logistic Regression (logit):

> **Pros**: Balanced detection of both positive and negative sentiments.

> **Cons:** Potential misclassification may lead to inaccuracies in understanding customer sentiment, affecting decision-making in marketing or product development.

> **Business Implication:** Ideal for a general overview of customer sentiment. Suitable for applications like brand monitoring where both positive and negative sentiments are equally important.


### Naive Bayes (nb):

> **Pros**: High detection of negative sentiments (low False Negatives).

> **Cons:** The tendency to miss positive sentiments may lead to underestimation of customer satisfaction, potentially affecting strategies for brand promotion and loyalty programs.

> **Business Implication:** Useful when it's crucial to capture negative feedback, such as in quality control or customer service improvement.

### Stochastic Gradient Descent (sgd):

> **Pros:** Balanced detection of positive and negative sentiments.

> **Cons:** Moderate incorrect classifications could create challenges in precisely targeting customer segments or tailoring personalized marketing strategies.

> **Business Implication:** A versatile option that might require further tuning for specific use cases like targeted marketing or product enhancement.

### LightGBM (lgbm):

> **Pros:** Reasonable detection of positive sentiments.

> **Cons:** The potential misclassification of both positive and negative sentiments might lead to misguided business strategies, such as incorrect product improvements or inefficient allocation of resources.

> **Business Implication:** May need tuning for applications like assessing customer satisfaction or promoting positive reviews.

### Support Vector Classifier (svc):

> **Pros:** Highest detection of positive sentiments.

> **Cons:** Overlooking negative feedback might lead to missed opportunities for addressing customer grievances, potentially harming brand reputation or customer retention.

> **Business Implication:** Suitable for highlighting and leveraging positive feedback, such as in advertising or enhancing positive brand image.

### XGBoost (xgboost):

> **Pros:** Balanced detection of positive and negative sentiments.

> **Cons:** Some misclassifications may reduce the effectiveness of competitive analysis or market segmentation, leading to suboptimal business decisions.

> **Business Implication:** A flexible option that might need more tuning for applications like market segmentation or competitive analysis.


#### Decision:

**Best Model:** Both logit and svc are strong candidates.

> If the goal is to obtain a **balanced view of customer sentiments**, <code style="background:red;color:white">logit</code> might be the preferred choice.

> If the focus is on **leveraging positive feedback for marketing or brand enhancement**, <code style="background:red;color:white">svc</code> might be more suitable.

####  Business Considerations:
    The choice of model should align with the specific goals of the sentiment analysis:

- **Customer Service Improvement:** Focus on models that detect negative sentiments effectively (e.g., nb).

- **Brand Promotion:** Consider models that highlight positive sentiments (e.g., svc).

- **Overall Market Analysis:** Choose a model that provides a balanced view (e.g., logit).

In summary, the selection of the model should be closely tied to the business objectives of the sentiment analysis. Understanding the context, the importance of positive vs. negative sentiments, and the specific use case will guide the final decision. Collaboration with domain experts and further validation can also help in optimizing the model for the desired business outcome.






























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
