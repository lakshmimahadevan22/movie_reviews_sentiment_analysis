# Sentiment Analysis of IMDB Movie Reviews

This project focuses on sentiment analysis of movie reviews using the IMDB dataset. The dataset contains 50,000 movie reviews labeled as positive or negative. The primary objective is to build and evaluate multiple machine learning models that can accurately classify the sentiment of these reviews.

## Dataset

| Sentiment | Number of Samples |
|-----------|-------------------|
| Positive  | 25,000            |
| Negative  | 25,000            |
| **Total** | **50,000**        |

Dataset Link: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Data Preprocessing

Before training, the dataset undergoes the following preprocessing steps:

- Removal of HTML tags using BeautifulSoup  
- Lowercasing all text  
- Removal of special characters using Regular Expressions  
- TF-IDF Vectorization to extract n-gram based features (bigrams used)  
- Dimensionality reduction using Truncated Singular Value Decomposition (SVD)

**Note:** Unlike deep learning approaches, this project does not use tokenization or padding, as it uses classical ML models with fixed-length feature vectors from TF-IDF.

## Feature Engineering

**TF-IDF Vectorizer**  
Extracts bigram features (2-word combinations) with a vocabulary of up to 50,000 features.

**Truncated SVD (Latent Semantic Analysis)**  
Reduces the feature space to 200 components, preserving key latent semantic structures while improving computational efficiency.

## Sentiment Classification Models

This project compares the performance of four classical machine learning models:

| Model                 | Description                                      |
|----------------------|--------------------------------------------------|
| Logistic Regression  | A simple linear model for binary classification  |
| Random Forest        | An ensemble method using bagging and decision trees |
| Support Vector Machine (SVM) | A powerful margin-based classifier for high-dimensional data |
| XGBoost              | Gradient boosting decision trees with high performance and regularization |

All models are tuned using GridSearchCV to find the best hyperparameters based on F1 Score.

## Model Training and Evaluation

Each model is trained on 80% of the data and tested on 20% using a stratified split. The following evaluation metrics are used:

- Accuracy  
- Precision  
- Recall  
- F1 Score

**From experimentation, XGBoost and SVM usually perform the best on this dataset due to their robustness in high-dimensional settings.**


## Requirements

To run this project, ensure you have the following Python libraries installed:

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- beautifulsoup4  


Download the IMDB dataset and unzip it to your local path.

Update the `BASE_DIR` variable inside `sentiment_analysis.py` with your dataset path:

```python
BASE_DIR = "path_to_your_dataset"
```
