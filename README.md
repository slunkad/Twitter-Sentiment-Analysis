# Twitter-Sentiment-Analysis
****The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.****

I have followed a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

## Feature Extraction

Two feature extraction techniques have been utilized in this project:

### Bag of Words (BoW)

The Bag of Words approach converts text into numerical features. Each tweet is represented as a vector, where each element corresponds to the count of a specific word in the tweet. The scikit-learn `CountVectorizer` is used to implement BoW.

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is another technique used to convert text into numerical features. It assigns weights to words based on their frequency in a tweet and inverse document frequency across all tweets. The scikit-learn `TfidfVectorizer` is used to implement TF-IDF.

## Models Implemented

Four machine learning models have been implemented for sentiment analysis:

### 1. Logistic Regression

Logistic Regression is a simple yet effective binary classification algorithm. In this project, it is adapted for multi-class classification by using the "one-vs-rest" approach.

### 2. Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.

### 3. Support Vector Machine (SVM)

SVM is a powerful classification algorithm that finds the best hyperplane to separate classes in a high-dimensional feature space.

### 4. XGBoost

XGBoost is an efficient and widely used gradient boosting algorithm known for its speed and performance.

## Model Tuning

To optimize the model's performance, hyperparameter tuning has been performed for each of the implemented models. This process involves searching for the best combination of hyperparameters that result in the highest accuracy or F1 score on a validation set. Techniques like Grid Search or Randomized Search can be used for tuning.

## Conclusion

This Twitter Sentiment Analysis project demonstrates the effectiveness of Bag of Words and TF-IDF techniques for feature extraction, along with four different machine learning models for sentiment classification. The model tuning process enhances the models' performance and ensures better generalization to new data.
