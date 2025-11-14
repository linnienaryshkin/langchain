# Developing Machine Learning Solutions

## Machine Learning Development Lifecycle (MLDL)

0. Business goal identification (key performance indicators (KPIs))
1. ML problem framing
2. Data processing (data collection, data preprocessing, and feature engineering)
3. Model development (training, tuning, and evaluation)
4. Model deployment (inference and prediction)
5. Model monitoring
6. Model retraining

### Use case: Amazon call center

Amazon aims to improve call-center efficiency and customer experience by automatically classifying customer calls. The project starts with defining the business goal—identifying the type of customer issue (e.g., _money transfer_ vs. _lost card_) to route calls correctly and support agents.

Data is collected from historical call recordings, transcribed, cleaned, and preprocessed to remove noise and standardize text. After that, labeled examples are used to train a machine-learning model capable of predicting the call category.

The model is evaluated using accuracy and other metrics, then tuned to improve performance. Once reliable, the model is deployed into the call-center workflow, where it helps classify incoming calls in real time and supports better routing and faster customer service.

## Machine Learning Models Performance Evaluation

- training set | 80% of the data
- validation set | 10% of the data | used to tune the model
- test set | 10% of the data | used to evaluate the model

### Model fit

- Underfitted | the model is too simple and does not fit the data
- Overfitted | the model is too complex and fits the noise in the data
- Balanced | the model is just right and fits the data

**Bias and variance** | the model should have low bias and low variance

## ML problems

- Classification | the model is trained to classify data into a specific category (Fraud detection, Image classification, etc.)
- Regression | the model is trained to predict a continuous value (Weather forecasting, Stock price prediction, etc.)

### Classification

- Binary classification | the model is trained to classify data into two categories (Yes/No, True/False, etc.)
- Multiclass classification | the model is trained to classify data into multiple categories (Fraud detection, Image classification, etc.)

Metrics:

- Confusion matrix | True positive (TP), False positive (FP), True negative (TN), False negative (FN)
- Accuracy | the percentage of correct predictions (TP + TN) / (TP + TN + FP + FN)
- Precision | the percentage of true positives out of all positive predictions (TP) / (TP + FP)
- Recall | the percentage of true positives out of all actual positives (TP) / (TP + FN)
- AUC-ROC | the area under the receiver operating characteristic curve

### Regression

- Linear regression | the model is trained to predict a continuous value (Weather forecasting, Stock price prediction, etc.)

Metrics:

- Mean squared error (MSE) | the average of the squares of the errors (y - y_pred)^2
- R squared | the coefficient of determination (R^2) | the percentage of the variance in the dependent variable that is explained by the independent variables | 1 means a perfect fit

## MLOps (Machine Learning Operations)

- Increase the pace of the model development lifecycle through automation.
- Improve quality metrics through testing and monitoring.
- Promote a culture of collaboration between data scientists, data engineers, software engineers, and IT operations.
- Provide transparency, explainability, audibility, and security of the models by using model governance.
