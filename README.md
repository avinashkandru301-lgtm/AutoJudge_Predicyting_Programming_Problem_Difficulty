# AutoJudge_Predicyting_Programming_Problem_Difficulty 

This project aims to automatically predict the difficulty level of programming problems using machine learning and natural language processing techniques.
Given a problem statement (title, description, input, and output), the system predicts:

Difficulty Class: Easy / Medium / Hard (Classification) Difficulty Score: A continuous numerical score representing relative difficulty (Regression)

DataSet used is the Custom JSON lines ( .json1) dataset of programming problems
it consists of a total of 4,112 problems. 

Approach & Models Used
1️. Data Processing & Cleaning
Merged multiple text fields into a single problem description
Converted text to lowercase
Handled missing values using .fillna("")
2️. Feature Engineering
The system uses a hybrid feature extraction strategy:

Text Features
Character-level TF-IDF
Captures mathematical symbols, formatting, and problem structure
N-grams: 3 to 5 characters
Keyword-Based Features
Explicit detection of:
Algorithms (e.g., graph, DP, greedy, BFS, DFS)
Data structures (e.g., array, stack, heap, queue)
Numeric / Structural Features
Extracted from problem text:

Length of problem statement
Number of numeric values
Presence of constraints (≤, <=)
Logarithmic terms (log)
Modulo operations (mod)
Time complexity expressions (e.g., O(n))
All numeric features are standardized using StandardScaler.

3️. Models Used
Classification Model
classifier: XGBoost Classifier
Predicts: Easy / Medium / Hard
Regression Model
Regression model : XGBoost Regressor**
Predicts a continuous difficulty score
Both models use the same feature vector for consistency.

Evaluation Metrics
Classification Performance
Metric: Accuracy, Precision, Recall, F1-score
Observed Accuracy: ~52%
Regression Performance
RMSE: ~1.99
MAE: ~1.64
R² Score: ~0.17
The regression model captures relative difficulty trends rather than absolute scores.

Web UI – Problem Difficulty Predictor
Overview
This web interface provides an interactive way to predict the difficulty of programming problem using trained machine learning model.

Users can paste:

Problem description
Input format
Output format
The app then predicts:

Difficulty Class: Easy / Medium / Hard
Difficulty Score: Continuous numerical value
The interface is built using Streamlit, making it lightweight and easy to run locally.

Backend Models Used
The web UI loads pre-trained models and preprocessors:

reg_class.pkl → XGBoost Classifier (difficulty class)
reg.pkl → XGBoost Regressor (difficulty score)
tfidf_char.pkl → Character-level TF-IDF vectorizer
tfidf_keywords.pkl → Keyword-based TF-IDF vectorizer
scaler.pkl → StandardScaler for numeric features
label_encoder.pkl → LabelEncoder for class labels
All models are loaded using joblib.

User Interface Components
The Streamlit UI includes:

Text Areas:

Problem Description
Input Description
Output Description
Predict Button:

Triggers feature extraction and inference
Results Display:

Predicted difficulty class
Predicted difficulty score (rounded to 2 decimals)
