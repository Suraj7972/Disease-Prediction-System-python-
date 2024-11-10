# Disease-Prediction-System-python-
In this project you can predict the disease using images and dataset which is very useful for farmer 
Disease Prediction System

This Python script implements a disease prediction system using machine learning models trained on a medical dataset. It combines predictions from three different models (Support Vector Machine, Naive Bayes, Random Forest) to provide a more robust diagnosis suggestion.

Key Features:

Data Preprocessing: Handles missing values and encodes categorical data (e.g., disease names) for machine learning compatibility.
Data Exploration: Analyzes the distribution of disease types to understand data balance.
Machine Learning Models: Trains three models (SVM, Naive Bayes, Random Forest) on the processed data.
Cross-Validation: Evaluates model performance using 10-fold cross-validation to avoid overfitting.
Model Comparison: Displays the mean accuracy of each model to identify the best performer.
Training and Testing: Trains the final models on the entire dataset and evaluates them on a separate test set.
Performance Evaluation: Calculates accuracy and generates confusion matrices to assess model performance.
Ensemble Prediction: Takes the mode of predictions from all three models to make the final prediction.
Prediction Function: Defines a function predictDisease that takes symptom names as input and returns predictions from each model and the final diagnosis.
Usage:

Prerequisites:

Ensure you have the following libraries installed:
numpy
pandas
scipy.stats
matplotlib.pyplot
seaborn
sklearn (with submodules preprocessing, model_selection, svm, naive_bayes, ensemble, metrics)
Data Files:

Place your training data in a CSV file named Training.csv.
Ensure the data has the target variable named prognosis and features (symptoms) as columns.
Place your test data in a CSV file named Testing.csv with the same format as the training data.
Run the Script:

Save the code as a Python file (e.g., disease_prediction.py).

Execute the script from your terminal:

Bash
python disease_prediction.py
Use code with caution.

Prediction Function:

After running the script, you can utilize the defined predictDisease function to get predictions for new cases.
Pass a comma-separated string of symptoms as input to the predictDisease function.
The function will return a dictionary containing predictions from each model and the final ensemble prediction.
Output:

The script will display:
Exploration of the disease distribution.
Mean accuracy scores of each model using cross-validation.
Accuracy scores and confusion matrices for the final models on the test set.
Sample prediction using the predictDisease function.
Further Enhancements:

Implement data cleaning and handling techniques to address potential data quality issues.
Explore feature engineering techniques to create new features from existing ones, potentially improving model performance.
Consider incorporating feature importance analysis to identify the most significant symptoms for prediction.
Develop a user interface or web application to make the prediction system more accessible and user-friendly.
This comprehensive README provides clear instructions, highlights key functionalities, and explores potential areas for improvement, guiding users effectively in utilizing the disease prediction system.
