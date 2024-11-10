# Disease-Prediction-System-python-
This article aims to implement a robust machine-learning model that can efficiently predict the disease of a human, based on the symptoms that he/she possesses. Let us look into how we can approach this machine-learning problem:
![Screenshot 2024-11-10 154337](https://github.com/user-attachments/assets/de3536fa-1a06-49fe-9b29-f3587bff83ac)

Approach:

Gathering the Data: Data preparation is the primary step for any machine learning problem. We will be using a dataset from Kaggle for this problem. This dataset consists of two CSV files one for training and one for testing. There is a total of 133 columns in the dataset out of which 132 columns represent the symptoms and the last column is the prognosis.
Cleaning the Data: Cleaning is the most important step in a machine learning project. The quality of our data determines the quality of our machine-learning model. So it is always necessary to clean the data before feeding it to the model for training. In our dataset all the columns are numerical, the target column i.e. prognosis is a string type and is encoded to numerical form using a label encoder.
Model Building: After gathering and cleaning the data, the data is ready and can be used to train a machine learning model. We will be using this cleaned data to train the Support Vector Classifier, Naive Bayes Classifier, and Random Forest Classifier. We will be using a confusion matrix to determine the quality of the models.
Inference: After training the three models we will be predicting the disease for the input symptoms by combining the predictions of all three models. This makes our overall prediction more robust and accurate.
At last, we will be defining a function that takes symptoms separated by commas as input, predicts the disease based on the symptoms by using the trained models, and returns the predictions in a JSON format. 

![Screenshot 2024-11-10 154407](https://github.com/user-attachments/assets/f7e87708-0f58-4f17-969b-5db70f91874c)

Implementation:




Make sure that the Training and Testing are downloaded and the train.csv, test.csv are put in the dataset folder. Open jupyter notebook and run the code individually for better understanding.
This Python script implements a disease prediction system using machine learning models trained on a medical dataset. It combines predictions from three different models (Support Vector Machine, Naive Bayes, Random Forest) to provide a more robust diagnosis suggestion.

![Screenshot 2024-11-10 154429](https://github.com/user-attachments/assets/0a019899-b98c-4eed-91ab-9b78c25e19c9)


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
