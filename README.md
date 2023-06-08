<!DOCTYPE html>
<html>
		
<body>

<h1>Machine Learning: Diabetes prediction</h1>
This notebook was developed using Python and its libraries to investigate health factors related to diabetes and build a machine learning model capable of classifying whether or not a patient has diabetes.
The data is available <a href="https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset">here</a>.<br>

<h2>Methodology</h2>

<h3>Dataset Information</h3>
The dataset focuses on health factors related to diabetes and aims to identify correlations between these factors and the presence of diabetes in patients. The dataset is comprised of 100,000 rows and 9 columns, representing various health-related attributes such as age, BMI, blood glucose level, and more.

<h3>Exploratory Data Analysis and Correlation Analysis</h3>
To gain insights into the dataset and understand the relationships between different health factors and diabetes, Exploratory Data Analysis (EDA) techniques were employed. The EDA process involved analyzing the dataset's characteristics, identifying correlations between features, and exploring potential patterns or trends that could inform the subsequent modeling steps.

<h3>Data Preprocessing</h3>
The following preprocessing steps were performed on the dataset before training the models:
<ul>
	<li>Standardization: Numerical features were standardized using <code>StandardScaler</code> to ensure all features were on a similar scale, which aids in the convergence of machine learning algorithms.</li>
	<li>One-Hot Encoding: Categorical features were encoded using <code>OneHotEncoder</code> to convert them into numerical representations suitable for machine learning algorithms. This step was achieved using Pipeline and ColumnTransformer to streamline the preprocessing pipeline.</li>
</ul>

<h3>Handling Imbalanced Data</h3>
As the target variable for diabetes classification was found to be imbalanced, techniques were applied to address this issue. Both <code>SMOTE</code> (Synthetic Minority Over-sampling Technique) and <code>RandomUnderSampler</code> were utilized to balance the dataset, which helps prevent the model from being biased toward the majority class. 
  
<h3>Train-Test Split</h3>
To evaluate the models' performance, the dataset was divided into training and testing sets using the <code>train_test_split</code> function. The test set, which was held out from the training process, allows for unbiased evaluation of the model's generalization ability.

<h3>Model Selection and Evaluation</h3>
Cross-validation using <code>StratifiedKFold</code> and <code>cross_val_scores</code> was performed to assess the performance of multiple models. <code>LogisticRegression</code>, <code>CatBoostClassifier</code>, <code>LGBMClassifier</code>, <code>RandomForestClassifier</code>, and other models were evaluated based on their performance metrics. The goal was to select three models that exhibited the best performance for further analysis.

<h3>Hyperparameter Tuning</h3>
The hyperparameters of the three selected models were tuned to optimize their performance. This process involved searching through different combinations of hyperparameters to find the configuration that maximizes the model's effectiveness in classifying diabetes.

<h3>Ensemble Voting Classifier</h3>
The three best-performing models were combined into a <code>VotingClassifier</code>, which aggregates the predictions from each model to make the final classification decision. This ensemble approach helps improve the overall predictive accuracy by leveraging the strengths of each individual model.
  
<h3>Feature Importance Analysis</h3>
The feature importance of each model was extracted individually to understand the relative importance of different features in predicting diabetes. These feature importance scores were then plotted, allowing for a visual assessment of the significant factors contributing to the classification task.

<h2>Results</h2>
<p>The following image show us the results of each model evaluated during the cross-validation.</p>
  
![image](https://github.com/gfacheti/ML-Diabetes/assets/106284497/0877fec2-7a61-4f56-bfff-e7685eae8886)

<p>As the best performers, AdaBoost, LGBM, and CatBoost were selected for tuning. The following images show the results and their best parameters.</p>

![image](https://github.com/gfacheti/ML-Diabetes/assets/106284497/9f5fcc87-0f16-43ee-a41c-aa2fe687e507)
![image](https://github.com/gfacheti/ML-Diabetes/assets/106284497/697bdc0f-65ea-40c0-840b-b5a52ee2c3a6)

<p>After the tuning, all three models were ensembled in a VotingClassifier. The results are shown below.</p>

![image](https://github.com/gfacheti/ML-Diabetes/assets/106284497/a5b8d004-e177-424e-bf1f-ce9c6ff87a32)
![image](https://github.com/gfacheti/ML-Diabetes/assets/106284497/84078f79-d097-472e-a99b-f10f52a02b2f)

<p>Finally, I evaluated the feature importances of each model. Overall, the results align well with the literature about risk factors for diabetes.</p>
  
![image](https://github.com/gfacheti/ML-Diabetes/assets/106284497/9ddd26a2-686d-4662-a872-8219dbe2c702)

</body>
</html>
