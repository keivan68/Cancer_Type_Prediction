# cancer_type
Multi-class Classification on Cancer Data To Predicting the Cancer Type

Multi-class Classification on Cancer Data To Predicting the Cancer Type by the Patients Features like Age, Height, Sex, and Medication
Methods: Feature Engineering, Neural Network, XGBoost, CatBoost

Key findings:


1. Outliers and errors are detected and removed with these criteria:
-	Females with prostate cancer
-	Males with ovary or cervix cancer
-	Negative ages
-	Heights less than 10 inches
-	Ages above 1 and heights below 17 inches
I assumed that there is no transgender person in the dataset, so that we can remove males with female-specific cancers and vice versa.

2. There are many rows with missing sex. We do not remove the missing data, instead will impute them by zero and consider them as a third type of sex, i.e., we have male, female, and 0 as sex.
 
3. A simple neural net with 3 layers will give 31.4 % accuracy on the multiclass classification.
Another type of neural net with 3 layers will give 31.4 % accuracy on the multiclass classification.
XGBoost with multiclass loss function will give 

4. By looking at the dictionaries dict1 and dic2 which represent each class accuracy, we infer that the classification accuracy for male-specific and female-specific cancers are high and some are predicted very well. This is because the fact that there is a tight correlation between sex and these types of cancers.
