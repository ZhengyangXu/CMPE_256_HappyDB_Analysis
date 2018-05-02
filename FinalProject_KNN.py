# -*- coding: utf-8 -*-
# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

# Importing the dataset
dataset_1 = pd.read_csv('cleaned_hm.csv')
dataset_2 = pd.read_csv('demographic.csv')
dataset_2.head()

#Merging the two data sets
hm_dataset = pd.merge(dataset_1 , dataset_2)
#dropping columns which won't be needed for analysis
hm_dataset.drop(['original_hm', 'cleaned_hm','modified','num_sentence','ground_truth_category','reflection_period','hmid'], axis=1, inplace=True)
hm_dataset.head()

#Arranging the matrix and putting dependent variable in the end
hm_dataset = hm_dataset[['wid', 'age', 'gender', 'marital', 'parenthood', 'country', 'predicted_category']]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_gender = LabelEncoder()
labelencoder_parenthood = LabelEncoder()
labelencoder_marital = LabelEncoder()
labelencoder_country = LabelEncoder()
labelencoder_category = LabelEncoder()

hm_dataset.count()

#Convert nan to mode values for all columns
hm_dataset['gender'] = hm_dataset.gender.fillna( hm_dataset.gender.mode()[0])
hm_dataset['parenthood'] = hm_dataset.parenthood.fillna( hm_dataset.parenthood.mode()[0])
hm_dataset['marital'] = hm_dataset.marital.fillna( hm_dataset.marital.mode()[0])
hm_dataset['country'] = hm_dataset.country.fillna( hm_dataset.country.mode()[0])

#Encode the columns
# f , m , o = 0, 1, 2
hm_dataset["gender"] = labelencoder_gender.fit_transform(hm_dataset["gender"])
gender_codes  = labelencoder_gender.inverse_transform([0,1,2]);

#n , y = 0, 1
hm_dataset["parenthood"] = labelencoder_parenthood.fit_transform(hm_dataset["parenthood"])
parenthood_codes= labelencoder_parenthood.inverse_transform([0,1]);

#'divorced', 'married', 'separated', 'single', 'widowed' = 0, 1, 2, 3, 4
hm_dataset["marital"] = labelencoder_marital.fit_transform(hm_dataset["marital"])
marital_codes= labelencoder_marital.inverse_transform([0,1,2,3,4]);

#Encode country
hm_dataset["country"] = labelencoder_country.fit_transform(hm_dataset["country"])

#'achievement', 'affection', 'bonding', 'enjoy_the_moment',
#'exercise', 'leisure', 'nature' = 0, 1, 2, 3, 4, 5, 6
hm_dataset["predicted_category"] = labelencoder_category.fit_transform(hm_dataset["predicted_category"])
category_codes= labelencoder_category.inverse_transform([0,1,2,3,4,5,6]);

#Get dummy variables for each encoded column
hm_imputed_dataset = pd.get_dummies(hm_dataset, columns=["gender", "parenthood","marital"], prefix=["gender", "parenthood","marital"])
hm_imputed_dataset = hm_imputed_dataset.rename(index=str, 
                                               columns={"gender_0": "gender_female", "gender_1": "gender_male",
                                                        "gender_2": "others", "parenthood_0": "parenthood_nokids",
                                                        "parenthood_1": "parenthood_haskids",
                                                        "marital_0": "marital_divorced", "marital_1": "marital_married",
                                                        "marital_2": "marital_separated", "marital_3": "marital_single",
                                                        "marital_4": "marital_widowed","category_0": "Category_Achivement",
                                                        "category_1": "Category_Affection", "category_2": "Category_Bonding",
                                                        "category_3": "Category_Enjoy-the-moment", "category_4": "Category_Exercise",
                                                        "category_5": "Category_leisure", "category_6": "Category_nature" })


#Preventing dummy variable trap
hm_dataset = hm_imputed_dataset[['country', 'gender_female', 'gender_male',
       'parenthood_nokids', 'marital_divorced', 'marital_married',
       'marital_separated', 'marital_single','predicted_category']]

#Age is having some noise as strings and float, convert everything to int
hm_age = pd.Series(hm_imputed_dataset.age)
hm_age = pd.to_numeric(hm_age, errors='coerce')
hm_dataset = pd.concat( [ hm_age, hm_dataset ] , axis=1 )

#Fill nan with mean
hm_dataset['age'] = hm_dataset.age.fillna(int(hm_dataset.age.mean()))
hm_dataset = hm_dataset[(hm_dataset['age'] >= 10) & (hm_dataset['age'] <= 100) ]


#Let's take KNN algorithm and try with different feature selection strategies
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Lets go for another algorithm which can classify non binary variables
# Fitting K-NN to the Training set
# n = 5 gives acc as 0.36, n = 10 acc as 0.37 n= 15 0.39, n=20 as 0.40
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 35, weights = 'uniform', metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn = pd.DataFrame(knn_classifier.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
cr_knn_parenthood = classification_report(y_test, y_pred_knn)
acc_knn_parenthood = accuracy_score(y_test, y_pred_knn)
cm_knn_parenthood = confusion_matrix(y_test, y_pred_knn)
precision_knn_parenthood = precision_score(y_test, y_pred_knn, average = 'weighted')
recall_knn_parenthood = recall_score(y_test, y_pred_knn, average = 'weighted')

#lets start with parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Lets go for another algorithm which can classify non binary variables
# Fitting K-NN to the Training set
# n = 5 gives acc as 0.36, n = 10 acc as 0.37 n= 15 0.39, n=20 as 0.40
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 35, weights = 'uniform', metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn = pd.DataFrame(knn_classifier.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
cr_knn_pm = classification_report(y_test, y_pred_knn)
acc_knn_pm = accuracy_score(y_test, y_pred_knn)
cm_knn_pm = confusion_matrix(y_test, y_pred_knn)
precision_knn_pm = precision_score(y_test, y_pred_knn, average = 'weighted')
recall_knn_pm = recall_score(y_test, y_pred_knn, average = 'weighted')
fscore_knn_pm = f1_score(y_test, y_pred_knn, average = 'weighted')


#lets take all
X = hm_dataset.iloc[:, [0,1,2,3,4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting K-NN to the Training set
# n = 5 gives acc as 0.36, n = 10 acc as 0.37 n= 15 0.39, n=20 as 0.40
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 35, weights = 'uniform', metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn = pd.DataFrame(knn_classifier.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
cr_knn_all = classification_report(y_test, y_pred_knn)
acc_knn_all = accuracy_score(y_test, y_pred_knn)
cm_knn_all = confusion_matrix(y_test, y_pred_knn)
precision_knn_all = precision_score(y_test, y_pred_knn, average = 'weighted')
recall_knn_all = recall_score(y_test, y_pred_knn, average = 'weighted')
fscore_knn_all = f1_score(y_test, y_pred_knn, average = 'weighted')

#Lets try SVM
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'sigmoid')
classifier_svm.fit(X_train, y_train)
# Predicting the Test set results
y_pred_svm = pd.DataFrame(classifier_svm.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_svm_parenthood = classification_report(y_test, y_pred_svm)
acc_svm_parenthood = accuracy_score(y_test, y_pred_svm)
cm_svm_parenthood = confusion_matrix(y_test, y_pred_svm)
precision_svm_parenthood = precision_score(y_test, y_pred_svm, average = 'weighted')
recall_svm_parenthood = recall_score(y_test, y_pred_svm, average = 'weighted')
fscore_svm_parenthood = f1_score(y_test, y_pred_svm, average = 'weighted')


#lets take parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'sigmoid')
classifier_svm.fit(X_train, y_train)
# Predicting the Test set results
y_pred_svm = pd.DataFrame(classifier_svm.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_svm_pm = classification_report(y_test, y_pred_svm)
acc_svm_pm = accuracy_score(y_test, y_pred_svm)
cm_svm_pm = confusion_matrix(y_test, y_pred_svm)
precision_svm_pm = precision_score(y_test, y_pred_svm, average = 'weighted')
recall_svm_pm = recall_score(y_test, y_pred_svm, average = 'weighted')
fscore_svm_pm = f1_score(y_test, y_pred_svm, average = 'weighted')

#lets take all
X = hm_dataset.iloc[:, [0,1,2,3,4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'sigmoid')
classifier_svm.fit(X_train, y_train)
# Predicting the Test set results
y_pred_svm = pd.DataFrame(classifier_svm.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
cr_svm_all = classification_report(y_test, y_pred_svm)
acc_svm_all = accuracy_score(y_test, y_pred_svm)
cm_svm_all = confusion_matrix(y_test, y_pred_svm)
precision_svm_all = precision_score(y_test, y_pred_svm, average = 'weighted')
recall_svm_all = recall_score(y_test, y_pred_svm, average = 'weighted')
fscore_svm_all = f1_score(y_test, y_pred_svm, average = 'weighted')


#SGDClassifier
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SVM to the Training set
from sklearn.linear_model import SGDClassifier
classifier_sgd = SGDClassifier()
classifier_sgd.fit(X_train, y_train)
# Predicting the Test set results
y_pred_sgd = pd.DataFrame(classifier_sgd.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_sgd_parenthood = classification_report(y_test, y_pred_sgd)
acc_sgd_parenthood = accuracy_score(y_test, y_pred_sgd)
cm_sgd_parenthood = confusion_matrix(y_test, y_pred_sgd)
precision_sgd_parenthood = precision_score(y_test, y_pred_sgd, average = 'weighted')
recall_sgd_parenthood = recall_score(y_test, y_pred_sgd, average = 'weighted')


#lets take parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SVM to the Training set
from sklearn.linear_model import SGDClassifier
classifier_sgd = SGDClassifier()
classifier_sgd.fit(X_train, y_train)
# Predicting the Test set results
y_pred_sgd = pd.DataFrame(classifier_sgd.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_sgd_pm = classification_report(y_test, y_pred_sgd)
acc_sgd_pm = accuracy_score(y_test, y_pred_sgd)
cm_sgd_pm = confusion_matrix(y_test, y_pred_sgd)
precision_sgd_pm = precision_score(y_test, y_pred_sgd, average = 'weighted')
recall_sgd_pm = recall_score(y_test, y_pred_sgd, average = 'weighted')


#lets take all
X = hm_dataset.iloc[:, :-1].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SGD to the Training set
from sklearn.linear_model import SGDClassifier
classifier_sgd = SGDClassifier()
classifier_sgd.fit(X_train, y_train)
# Predicting the Test set results
y_pred_sgd = pd.DataFrame(classifier_sgd.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_sgd_all = classification_report(y_test, y_pred_sgd)
acc_sgd_all = accuracy_score(y_test, y_pred_sgd)
cm_sgd_all = confusion_matrix(y_test, y_pred_sgd)
precision_sgd_all = precision_score(y_test, y_pred_sgd, average = 'weighted')
recall_sgd_all = recall_score(y_test, y_pred_sgd, average = 'weighted')


#Let's try RandomForestClassifier
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting RandomForest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators = 25 )
classifier_rfc.fit(X_train, y_train)
# Predicting the Test set results
y_pred_rfc = pd.DataFrame(classifier_rfc.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_rfc_parenthood = classification_report(y_test, y_pred_rfc)
acc_rfc_parenthood = accuracy_score(y_test, y_pred_rfc)
cm_rfc_parenthood = confusion_matrix(y_test, y_pred_rfc)
precision_rfc_parenthood = precision_score(y_test, y_pred_rfc, average = 'weighted')
recall_rfc_parenthood = recall_score(y_test, y_pred_rfc, average = 'weighted')


#lets take parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators = 25 )
classifier_rfc.fit(X_train, y_train)
# Predicting the Test set results
y_pred_rfc = pd.DataFrame(classifier_rfc.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_rfc_pm = classification_report(y_test, y_pred_rfc)
acc_rfc_pm = accuracy_score(y_test, y_pred_rfc)
cm_rfc_pm = confusion_matrix(y_test, y_pred_rfc)
precision_rfc_pm = precision_score(y_test, y_pred_rfc, average = 'weighted')
recall_rfc_pm = recall_score(y_test, y_pred_rfc, average = 'weighted')


#lets take all
X = hm_dataset.iloc[:, :-1].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators = 25 )
classifier_rfc.fit(X_train, y_train)
# Predicting the Test set results
y_pred_rfc = pd.DataFrame(classifier_rfc.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_rfc_all = classification_report(y_test, y_pred_rfc)
acc_rfc_all = accuracy_score(y_test, y_pred_rfc)
cm_rfc_all = confusion_matrix(y_test, y_pred_rfc)
precision_rfc_all = precision_score(y_test, y_pred_rfc, average = 'weighted')
recall_rfc_all = recall_score(y_test, y_pred_rfc, average = 'weighted')



#Let's try DecisionTreeClassifier
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)
# Predicting the Test set results
y_pred_dt = pd.DataFrame(classifier_dt.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_dt_parenthood = classification_report(y_test, y_pred_dt)
acc_dt_parenthood = accuracy_score(y_test, y_pred_dt)
cm_dt_parenthood = confusion_matrix(y_test, y_pred_dt)
precision_dt_parenthood = precision_score(y_test, y_pred_dt, average = 'weighted')
recall_dt_parenthood = recall_score(y_test, y_pred_dt, average = 'weighted')


#lets take parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)
# Predicting the Test set results
y_pred_dt = pd.DataFrame(classifier_dt.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_dt_pm = classification_report(y_test, y_pred_dt)
acc_dt_pm = accuracy_score(y_test, y_pred_dt)
cm_dt_pm = confusion_matrix(y_test, y_pred_dt)
precision_dt_pm = precision_score(y_test, y_pred_dt, average = 'weighted')
recall_dt_pm = recall_score(y_test, y_pred_dt, average = 'weighted')


#lets take all
X = hm_dataset.iloc[:, :-1].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)
# Predicting the Test set results
y_pred_dt = pd.DataFrame(classifier_dt.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_dt_all = classification_report(y_test, y_pred_dt)
acc_dt_all = accuracy_score(y_test, y_pred_dt)
cm_dt_all = confusion_matrix(y_test, y_pred_dt)
precision_dt_all = precision_score(y_test, y_pred_dt, average = 'weighted')
recall_dt_all = recall_score(y_test, y_pred_dt, average = 'weighted')





#Let's try GaussianNB
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_nb = pd.DataFrame(classifier_nb.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_nb_parenthood = classification_report(y_test, y_pred_nb)
acc_nb_parenthood = accuracy_score(y_test, y_pred_nb)
cm_nb_parenthood = confusion_matrix(y_test, y_pred_nb)
precision_nb_parenthood = precision_score(y_test, y_pred_nb, average = 'weighted')
recall_nb_parenthood = recall_score(y_test, y_pred_nb, average = 'weighted')


#lets take parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_nb = pd.DataFrame(classifier_nb.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_nb_pm = classification_report(y_test, y_pred_nb)
acc_nb_pm = accuracy_score(y_test, y_pred_nb)
cm_nb_pm = confusion_matrix(y_test, y_pred_nb)
precision_nb_pm = precision_score(y_test, y_pred_nb, average = 'weighted')
recall_nb_pm = recall_score(y_test, y_pred_nb, average = 'weighted')


#lets take all
X = hm_dataset.iloc[:, :-1].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_nb = pd.DataFrame(classifier_nb.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_nb_all = classification_report(y_test, y_pred_nb)
acc_nb_all = accuracy_score(y_test, y_pred_nb)
cm_nb_all = confusion_matrix(y_test, y_pred_nb)
precision_nb_all = precision_score(y_test, y_pred_nb, average = 'weighted')
recall_nb_all = recall_score(y_test, y_pred_nb, average = 'weighted')


#Let's try LogisticRegression
#lets start with parenthood
X = hm_dataset.iloc[:, [4]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_lr = pd.DataFrame(classifier.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
cr_lr_parenthood = classification_report(y_test, y_pred_lr)
acc_lr_parenthood = accuracy_score(y_test, y_pred_lr)
cm_lr_parenthood = confusion_matrix(y_test, y_pred_lr)
precision_lr_parenthood = precision_score(y_test, y_pred_lr, average = 'weighted')
recall_lr_parenthood = recall_score(y_test, y_pred_lr, average = 'weighted')


#lets take parenthood and marital status
X = hm_dataset.iloc[:, [4,5,6,7,8]].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_lr = pd.DataFrame(classifier.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_lr_pm = classification_report(y_test, y_pred_lr)
acc_lr_pm = accuracy_score(y_test, y_pred_lr)
cm_lr_pm = confusion_matrix(y_test, y_pred_lr)
precision_lr_pm = precision_score(y_test, y_pred_lr, average = 'weighted')
recall_lr_pm = recall_score(y_test, y_pred_lr, average = 'weighted')


#lets take all
X = hm_dataset.iloc[:, :-1].values
y = hm_dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test = pd.DataFrame(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_lr = pd.DataFrame(classifier.predict(X_test))
# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
cr_lr_all = classification_report(y_test, y_pred_lr)
acc_lr_all = accuracy_score(y_test, y_pred_lr)
cm_lr_all = confusion_matrix(y_test, y_pred_lr)
precision_lr_all = precision_score(y_test, y_pred_lr, average = 'weighted')
recall_lr_all = recall_score(y_test, y_pred_lr, average = 'weighted')


plt.rcdefaults()
plt.plot([1, 2, 3], [acc_lr_parenthood, acc_lr_pm, acc_lr_all], label='Logistic Regression')
plt.plot([1, 2, 3], [acc_knn_parenthood, acc_knn_pm, acc_knn_all], label= 'K Nearest Neighbors')
plt.plot([1, 2, 3], [acc_svm_parenthood, acc_svm_pm, acc_svm_all], label= 'SVM')
plt.plot([1, 2, 3], [acc_sgd_parenthood, acc_sgd_pm, acc_sgd_all], label='SGD Classifier')
plt.plot([1, 2, 3], [acc_dt_parenthood, acc_dt_pm, acc_dt_all], label='Decision Tree Classifier')
plt.plot([1, 2, 3], [acc_rfc_parenthood, acc_rfc_pm, acc_rfc_all], label= 'Random Forest')
plt.plot([1, 2, 3], [acc_nb_parenthood, acc_nb_pm, acc_nb_all], label='GaussianNB')

plt.xticks([1, 2, 3], ['With\nOnly\nParenthood', 'With\nOnly\nParenthoodAndMarital', 'With\nAll\nDimensions'])
plt.ylabel('Accuracy')
plt.title('Comparision of Classification Algorithms')
plt.legend()
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
