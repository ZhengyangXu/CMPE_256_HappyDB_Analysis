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

#Get dummy variables for each encoded column
hm_imputed_dataset = pd.get_dummies(hm_dataset, columns=["gender", "parenthood","marital"], prefix=["gender", "parenthood","marital"])

#Re-arrange dataframe and put dependent variable in the end
hm_dataset = hm_imputed_dataset[['country', 'gender_0', 'gender_1',
       'gender_2', 'parenthood_0', 'parenthood_1', 'marital_0', 'marital_1',
       'marital_2', 'marital_3', 'marital_4','predicted_category']]

#Preventing dummy variable trap
hm_dataset = hm_imputed_dataset[['country', 'gender_0', 'gender_1',
       'parenthood_0', 'marital_0', 'marital_1',
       'marital_2', 'marital_3','predicted_category']]

#Age is having some noise as strings and float, convert everything to int
hm_age = pd.Series(hm_imputed_dataset.age)
hm_age = pd.to_numeric(hm_age, errors='coerce')
hm_dataset = pd.concat( [ hm_age, hm_dataset ] , axis=1 )
#Fill nan with mean
hm_dataset['age'] = hm_dataset.age.fillna(int(hm_dataset.age.mean()))
hm_dataset = hm_dataset[(hm_dataset['age'] >= 10) & (hm_dataset['age'] <= 100) ]
hm_dataset.count()

#Step 2: Splitting data into test and train
#Split independent and dependent variables as X and y
X = hm_dataset.iloc[:, :-1].values
y = hm_dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
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
y_pred = pd.DataFrame(classifier.predict(X_test))

# Metrics for the algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
cr_lr = classification_report(y_test, y_pred)
acc_lr = accuracy_score(y_test, y_pred)
cm_lr = confusion_matrix(y_test, y_pred)
precision_lr = precision_score(y_test, y_pred, average = 'weighted')
recall_lr = recall_score(y_test, y_pred, average = 'weighted')

#Since the logistic regression is only able to predict 2 values ie since it is 
# a binary classifier, it is not suitable for our problem statement

#Lets go for another algorithm which can classify non binary variables
# Fitting K-NN to the Training set
# n = 5 gives acc as 0.36, n = 10 acc as 0.37 n= 15 0.39, n=20 as 0.40
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 40, weights = 'uniform', metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn = pd.DataFrame(knn_classifier.predict(X_test))
cr_knn = classification_report(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average = 'weighted')
recall_knn = recall_score(y_test, y_pred_knn, average = 'weighted')


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

# Predicting the Test set results
y_pred_nb = pd.DataFrame(classifier_nb.predict(X_test))
y_pred_nb
cr_nb = classification_report(y_test, y_pred_nb)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'sigmoid')
classifier_svm.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svm = pd.DataFrame(classifier_svm.predict(X_test))
y_pred_svm
cr_svm = classification_report(y_test, y_pred_nb)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)







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

#Get dummy variables for each encoded column
hm_imputed_dataset = pd.get_dummies(hm_dataset, columns=["gender", "parenthood","marital"], prefix=["gender", "parenthood","marital"])

#Re-arrange dataframe and put dependent variable in the end
hm_dataset = hm_imputed_dataset[['country', 'gender_0', 'gender_1',
       'gender_2', 'parenthood_0', 'parenthood_1', 'marital_0', 'marital_1',
       'marital_2', 'marital_3', 'marital_4','predicted_category']]

#Preventing dummy variable trap
hm_dataset = hm_imputed_dataset[['country', 'gender_0', 'gender_1',
       'parenthood_0', 'marital_0', 'marital_1',
       'marital_2', 'marital_3','predicted_category']]

#Age is having some noise as strings and float, convert everything to int
hm_age = pd.Series(hm_imputed_dataset.age)
hm_age = pd.to_numeric(hm_age, errors='coerce')
hm_dataset = pd.concat( [ hm_age, hm_dataset ] , axis=1 )
#Fill nan with mean
hm_dataset['age'] = hm_dataset.age.fillna(int(hm_dataset.age.mean()))
hm_dataset = hm_dataset[(hm_dataset['age'] >= 10) & (hm_dataset['age'] <= 100) ]
hm_dataset.count()
