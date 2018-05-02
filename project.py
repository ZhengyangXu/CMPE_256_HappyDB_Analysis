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

#Number of countries
len(dataset_2.country.unique())
dataset_2.marital.unique()
len(dataset_2.marital.unique())
#We can impute nan values using mode
dataset_2.parenthood.unique()
#We can impute nan values using mode
dataset_2.gender.unique()
#We can impute nan values using mode
dataset_2.count()
#Merging the two data sets
hm_new = pd.merge(dataset_1 , dataset_2)
hm_new.head()
#dropping columnss which won't be needed
hm_new.drop(['original_hm', 'cleaned_hm','modified','num_sentence','ground_truth_category','reflection_period','hmid', 'wid'], axis=1, inplace=True)
hm_new.head()
hm_new.count()
#Arranging the matrix and put dependent variable in the end
#using pd
hm_new = hm_new[['age', 'gender', 'marital', 'parenthood', 'country', 'predicted_category']]
#Using np arrays
hm_dataset = hm_new.as_matrix()
hm_dataset = np.append(arr = hm_dataset[:, 1:], values = hm_dataset[:, [0]], axis = 1)

hm_dataset
print(type(hm_dataset))

hm_dataset.dtypes
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
hm_dataset[:, 1] = labelencoder.fit_transform(hm_dataset[:, 1])


imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
hm_dataset['gender'] = pd.DataFrame(imp.fit_transform(hm_dataset['gender']))
#imputed_DF.columns = dataset_2.columns
#imputed_DF.index = dataset_2.index

X = hm_dataset[:, :5]
y = hm_dataset[:, 5]


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X
print(type(X))
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])