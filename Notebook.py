  # Part A. Setup and data
  # This first part was provided for the purpose of this final project. 
  # It handles the import, the download, the target encoding, the train 
  # and test split, and the detection of which columns are numeric and 
  # which are categorical.


## loading everything we will need
# We start with the imports. This cell loads everything the notebook needs, 
# so we run it once at the top.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
                            

import keras
from keras import layers

## loading the data
# We download the Adult data set from OpenML. The features arrive in `X` as a table, 
# and the target in `y` as text, either `>50K` or `<=50K`. The download runs once 
# and may take a moment.

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

print("Rows and columns:", X.shape)
print("Target values:", y.value_counts().to_dict())
X.head()

## encoding the target
# The target is text with two values. We convert it to numbers so both models can use it. 
# LabelEncoder maps the two classes to 0 and 1 in alphabetical order, so <=50K becomes 0 
# and >50K becomes 1. The higher income class is therefore the positive class, labelled 1.

y = LabelEncoder().fit_transform(y)
print("Positive rate (share earning >50K):", round(y.mean(), 3))

## train and test split
# We split the data into a training set and a test set before doing anything else, and 
# we keep the class balance with stratify. Both models are trained on the training set 
# and judged on the same held-back test set, so the comparison is fair.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training rows:", X_train.shape[0])
print("Test rows:    ", X_test.shape[0])   

##  numeric and categorical columns
# The two kinds of feature need different preprocessing, so we sort the columns 
# into a numeric group and a categorical group. We detect them from the column 
# data types.

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

print("Numeric features:    ", numeric_features)
print("Categorical features:", categorical_features)


    # Part B. The classical mode.
    # My first task is to build a classical model that will make predictions about 
    # which households belond in a higher-income bracket, so that a benefit ot 
    # outreach programme can be targeted. 
    


