
  # Part A. Setup and data
  # This first part was provided for the purpose of this final project. 
  # It handlse the import, the download, the target encoding, the train 
  # and test split, and the detection of which columns are numeric and 
  # which are categorical.
  
  ## loading everything we will need
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
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

print("Rows and columns:", X.shape)
print("Target values:", y.value_counts().to_dict())
X.head()
  
  ##  encoding the target
y = LabelEncoder().fit_transform(y)
print("Positive rate (share earning >50K):", round(y.mean(), 3))

  ##  train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training rows:", X_train.shape[0])
print("Test rows:    ", X_test.shape[0])

  ##  numeric and categorical columns
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

print("Numeric features:    ", numeric_features)
print("Categorical features:", categorical_features)


  #Part B. The classical mode.
  # My first task is to build a classical model that will make predictions about 
  # which households belond in a higher-income bracket, so that a benefit ot 
  # outreach programme can be targeted. 
  
  # first time making a python script, kinda nervous (˶˃ ᵕ ˂˶)
