# Part A. Setup and data
    # This first part was provided for the purpose of this final project. 
    # It handles the import, the download, the target encoding, the train 
    # and test split, and the detection of which columns are numeric and 
    # which are categorical.


## loading everything we will need
    # I start with the imports. 
    # This cell loads everything the notebook needs.
    

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
    # I download the Adult data set from OpenML. The features arrive in `X` as a table, 
    # and the target in `y` as text, either `>50K` or `<=50K`. The download runs once 
    # and may take a moment.

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

print("Rows and columns:", X.shape)
print("Target values:", y.value_counts().to_dict())
X.head()

## encoding the target
    # The target is text with two values. I convert it to numbers so both models
    # can use it. LabelEncoder maps the two classes to 0 and 1 in alphabetical 
    # order, so <=50K becomes 0 and >50K becomes 1. The higher income class is 
    # therefore the positive class, labelled 1.

y = LabelEncoder().fit_transform(y)
print("Positive rate (share earning >50K):", round(y.mean(), 3))

## train and test split
    # I split the data into a training set and a test set before doing anything 
    # else, and I keep the class balance with stratify. Both models are trained
    # on the training set and judged on the same held-back test set, so the 
    # comparison is fair.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training rows:", X_train.shape[0])
print("Test rows:    ", X_test.shape[0])   

##  numeric and categorical columns
    # These two kinds of feature need different preprocessing, so I sort the 
    # columns into a numeric group and a categorical group. 
    # I detect them from the column data types.

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Part B. The classical mode.
    
# Preprocessor with a ColumnTransformer.

# For the NUMERIC columns, I chain two steps in a Pipeline:
    #- SimpleImputer to fill missing values (strategy="median")
    #- StandardScaler to put the features on the same scale

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# For the CATEGORICAL columns, I chain two steps in a Pipeline:
    #- SimpleImputer to fill missing values (strategy="most_frequent")
    #- OneHotEncoder to turn categories into numbers. Set
    #   -handle_unknown="ignore", so unseen categories in the test set do not error
    #   -sparse_output=False, so the result is a plain array the network can use 
    #    later

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Then combine the two with a ColumnTransformer, applying each pipeline
# to its own list of columns (numeric_features and categorical_features).

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("categ", categorical_pipe, categorical_features),
        ],
        remainder="drop",
)

# I combine the preprocessor and a LogisticRegression into one Pipeline,
# so that preprocessing is fitted only on the training data inside each fold.
# LogisticRegression(max_iter=1000) gives it enough iterations to converge.
.

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])
print(pipe)

# I run a 5-fold cross-validation on the TRAINING data, scoring by F1 and
# use StratifiedKFold(n_splits=5, shuffle=True, random_state=42).

cv = StratifiedKFold(
    n_splits=5, 
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print("F1 Score:", cv_scores)
print("Mean F1:", cv_scores.mean().round(4))
print("Standard Deviation:", cv_scores.std().round(4))

# Fitting the classifier(clf) on the training data, then predict on 
# the test data. I store the results so I can compare them later.
# Then I draw the confusion matrix for the classical model.

pipe.fit(
    X_train,
    y_train
)
y_predict = pipe.predict(X_test)

clf_acc = accuracy_score(y_test, y_predict).round(4)
clf_f1 = f1_score(y_test, y_predict).round(4)

print("Accuracy:", clf_acc)
print("F1 score:", clf_f1)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_predict,
    display_labels=["<=50K", ">50K"],
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.show()