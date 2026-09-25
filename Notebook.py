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
print(X.head())

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

clf_acc = round(accuracy_score(y_test, y_predict), 4)
clf_f1 = round(f1_score(y_test, y_predict), 4)

print("Accuracy:", clf_acc)
print("F1 score:", clf_f1)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_predict,
    display_labels=["<=50K", ">50K"],
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.show(block=False)
plt.pause(1)
plt.close()

## Part C. The neural network

# Before building the neural network, make a prediction.
    # Do you expect its F1 score to be higher than, lower than, 
    # or similar to that of logistic regression? Do you expect a large or
    # a small difference? Briefly explain your reasoning.

    # MY PREDICTION: I expect the F1 score to be similar to that of logistic
    # regression, with only a small difference. Logistic regression is already a
    # strong baseline for this tabular data after numeric scaling and categorical
    # one-hot encoding. The neural network may learn some nonlinear interactions,
    # but its improvement is unlikely to be large because the input representation
    # and the underlying task are well suited to a linear model.


# A neural network needs a plain numeric array as input, so I apply the 
# preprocessor to turn the mixed table into numbers. I fit the preprocessor 
# on the training data only, then transform both sets, so no information 
# leaks from the test set.

X_train_prep = preprocess.fit_transform(X_train)   # fit on training data only
X_test_prep  = preprocess.transform(X_test)        # reuse the same fitting

print("Training shape:", X_train_prep.shape)

# Building a dense network with the Keras Sequential API.
    # - an Input layer whose shape is the number of columns in X_train_prep
    #     (X_train_prep.shape[1])                                 
    # - a Dense hidden layer (try 64 units, relu)
    # - a Dropout layer (0.3) to reduce overfitting
    # - a second Dense hidden layer (try 32 units, relu)
    # - a Dense output layer with 1 unit and a sigmoid activation

net = keras.Sequential([
    layers.Input(shape=(X_train_prep.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

net.summary()

# Compiling net with:
    # optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
# Then training it with:
    # an explicit 10% validation set, an EarlyStopping callback (monitor "val_loss",
    # patience about 3, restore_best_weights=True), up to about 20 epochs,
    # and batch_size=128. Store the result in history.

net.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = net.fit(
    X_train_prep, 
    y_train,
    validation_split=0.1,
    callbacks=[early_stopping],
    epochs=20,
    batch_size=128
)

# Producing predictions and evaluating the network.
# Then computing nn_acc and nn_f1, printing them, and 
# drawing the confusion matrix.

nn_prob = net.predict(X_test_prep).ravel()     # probabilities
nn_pred = (nn_prob > 0.5).astype(int)           # threshold at 0.5

nn_acc = round(accuracy_score(y_test, nn_pred), 4)
nn_f1 = round(f1_score(y_test, nn_pred), 4)

print("Accuracy NN:", nn_acc)
print("F1 Score NN:", nn_f1)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    nn_pred,
    display_labels=["<=50K", ">50K"],
    cmap="Blues"
)

plt.title("Confusion Matrix - Neural Network")
plt.show()

# Part D. The comparison
# Now we answer the central question. We have four numbers: the accuracy and 
# F1 of the classical model, and the accuracy and F1 of the network, all 
# measured on the same test set. Because the higher-income class is the 
# minority class, use F1 as your main comparison metric and accuracy as 
# supporting information. 

print("Model Comparison:")
print(
    f"Logistic Regression F1: {clf_f1:.4f}"
    f"Logistic Regression Accuracy: {clf_acc:.4f}"
)
print(
    f"Neural Network F1: {nn_f1:.4f}"
    f"Neural Network Accuracy: {nn_acc:.4f}"
)

#### Ethical considerations

# There are a few ethical risks we have to consider before deploying this model.
# Firstly, a model that uses characteristics like sex, race, and nationality in 
# order to predict their income bracket is incredibly sensitive to reproducing 
# existing inequalities. Ensuring equity and fairness is a priority. A failed 
# prediction means either someone eligible not benefitting from the programme, 
# or someone ineligible benefiting. Those# mistakes have great real-world cost 
# for people, organisations, and gonverment, defeating also the purpose of the 
# programme and wasting important funds.

# In my opinion, those mentioned considerations make a simpler, more transparent 
# model is more preferable to an opaque one when decisions affect people.

###3 Wrapping up
#Bring your findings together. Write a short paragraph that answers the central question directly:

# Which model had the higher F1 score, and by how much? Did accuracy lead to the same conclusion?
# Why would accuracy alone be insufficient for this data set?
# Use the confusion matrices to compare false negatives and false positives. 
# Which model identified the higher-income class more successfully?
# The network has far more moving parts than logistic regression. 
# On this data set, was that extra complexity worth it?
# Considering the size and structure of this tabular data set, suggest one reason why one model
#  may have performed better than the other.
# What to hand in: this completed notebook, with both confusion matrices 
# and the Part D comparison visible in the output, your prediction in Part C, 
# and your written answer in this cell.

# (Replace this line with your paragraph.)