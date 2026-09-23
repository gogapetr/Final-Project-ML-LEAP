# Final-Project-ML-LEAP

A public agency wants to use survey data to flag which households are likely to
be in a higher-income bracket, so that a benefit or outreach programme can be
targeted. Does a simple, transparent model whose decisions can be explained to
the public beat a more complex neural network that might, or might not,
be more accurate?

## Project title and short description

Title: ???

Description:
This is the final project of "Machine Learning and applications", an online course by LEAP: unLocking carEer potentiAl with comPlex systems, data analytics and machine learning, an Erasmus+ project that aims at reducing the ICT skills gap, in the fields of Data Analytics and Machine Learning, by developing and offering a flexible and personalised learning content for diverse learners.
For my final project I choose "Option 2: Does a neural network beat a classical model?".
Thi is the scenario: A public agency wants to use survey data to flag which households are likely to be in a higher-income bracket, so that a benefit or outreach programme can be targeted. Two kinds of models are on the table: a simple, transparent one whose decisions can be explained to the public, and a more complex neural network that might, or might not, be more accurate.
My task is to build both models, compare them fairly, and judge whether the added complexity earns its place. The code in Part A is provided, but Part B and Part C will be submited and graded.

## Problem statement: what you tried to predict, classify, cluster, or explain

We are working with a classification problem and we will solve it using two approaches:
logistic regression (Part B) and a small neural network (Part C). 
Prediction task is to determine whether a person's income is over $50,000 a year.

## Data set: where it came from and what the main variables or inputs represent

We are using Adult Census Income data set created by Barry Becker and Ronny Kohavi.
<https://doi.org/10.24432/C5XW20>.
Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

### Dataset Characteristics

Multivariate

### Subject Area

Social Science

### Associated Tasks

Classification

### Feature Type

Categorical, Integer

### Has Missing Values?

Yes

### Instances

48842

### Features

14

### Listing of attributes

>50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Method: the main model or models used, including any preprocessing steps

Part A: the import, the download, the target encoding, the train and test split, and the detection of which columns are numeric and which are categorical

## Results: the most important numbers, plots, and observations

Results

## Interpretation: what the results mean, including limitations

Interpreting...

## Reflection: what you would improve with more time

Reflecting