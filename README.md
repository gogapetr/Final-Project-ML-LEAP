# Final-Project-ML-LEAP

A public agency wants to use survey data to flag which households are likely to
be in a higher-income bracket, so that a benefit or outreach programme can be
targeted. Does a simple, transparent model whose decisions can be explained to
the public beat a more complex neural network that might, or might not,
be more accurate?

## Project title and short description

Title: Does a neural network beat a classical model?

Description:

This is the final project of "Machine Learning and applications", an online course by
"LEAP: unLocking carEer potentiAl with comPlex systems, data analytics and machine
learning", an Erasmus+ project that aims at reducing the ICT skills gap, in the fields
of Data Analytics and Machine Learning, by developing and offering a flexible and
personalised learning content for diverse learners.

For my final project I chose "Option 2: Does a neural network beat a classical model?".
In this scenario, an agency wants to use survey data to predict which individuals
are in a higher-income bracket (>50K). The end goal was to use the predictions for a
benefit or outreach programme. Two kinds of models were on the table: a simple,
transparent one whose decisions can be explained to the public, and a more complex
neural network that might, or might not, be more accurate.

My task was to build both models, compare them fairly, and judge whether the added
complexity earns its place. The code in Part A was provided, but Part B, Part C and
Part D were submited and graded.

## Problem statement: what you tried to predict, classify, cluster, or explain

This a classification problem. I solved it using two approaches:
logistic regression (Part B) and a small neural network (Part C).
Finally, we compared them (Part D).
The task was to predict whether a person’s income exceeds $50,000 per year.

## Data set: where it came from and what the main variables or inputs represent

We used the Adult Census Income data set created by Barry Becker and Ronny Kohavi.
<https://doi.org/10.24432/C5XW20>.
Extraction was done by Barry Becker from the 1994 Census database.  
A set of reasonably clean records was extracted using the following conditions:
((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

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

#### age: continuous

#### workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked

#### fnlwgt (final weight): continuous

#### education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool

#### education-num: continuous

#### marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse

#### occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces

#### relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried

#### race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black

#### sex: Female, Male

#### capital-gain: continuous

#### capital-loss: continuous

#### hours-per-week: continuous

#### native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands

## Method: the main model or models used, including any preprocessing steps

### Part A: Set up and Data

The first part was provided for the purpose of this final project.
It handles the import, the download, the target encoding, the train and test split, and the
detection of which columns are numeric and which are categorical.

### Part B: Logistic Regression

As the classical baseline I used a logistic regression. Previously we split the data set into
training data (80%) and test data (20%) using stratification and random_state=42. Also, I sorted
columns into a numeric group and a categorical group. Now I was ready to start processing data.
For numeric features process involved using a median imputation and standard scaling, while for
categorical features using most-frequent value imputation and one-hot encoding.
Unknown categories were ignored, and the encoded data was converted to a dense numeric array.
The preprocessing and logistic-regression model were combined in a scikit-learn pipeline. I
evaluated the model using five-fold stratified cross-validation with F1 as the scoring metric,
then evaluated accuracy and F1 on the held-out test set. Finally, I drew the confusion matrix
for the classical model.

### Part C: Neural Network

I built a dense network using the Keras Sequential API with an input layer, a
64-unit ReLU, a hidden layer, 30% dropout, a 32-unit ReLU hidden layer, and a one-unit
sigmoid output layer. The sigmoid output produces the probability that income is above
$50,000.

The network was compiled with the Adam optimizer, binary cross-entropy loss, and
accuracy as an additional metric. It was trained for up to 20 epochs with a batch size of 128.
There was an explicit 10% validation set. Early stopping monitored validation
loss, used a patience of three epochs, and restored the best model weights.

### Part D: The Comparison

Both models were evaluated on the same held-out test set. Accuracy and F1 score
were reported, with F1 treated as the main metric because the higher-income
class is the positive minority class. Confusion matrices were also used to
compare false positives and false negatives. Ethical considerations were addressed.

## Results: the most important numbers, plots, and observations

Logistic Regression: Accuracy = 0.8524 and F1 score = 0.6562
Neural Network: Accuracy = 0.8561 and F1 score = 0.6744

On the held-out test, the logistic regression achieved an accuracy score of 0.8524 and a F1
score of 0.6552. The neural network achieved an accuracy score of 0.8561 and a F1 score of
0.6744. F1 is the more informative metric than accuracy, because the higher-income class(50K)
is the minority class.

Overall, the neural network performs slightly higher on this test set. The neural network is a
more complex model that allowed for a slightly more improved predictive performance. The
confusion matrices show that both models are making a similar number of mistakes. It is
noteworthy that the neural network correctly identifies slightly more cases where people have
an income of >50K and makes fewer false positives.

## Interpretation: what the results mean, including limitations

While the neural network model is only slightly better than the logistic regression, the
difference in predictive performance does not change the fact that the simpler and more
transparent model is still a strong baseline.

The improvement in F1 may indicate improved identification of the positive class, because of
some additional patterns in the data that the neural model is picking up. Ultimately, the modest
improvement is not enough to justify the usage of such a complex model. Especially, since the
current setting values both transparency and explainability.

Considering the potential applications of the model, it is crucial to underline the fact that
the model's predictions will inform real-world decisions about benefits and outreach. The
logistic regression might perform worse, but the upside is that its decisions are easier to
understand and explain. The observed difference in predictive performance between logistic
regression and neural network is not significant enough to justify using a more complex model.
This conclusion is also supported by the confusion matrices.

### Limitations

1. Imbalanced Dataset: The majority class contributes more examples to the training data and
overall loss. Accuracy is important in the case of a reasonably balanced data set. Here,
accuracy can be misleading here, because the model could be performing poorly on the minority
class.

2. Low F1 scores: The F1 scores alone do not establish readiness for deployment; further
validation and consideration of application-specific costs are needed.

3. Ethical Concerns: A model does not consider matters of fairness or policy questions.
Undoubtedly, structural inequalities are reflected in features of the data set (e.g. race, sex)
and the deployment of a model trained on such data raises ethical concerns.

## Reflection: what you would improve with more time

If I had more time, I would prioritise addressing the limitations listed above as best as I
could. It is important to improve the model by making it more balanced, more reliable, and more
fair. Performance could benefit from a more balanced data set and precision-recall curves and
F1/recall tradeoff analysis. I would also investigate whether the neural network's small
advantage is consistent or just random. This requires actions, like a closer look at mean and
standard deviation across runs and repeated cross-validation. Finally, I would also examine
more closely the results for structural inequalities. For example, whether the model is
systemically missing eligible households or whether certain sensitive features, like race, sex
and age, are responsible for unfair predictions.

Overall, the results indicate that the neural network is better, but not decisively so. Because
this comparison is based on one run, repeated training runs and additional validation are
needed to determine whether the advantage is consistent. The logistic regression is still
standing as an attractive and trasparent alternative. Ultimately, both models need further
improvement and refinement before being used in a real desicion-support system.
