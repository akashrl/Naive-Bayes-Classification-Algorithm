# Naive-Bayes-Classification-Algorithm

See .py files under Q2 and Q3 for algorithm implementation.

See ***NBC_Proj.pdf*** for full project and algorithm details.

In this programming assignment I have implemented a naive Bayes classification algorithm
and evaluated it on a modified Yelp dataset.

**Note:** This was implemented from scratch in Python. No models already implemented in machine learning libraries (e.g., scikit-learn) were used. 

## Dataset Details:
The pre-processed yelp data.csv is a datafile which has 24, 813 instances in rows and 20 attributes. Here, 4 attributes are multi-valued: ({ambience, parking, dietaryRestrictions,
recommendedFor}).

## Algorithm Details:
This implements algorithms to learn (i.e., estimate) and apply (i.e., predict) the NBC.

## Pre-processing
**The following steps have been followed:**

• Binary features (columns) should be created for every value corresponding to multivalued attributes ({ambience, parking, dietaryRestrictions, recommendedFor}).
For example: The attribute ”recommendedFor’ can take the following multiple values for a data instance: (’breakfast’,’lunch’,’brunch’,’dinner’,’latenight’,’dessert’). The column corresponding to ”recommendedFor” should be replaced by 6 new columns having binary indicator (True/False) illustrating whether that value is true for that data instance.

• Missing value treatment in attributes was done by adding ”None”.

• If any further pre-processing of the data is required or used, it should be clearly
mentioned in the report.

## Features:
All the attributes except the last in the pre-processed data were used as features for NBC .i.e. X.

## Class label
The attribute outdoorSeating was the class label, i.e., Y = {1, 0}.

## Smoothing:
**Laplace smoothing** was implemented in the parameter estimation. For an attribute Xi with k values, Laplace correction adds 1 to the numerator and k to the denominator of the maximum likelihood estimate for P(Xi = xi
| Y ).

## Evaluation
When the learned NBC was applied to predict the class labels of the examples in the test set, (i) zero-one loss, and (ii) squared loss were used to evaluate the predictions.
Let y(i) be the true class label for example i and let ˆy(i) be the prediction for i. Let pi refer to the probability that the NBC assigns to the true class of example i (i.e., pi := p(ˆy(i) = y(i)). If pi ≥ 0.5, the prediction for example i will be correct (i.e., ˆy(i) = y(i)), but otherwise if pi < 0.5, the prediction will be incorrect (i.e., ˆy(i) 6= y(i)).
