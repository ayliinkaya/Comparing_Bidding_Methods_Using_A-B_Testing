# Comparing Bidding Methods Using A/B Testing

# What is A/B testing?

# A/B tests compare the performance of two content versions to see which one appeals more to users.
# It compares the control group (A) with the test group (B) to measure the most successful based on your key metrics.

# Let's make an AB test example using a real dataset.

# Business Problem

# Suppose we have a business problem as below:

# One of our customers wants to test the "average bidding" feature introduced by Facebook, which is an alternative to the
# bidding type called "maximum bidding". Our client wants to do an A/B test to see if the average bidding yield is more
# than the maximum bidding.

# The criterion of success for our client is 'Purchase'. Therefore, the focus should be on the 'Purchase' metric
# for statistical testing.

# Dataset Story

# In this dataset, which includes the website information of a company, there is information such as the number of
# advertisements that users see and click, as well as earnings information from here. There are two separate data sets,
# the control and test groups. Maximum Bidding was applied to the control group and Average Bidding was applied to the test group.

# Features

# Impression: Number of ad views
# Click: Number of clicks on the displayed ad
# Purchase: Number of products purchased after ads clicked
# Earning: Earnings after purchased products

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Control Group (Maximum Bidding)

A = pd.read_excel("ab_testing/ab_testing.xlsx",sheet_name="Control Group")

# Test Group (Average Bidding)

B = pd.read_excel("ab_testing/ab_testing.xlsx",sheet_name="Test Group")

# Analysis of control and test group data

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(A)

check_df(B)

# Joining control and test group data with the 'concat method'

A.columns = [col + "(A)" for col in A]
B.columns = [col + "(B)" for col in B]

df = pd.concat([A, B], axis=1)

df.head()

# Defining the A/B Test Hypothesis

# H0 : M1 = M2 (There isn't a statistically significant difference in purchase averages between maximum bidding and average bidding)
# H1 : M1!= M2 # H1 : M1!= M2 (There is a statistically significant difference in purchase averages between maximum bidding and average bidding)

# Analysis of purchase averages for the control and test group

df[["Purchase(A)", "Purchase(B)"]].mean()

# There seems to be a mathematical difference between the purchase averages of the control group and the test group.
# However, this difference may be due to chance.

# Assumption of Normality

# H0: Normal distribution assumption is provided.
# H1: The assumption of normal distribution is not provided.

# Performing Hypothesis Testing

# Shapiro-Wilk Test for Control Group (Maximum Bidding):

test_stat, pvalue = shapiro(df["Purchase(A)"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 CANNOT BE REJECTED since p-value = 0.5891 > 0.05. Therefore, the assumption of normal
# distribution is provided for the control group.

# Shapiro-Wilk Test for Test Group (Average Bidding):

test_stat, pvalue = shapiro(df["Purchase(B)"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 CANNOT BE REJECTED since p-value = 0.1541 > 0.05. Therefore, the assumption of normal distribution is provided for the test group.

# The Assumption of Homogeneity of Variance

# H0: Variances are homogeneous.
# H1: Variances are not homogeneous.

test_stat, pvalue = levene(df["Purchase(A)"], df["Purchase(B)"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 CANNOT BE REJECTED since p-value = 0.1083 > 0.05. Therefore, the variances are homogeneous.

# Since both the assumption of normality and homogeneity of variance are provided, we apply the parametric test, namely
# ttest, by making equal_var=True:

test_stat, pvalue = ttest_ind(df["Purchase(A)"], df["Purchase(B)"], equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 CANNOT BE REJECTED since p-value = 0.3493 > 0.05. Therefore, There isn't a statistically significant difference in
# purchase averages between maximum bidding and average bidding. The difference between the values of 550,89406 and 582,10610
# obtained as a result of the purchase average analysis (with mean method) for the control and test group is purely by chance.

# Note: In this project, Two-Sample Independent t-Test was used as it was desired to make a comparison between the mean
# of the 'Purchase' variables of the control group and test group.