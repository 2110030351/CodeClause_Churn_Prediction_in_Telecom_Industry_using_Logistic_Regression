#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction in Telecom Industry using Logistic Regression

# # Problem Statement:
# 
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# For many incumbent operators, retaining high profitable customers is the number one business goal.
# 
# To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# 
# In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

# # Methodology
# 1. Data understanding
# 2. Data cleaning
# 3. Visualization
# 4. Data preparation
# 5. Model Building                
# 
#     5.1 Model with interpretability
#     
#     5.2 Model with good prediction
#     
# ![image.png](attachment:image.png)

# # Loading the dataset

# In[1]:


import pandas as pd
tel_1 = pd.read_csv(r"C:\Users\manas\Downloads\archive (3)\churn-bigml-80.csv")
tel_2 = pd.read_csv(r"C:\Users\manas\Downloads\archive (3)\churn-bigml-20.csv")

# Exploring the dataset
# Returns number of rows and columns of the dataset
# load all dataset into a DataFrame
telcom = pd.concat([tel_1, tel_2], ignore_index=True)
print(telcom.shape)


# # Importing necessary libraries

# In[2]:


import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


telcom.shape


# In[4]:


telcom.head()


# In[102]:


telcom.info()


# # accessing Churn feature

# In[155]:


telcom['Churn'].head(10)


# Descriptive Analysis and Data Visualization

# In[156]:


telcom.describe()


# # Count the number of data points in each category
# 

# In[157]:


y = telcom['Churn'].value_counts()
y


# # Create the pie chart
# 

# In[158]:


plt.pie(y, labels=y.index, autopct='%1.1f%%')
plt.title('Distribution of Churn')
plt.legend(title='Churn')
plt.show()


# In[159]:


sns.barplot(x=y.index, y=y.values)


# # Statistics for both the classes
# #Group telcom by 'Churn' and compute the mean
# 

# In[160]:


telcom.groupby(['Churn']).mean()


# In[166]:


telcom.groupby(['Churn']).std()


# In[ ]:





# # Exploring feature distributions
# 

# In[172]:


# visualize the distribution of 'Account length'
sns.distplot(telcom['Account length'])

# display the plot
plt.show()

sns.distplot(telcom['Total day minutes'])
plt.show()

sns.distplot(telcom['Total eve minutes'])
plt.show()

sns.distplot(telcom['Total intl minutes'])
plt.show()


# # Check for missing values
# 

# In[174]:


has_missing = telcom.isnull().any()
has_missing


# # check for duplicate rows 
# 

# In[176]:


duplicate_rows = telcom[telcom.duplicated()]
duplicate_rows


# In[177]:


telcom.head()


# In[178]:


telcom.dtypes


# In[7]:


# Encoding binary features
# Convert the boolean values to integers
bool_columns = telcom.select_dtypes(include=['bool']).columns
print(bool_columns)

# Find the columns of object type
object_columns = telcom.select_dtypes(include=['object']).columns
print(object_columns)


# In[8]:


telcom[bool_columns] = telcom[bool_columns].astype(int)
# Replace 'no' with 0 and 'yes' with 1 in 'International plan' and 'Voice mail plan'
telcom[['International plan','Voice mail plan']] = telcom[['International plan','Voice mail plan']].apply(lambda x: x.map({'No': 0, 'Yes': 1}))
# see the results
telcom[['International plan','Voice mail plan','Churn']].head()


# In[9]:


# Feature selection and engineering


# In[12]:


# drop 'State' feature
telcom = telcom.drop(telcom[['state']], axis=1)

# Calculate the correlation matrix
corr_matrix = telcom.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)

# Drop the correlated features from the dataset
telcom = telcom.drop(telcom[to_drop], axis=1)

telcom.head()


# In[134]:


# Feature scaling



# In[13]:


telcom['Total intl calls'].describe()


# In[14]:


telcom['Total night minutes'].describe()


# In[137]:


# from sklearn.preprocessing import StandardScaler


# In[15]:


# Scale telcom using StandardScaler
features_to_scale = [column for column in telcom.columns if column not in ['International plan','Voice mail plan','Churn']]
# print(features_to_scale)
telcom_scaled = StandardScaler().fit_transform(telcom[features_to_scale])

# Add column names back for readability
telcom_scaled_df = pd.DataFrame(telcom_scaled, columns=features_to_scale)

# summary statistics
print(telcom_scaled_df.describe())

# final preprocessed dataframe
telcom = pd.concat([telcom_scaled_df, telcom[['International plan', 'Voice mail plan','Churn']]], axis=1)


# In[139]:


# 4. Model Building and Performance Evaluation
#Model Selection:
#Logistic Regression


# In[16]:



# from sklearn.linear_model import LogisticRegression

# instantiate our classifier
clf = LogisticRegression()

#Creating training and test sets
# from sklearn.model_selection import train_test_split

# create feature variable (which holds all of the features of telco by dropping the target variable 'Churn' from telco)
X = telcom.drop(telcom[['Churn']], axis=1)

# create target variable
y = telcom['Churn']

# Create training and testing sets (here 80% of the data is used for training.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Fit to the training data
clf.fit(X_train, y_train)

# The predicted labels of classifier
y_pred = clf.predict(X_test)
#Check each sets length
print(X_train.shape)
print(X_test.shape)


# In[141]:


#Confusion matrix


# In[17]:


# Calculate the confusion matrix
matrix = confusion_matrix(y_test, y_pred)
# print(matrix)

# Plot the confusion matrix using seaborn
sns.heatmap(matrix, annot=True, fmt='d', cmap='magma')

# Add labels to the plot
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


# In[18]:


print(classification_report(y_test, y_pred))


# # Accuracy, Precision, Recall and F1 Score
# Accuracy is a measure of how well a classifier performs in terms of correctly predicting the class of an input sample.
# 
# Recall is a measure of the proportion of positive examples that were correctly classified by the model. It is calculated using the following formula:
#  Recall=TruePositives/(TruePositives+FalseNegatives)
# 
# Precision is a measure of the proportion of predicted positive examples that are actually positive. It is calculated using the following formula:
#  Precision=TruePositives/(TruePositives+FalsePositives)
#  
# The F1 score is a measure of the accuracy of a classifier, defined as the harmonic mean of precision and recall.
#  F1=2*
#  

# In[19]:


print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
print("F1 score: {:.2f}".format(f1_score(y_test, y_pred)))


# In[145]:


# ROC Curve


# In[20]:


# Generate the probabilities
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Use roc_curve() to calculate the false positive rate, true positive rate, and thresholds.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the ROC curve
plt.plot(fpr, tpr)

# Add labels and diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0, 1], [0, 1], "k--")
plt.show()


# In[147]:


# the area under the ROC curve


# In[21]:


roc_auc_score(y_test, y_pred_prob)


# In[149]:


#5. Making Predictions (whether a new customer will churn)


# In[22]:


def make_prediction(customer):
    pred = clf.predict(customer)
    if pred[0] == 1:
        print("[1] The customer will Churn.")
    else:
        print("[0] The customer will not Churn")


# In[151]:


# scaled input values


# In[23]:


new_customer1 = [[0.6262585675178604,
                  1.7188173197427594,
                 -1.0535424482925813,
                 -0.6197347815607696,
                 -1.1276788128173842,
                 0.5464802852218092,
                 -0.8676148392853111,
                 0.3011544282701762,
                 0.4523525497250106,
                 -0.6011950896927287,
                 -0.4279320210630441,
                 0.0,
                 0.0]]

new_customer2 = [[0.5257967737031338,
                  -0.5236032802413713,
                  0.9387740897371452,
                  1.5730210856813158,
                  0.8326323403400316,
                  -0.0559403500169171,
                  -0.3653036104833324,
                  -2.20323162813801,
                  0.27323229022856793,
                  -1.0075595662585095,
                  -1.1882184955849664,
                  1.0,
                  0.0]]

# make prediction on new customers
make_prediction(new_customer1)
make_prediction(new_customer2)

