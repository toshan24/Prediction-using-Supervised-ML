#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation : Data Science and Business Analytics Intern
# **Author: Pankaj Sharma**
# 

# #### Task 1 : Prediction using Supervised ML
# To predict the percentage of a student based on the number of study hours.
# The given data set consists of Hours and Scores. Simple Linear Regression is going to be utilised as this is only for 2 variables.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


url = 'http://bit.ly/w-data'
data = pd.read_csv(url)
print('This data has been imported!')


# In[3]:


data.head()


# ### To understand relationship of this data, we are going to plot the data points and obtain a graphical representation.

# In[9]:


data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores in Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# #### Observed: A Positive linear relation between the number of hours studied and percentage score.

# ##### Preparation Of the Data

# Dividing the data into "attributes" and "labels" as inputs and outputs respectively.

# In[12]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# ### Training and Testing Sets

# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[18]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print('Training has been completed!')


# ### Plotting the regression line

# In[19]:


line = regressor.coef_* x + regressor.intercept_
plt.scatter(x,y)
plt.plot(x, line, color = "red")
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scored')
plt.show()


# **Given above is regression line**

# ### Making predictions:

# In[20]:


print(x_test) # testing data 
#in Hours
y_pred = regressor.predict(x_test)


# ##### Actual vs Predicted
# 

# In[22]:


# Actual vs Predicted
df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
df


# ## Evaluating the model using Mean Absolute Error

# In[25]:


from sklearn import metrics
print('Mean Aboslute Error: ', metrics.mean_absolute_error(y_test, y_pred))


# # Query in Task 1
# #### Task : What will be the predicted score if a student studies for 9.25 hours a day?

# In[27]:


hours = [[9.25]]
pred_value = regressor.predict(hours)
print('Number of hours : {}'.format(hours))
print('Predicted Score : {}'.format(pred_value[0]))


# # Conclusion 
# ### The student who studies for 9.25 hours a day might get a predicted score of "93.69"
