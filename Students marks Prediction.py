#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from sklearn.metrics import r2_score


# In[49]:


data = pd.read_csv('D:/Student_Marks.csv')


# In[50]:


data.info()


# In[51]:


data.describe()


# In[52]:


data.isnull()


# In[53]:


data.shape


# In[54]:


x = data[['number_courses', 'time_study']]
y = data['Marks']


# In[55]:


import matplotlib.pyplot as plt
plt.scatter(x=data["time_study"],y=data["Marks"])
plt.title("Student's Marks and study time ")
plt.xlabel("Study time")
plt.ylabel("Marks")


# In[56]:


x = data[['number_courses', 'time_study']]
y = data['Marks']


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[63]:


# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
linear_predictions = linear_model.predict(x_test)
linear_mse = mean_squared_error(y_test, linear_predictions)
L_r2_TStudy = r2_score(y_test, linear_predictions)
L_mae = mean_absolute_error(y_test, linear_predictions)


# In[71]:


plt.scatter(y_test, linear_predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Optimal Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()
plt.show()


# In[64]:


# Polynomial Regression model
poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x_train)
poly_model = LinearRegression()
poly_model.fit(x_poly, y_train)
x_test_poly = poly_features.transform(x_test)
poly_predictions = poly_model.predict(x_test_poly)
poly_mse = mean_squared_error(y_test, poly_predictions)
P_r2_TStudy = r2_score(y_test, poly_predictions)
R_mae = mean_absolute_error(y_test, poly_predictions)


# In[72]:


plt.scatter(y_test, poly_predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Optimal Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Polynomial Regression: Actual vs. Predicted')
plt.show()


# In[65]:


print("Linear Regression MSE:", linear_mse)
print("Polynomial Regression MSE:", poly_mse)
print('The Linear Regression r squares is :', L_r2_TStudy*100)
print('The Polynomial Regression r squares is :', P_r2_TStudy*100)
print('The Linear Regression MAE is :',L_mae )
print('The Polynomial Regression MAE is :',R_mae )

