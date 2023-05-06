#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This notebook contains the code needed for a bank to categorize the credit scores of its customers.

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

#Put an 'r' in front of the string in order for the path to be read as a raw string

data=pd.read_csv(r"C:\Users\Zach\Desktop\Credit Score Data\train.csv")
print(data.head())


# In[3]:


#learn about the data

print(data.info())


# In[4]:


#Scour the data for nulls

print(data.isnull().sum())


# In[5]:


data["Credit_Score"].value_counts()


# In[6]:


#Explore the relationship between occupation and credit score

fig = px.box(data, 
             x="Occupation",  
             color="Credit_Score", 
             title="Credit Scores Based on Occupation", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.show()


# In[7]:


#Now explore credit scores based on annual income

fig = px.box(data, 
             x="Credit_Score", 
             y="Annual_Income", 
             color="Credit_Score",
             title="Credit Scores Based on Annual Income", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[8]:


#explore the relationship between salaries and credit scores

fig = px.box( data, x="Credit_Score", y="Monthly_Inhand_Salary", color="Credit_Score", 
             title="Credit Scores based on monthly inhand salary", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[9]:


#Determining the relationship between bank accounts and credit scores

fig = px.box( data, x="Credit_Score", y="Num_Bank_Accounts", color="Credit_Score", 
             title="Credit Scores based on number of bank accounts", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[10]:


#Determining the relationship between the number of credit cards and credit scores

fig = px.box( data, x="Credit_Score", y="Num_Credit_Card", color="Credit_Score", 
             title="Credit Scores based on number of credit cards", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[11]:


#Determining the relationship between interest rates and credit scores

fig = px.box( data, x="Credit_Score", y="Interest_Rate", color="Credit_Score", 
             title="Credit Scores based on the average interest rate", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[12]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Num_of_Loan", color="Credit_Score", 
             title="Credit Scores based on number of number of loans", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[13]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Delay_from_due_date", color="Credit_Score", 
             title="Credit Scores based on average number of days delayed for credit card payments", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[15]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Num_of_Delayed_Payment", color="Credit_Score", 
             title="Credit Scores based on number of delayed payments", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[16]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Outstanding_Debt", color="Credit_Score", 
             title="Credit Scores based on Outstanding Debt", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[17]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Credit_Utilization_Ratio", color="Credit_Score", 
             title="Credit Scores based on Credit Utilization Ratio", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[18]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Credit_History_Age", color="Credit_Score", 
             title="Credit Scores based on Credit History Age", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[20]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Total_EMI_per_month", color="Credit_Score", 
             title="Credit Scores based on total number of EMIs per month", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[21]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Amount_invested_monthly", color="Credit_Score", 
             title="Credit Scores based on Amount Invested Monthly", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[22]:


#Determining the relationship between number of loans and credit scores

fig = px.box( data, x="Credit_Score", y="Monthly_Balance", color="Credit_Score", 
             title="Credit Scores based on Monthly Balance Left", 
             color_discrete_map={'Poor':'red', 'Standard':'yellow', 'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[25]:


#Here, we transform the data into a nuemricla set so it can be used to train machine learning models

data["Credit_Mix"] = data["Credit_Mix"].map({"Standard":1, "Good":2, "Bad":3})


# In[26]:


from sklearn.model_selection import train_test_split

x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
                  "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
                  "Num_of_Delayed_Payment", "Credit_Mix", "Outstanding_Debt",
                  "Credit_History_Age","Monthly_Balance"]])

y = np.array(data[["Credit_Score"]])


# In[28]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)


# In[29]:


#For getting user inputs

print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 2) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))


# In[ ]:




