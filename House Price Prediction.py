#!/usr/bin/env python
# coding: utf-8

# ## Import required packages

# In[20]:


#Regular EDA (exploratory data analysis) and plotting libraries
import math
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


#Package for splitting the dataset to training set and test set
from sklearn.model_selection import train_test_split, cross_val_predict

#Package for Linear Regression model
from sklearn.linear_model import LinearRegression

#Packages to perform Exhaustive Search
from dmba import regressionSummary, exhaustive_search
from dmba import adjusted_r2_score, AIC_score, BIC_score

#Package for KNN Regressor model
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

#Package for data standardization
from sklearn.preprocessing import StandardScaler

#Package for model evaluation
from dmba import regressionSummary, classificationSummary
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score


# ## Importing housing dataset

# In[21]:


#Importing housing dataset
housing_df = pd.read_csv('HousingDataSet.csv')


# ## Data preprocessing

# In[22]:


#Viewing the first 10 rows
housing_df.head(10)


# In[23]:


#Viewing dataframe structure
housing_df.shape


# In[24]:


#Counting the number of values in each column
housing_df.count()


# In[25]:


#Counting the number of unique value in each column
housing_df.nunique()


# In[26]:


#Rechecking if there are any null values in our dataset
housing_df.info()


# In[27]:


#Plotting null values in our dataset by using heatmap
sns.heatmap(housing_df.isnull())
plt.title("Empty Data")


# In[28]:


#Because yr_built is a ordinal variable, I will transform it to a numeric variable.
#Creating AGE variable (age of the property)
housing_df['age'] = 2022 - housing_df['yr_built']


# In[29]:


#Changing yr_renovated variable to dummy variable (whether the apartment was renovated or not)
housing_df.loc[housing_df['yr_renovated'] != 0, 'yr_renovated'] = 1


# In[39]:


#Renaming column yr_renovated to renovated
housing_df.rename(columns={'yr_renovated': 'renovated'}, inplace=True)


# In[40]:


#Droping unnecessary columns id, date, yr_built, zipcode in housing dataset
housing_df.drop(['id', 'date', 'yr_built', 'zipcode'], axis=1, inplace=True)


# In[41]:


#Viewing dataframe structure
housing_df.shape


# ## Data partitioning

# In[42]:


#Creating X and y data matrices (X = predictor variables, y = outcome variable)
X=housing_df.drop(labels=['price'], axis=1)
y=housing_df['price']


# In[43]:


#Splitting the dataset into training set size = 0.6 and validation set size = 0.4
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)


# In[44]:


train_X.shape, train_y.shape, valid_X.shape, valid_y.shape


# To avoid overfitting, we will be running all the functions and training only on the training set (not train again on the test dataset), and then what features we take away from the train dataset, we are going do the same on the validation set.

# ## Feature selection using correlation heatmap

# In[45]:


#Constructing a heatmap of correlation on the training set (for only independent features)
corr = train_X.corr()
sns.heatmap(corr)
fig, ax = plt.subplots()
fig.set_size_inches(11, 7)
sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdBu", center=0, ax=ax)


# In[46]:


#Build corr function and use it to eliminate predictors on train x and valid x (sqft_above)
#With the following function we can select highly correlated features
#The function will remove the first feature that is correlated with other features

def correlation(dataset, threshold): # correlation-function name, dataset=train_X, threshold=0.8)
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)): # i goes through all the values in the correlation matrix columns
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we use each absolute coeff value to compare with threshold value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname) # add the column name to the col_corr set created
    return col_corr


# In[47]:


#Build corr function and use it to eliminate predictors on train x and valid x (sqft_above)
#Count the number of highly correlated features (cutoff 0.8)
corr_features = correlation(train_X, 0.8)
len(set(corr_features))


# In[48]:


#Build corr function and use it to eliminate predictors on train x and valid x (sqft_above)
#Print the name of highly correlated features
corr_features


# In[49]:


#Build corr function and use it to eliminate predictors on train x and valid x (sqft_above)
#Remove highly correlated feature (sqft_above) out of training dataset
train_X.drop(corr_features,axis=1, inplace= True)


# In[50]:


#Build corr function and use it to eliminate predictors on train x and valid x (sqft_above)
#Remove highly correlated feature (sqft_above) out of test dataset
valid_X.drop(corr_features,axis=1, inplace= True)


# In[53]:


train_X.shape, valid_X.shape


# ## Demo Linear Regression model

# In[54]:


#Build demo regression model
model1 = LinearRegression().fit(train_X, train_y)


# In[55]:


print('Intercept:', model1.intercept_)
print(pd.DataFrame({'Predictor': train_X.columns, 'Coefficients': model1.coef_}))


# In[56]:


# print performance measures (training data)
regressionSummary(train_y, model1.predict(train_X))


# In[57]:


# Use predict() to make predictions on test set
y_pred1 = model1.predict(valid_X)


# In[58]:


result = pd.DataFrame({'Predicted Values': y_pred1, 'Actual Values': valid_y, 'Residuals': valid_y - y_pred1})
print(result.head(2))


# In[59]:


#print performace measures (test data)
regressionSummary(valid_y, model1.predict(valid_X))


# In[60]:


#Predictive accuracy measurement for demo model
from sklearn.metrics import r2_score
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred1))
#Coefficient of determination measures how well the regession model predicts the outcome variable.


# ## Conducting exhaustive search

# In[62]:


#Exhaustive Search Code on training dataset
def train_model(variables):
    model1 = LinearRegression()
    model1.fit(train_X[list(variables)], train_y)
    return model1
#use funtion to pick list of variables all over again

def score_model(model1, variables):
    pred_y = model1.predict(train_X[list(variables)])
    # we negate as score is optimized to be as low as possible
    return -adjusted_r2_score(train_y, pred_y, model1)

allVariables = train_X.columns
results = exhaustive_search(allVariables, train_model, score_model)

data = []
for result in results:
    model1 = result['model']
    variables = list(result['variables'])
    AIC = AIC_score(train_y, model1.predict(train_X[variables]), model1)

    d = {'n': result['n'], 'r2adj': -result['score'], 'AIC':AIC}
    d.update({var: var in result['variables'] for var in allVariables})
    data.append(d)
pd.DataFrame(data, columns=('n', 'r2adj', 'AIC') + tuple(sorted(allVariables)))

#Note: 𝑅𝑎𝑑𝑗^2 0.685 indicates that a model with 6 predictors is good (not so different with R^2 0.695 of 16 predictors).
#This model can help us save storage space, amd time on processing/analysis


# In[88]:


#Remove unnecessary features out of traing dataset, only keep 'true' predictors at line n = 6
train_X = train_X[['age', 'grade', 'lat', 'sqft_living', 'view', 'waterfront']]


# In[89]:


#Remove unnecessary features out of test dataset
valid_X = valid_X[['age', 'grade', 'lat', 'sqft_living', 'view', 'waterfront']]


# ## Multiple linear regression model

# In[90]:


#Build new model with 6 predictors based on Exhaustive Search result
model2 = LinearRegression().fit(train_X, train_y)


# In[91]:


print('Intercept:', model2.intercept_)
print(pd.DataFrame({'Predictor': train_X.columns, 'Coefficients': model2.coef_}))


# In[92]:


# Print performance measures (training data)
regressionSummary(train_y, model2.predict(train_X))


# In[93]:


# Use predict() to make predictions on a new set
y_pred2 = model2.predict(valid_X)


# In[94]:


#Use regrssion model to predict the prices of the two houses
result = pd.DataFrame({'Predicted Values': y_pred2, 'Actual Values': valid_y, 'Residuals': valid_y - y_pred2})
print(result.head(2))


# In[95]:


#print performace measures (test data)
regressionSummary(valid_y, model2.predict(valid_X))

#Training data Errors < Test data errors, model is not overfitting
#The difference in errors between Training set and Test set are not significant => good


# In[96]:


#Predictive accuracy measurement
from sklearn.metrics import r2_score
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred2))
#Coefficient of determination measures how well the regession model predicts the outcome variable.
#the larger the R^2, the better the regression model can explain the outcome variable and fit the actual values


# In[97]:


#Performance measurement
#charts of error distribution for Training set and Test set
#training set
train_e = train_y - model2.predict(train_X)
fig, ax = plt.subplots()
ax = train_e.hist()
ax.set_xlabel('Training')


# In[98]:


# validation set
valid_e = valid_y - model2.predict(valid_X)
fig, ax = plt.subplots()
ax = valid_e.hist()
ax.set_xlabel('Validation')


# Similar error distribution (highly distributed around 0) of training and valid sets 

# In[99]:


#When evaluate performance of model 1 (16 predictors) and 2 (6 predictors)
#Root Mean Squared Error (RMSE) : model 1 213188.0633 ~ model 2 217316.3443
#Coefficient of determination (R^2): model1 0.69 ~ model2 0.68
#But we do a good job on eliminating redundant variables (10) which avoids a waste of space in the source, 
#makes us easier to focus on those that important, and optimize the speed of model deployment


# ## KNN REGRESSOR MODEL BUILDING

# To build the KNN regressor model we will use the same predictors 'age', 'grade', 'lat', 'sqft_living', 'view', 'waterfront' on the same training and test datasets. Later we need to compare the performance of these 2 models

# In[100]:


#Construct model with k = 5
model3 = KNeighborsRegressor(n_neighbors=5).fit(train_X, train_y)


# In[101]:


# Print performance measures (training data) (model 3)
regressionSummary(train_y, model3.predict(train_X))


# In[102]:


#Model evaluation (model 3)
# Use predict() to make predictions on a new set
y_pred3 = model3.predict(valid_X)


# In[103]:


#print performance measures (test data) (model 3)
regressionSummary(valid_y, model3.predict(valid_X))


# In[104]:


#model 3
#Training data Errors < Test data errors, model is not overfitting
#The difference in errors between Training set and Test set are not significant => good


# In[105]:


#Predictive accuracy measurement
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred3))


# In[106]:


#Construct model with k = 10
model4 = KNeighborsRegressor(n_neighbors=10).fit(train_X, train_y)


# In[107]:


# Print performance measures (training data) (model 4)
regressionSummary(train_y, model4.predict(train_X))


# In[108]:


#Model evaluation (model 4)
# Use predict() to make predictions on a new set
y_pred4 = model4.predict(valid_X)


# In[109]:


#print performance measures (test data) (model 4)
regressionSummary(valid_y, model4.predict(valid_X))


# In[110]:


#model 4
#Training data Errors < Test data errors, model is not overfitting
#The difference in errors between Training set and Test set are not significant => good


# In[112]:


#Predictive accuracy measurement
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred4))


# In[113]:


#Compare model 3 and 4
#Coefficient of determination: model4 0.54 > model3 0.51
#Root Mean Squared Error (RMSE): model4 262065.4031 < model3 268951.2651
#=> choose model 4 with k=10 which has better prediction accuracy 


# In[115]:


#Use regrssion model to predict the prices of the two houses
result = pd.DataFrame({'Predicted Values': y_pred4, 'Actual Values': valid_y, 'Residuals': valid_y - y_pred4})
print(result.head(2))


# ## Compare the prediction performance of multiple linear regression model and KNN regressor model 

# * Coefficient of determination: model4 0.54 < model2 0.68
# * Root Mean Squared Error (RMSE): model4 262065.4031 > model2 217316.3443
# * Therefore, we choose model 2 which has better prediction accuracy 

# ## Enhancing model performance by feature scaling

# In[116]:


#Checking data range of 6 predictors
train_X.describe()


# Features have different scales.

# In[117]:


#Plotting sqft_living distribution
dis_sqft_living = sns.displot(train_X, x="sqft_living")


# In[118]:


#Plotting price distribution
dis_price = sns.displot(train_y)


# It is better to perform data standardization because the dataset contains so many outliers.

# In[119]:


#Conducting data standardization
scaler = StandardScaler()
stan_train_X = pd.DataFrame(scaler.fit_transform(train_X),index=train_X.index,columns=train_X.columns)
stan_valid_X = pd.DataFrame(scaler.fit_transform(valid_X),index=valid_X.index,columns=valid_X.columns)


# In[120]:


#Checking the data range of standardized data
stan_train_X.describe()


# In[121]:


#Building multiple linear regression model using standardized data
model5 = LinearRegression().fit(stan_train_X, train_y)


# In[122]:


#Printing model performance measurement (on training data) model5
regressionSummary(train_y, model5.predict(stan_train_X))


# In[123]:


#Using predict() function to make predictions on test set model5
y_pred5 = model5.predict(stan_valid_X)


# In[124]:


#Printing model performance measurement (on test data) model5
regressionSummary(valid_y, model5.predict(stan_valid_X))

#Training data Errors < Test data errors, model is not overfitting
#The difference in errors between Training set and Test set are not significant => good


# In[125]:


#Printing predictive accuracy measurement model5
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred5))


# In[139]:


#Constructing model with k = 5 using standardized data
model6 = KNeighborsRegressor(n_neighbors=5).fit(stan_train_X, train_y)


# In[140]:


#Printing model performance measurement (on training data) model6
regressionSummary(train_y, model6.predict(stan_train_X))


# In[141]:


#Using predict() function to make predictions on test set model6
y_pred6 = model6.predict(stan_valid_X)


# In[142]:


#Printing model performance measurement (on test data) model6
regressionSummary(valid_y, model6.predict(stan_valid_X))


# In[143]:


#Printing predictive accuracy measurement model6
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred6))


# In[144]:


#Constructing model with k = 10 using standardized data
model7 = KNeighborsRegressor(n_neighbors=10).fit(stan_train_X, train_y)


# In[145]:


#Printing model performance measurement (on training data) model6
regressionSummary(train_y, model7.predict(stan_train_X))


# In[146]:


#Using predict() function to make predictions on test set model6
y_pred7 = model7.predict(stan_valid_X)


# In[147]:


#Printing model performance measurement (on test data) model6
regressionSummary(valid_y, model7.predict(stan_valid_X))


# In[148]:


#Printing predictive accuracy measurement model6
print('Coefficient of determination (R^2): %.2f'
      % r2_score(valid_y, y_pred7))


# ## Conclusion

# * Our optimal model is model6 which was built by using KNN regressor algorithm on standardized data with k = 5
# * Model accuracy Coefficient of determination (R^2): 0.79

# In[ ]:




