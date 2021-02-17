#        THE SPARK FOUNDATION INTERNSHIP

#        SHREYANSH JAIN
#        TASK 1 : Predict the percentage of an student based on the no. of hours of studies

#        PRIDECTION USING SUPERVISED MACHINE LEARNING




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#     reading data set

data = pd.read_csv("http://bit.ly/w-data") 
data.head()
data.isnull().sum()
data.shape

data.select_dtypes(include = ["category"])


###  get correlations of each features in dataset
getcorr = data.corr()
top_corr_features = getcorr.index
plt.figure(figsize = (20,20))
### heat map

A = sns.heatmap(data[top_corr_features].corr(),annot = True, cmap = "RdYlGn")


data.plot(x = 'Scores' , y = 'Hours', style = 'o')
plt.title("hours vs percentage")
plt.ylabel("study Hours" ,color = 'red')
plt.xlabel("percentage scores",color = 'red')
plt.show()

independent = data.iloc[:,:-1].values
dependent = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
independent_train , independent_test , dependent_train , dependent_test = train_test_split(independent, dependent)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(independent_train , dependent_train)

LinearRegression(copy_X = True, fit_intercept=True, n_jobs= None , normalize = False)

line =   regressor.coef_*independent+regressor.intercept_
plt.scatter(independent,dependent)
plt.plot(independent,line)
plt.show()


print(independent_test)

dependent_pred = regressor.predict(independent_test)

model = pd.DataFrame({'Actual':dependent_test,'Predicted':dependent_pred})

hours = [[9.25]]
own_pred= regressor.predict(hours)
print("Number of hours = {}".format(hours))
if own_pred[0]>100 :
    print("prediction score=100")
else:
    print("predicition score = {}".format(own_pred[0]))

from sklearn import metrics 
print('Mean Absolute Error:',
      metrics.mean_absolute_error(dependent_test , dependent_pred))

print('variance score :%2f'%regressor.score(independent_test,dependent_test))


          ### TASK 1 COMPLETED!!


















