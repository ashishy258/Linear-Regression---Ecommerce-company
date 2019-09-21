#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Get the Data
df=pd.read_csv('Ecommerce Customers')
df.head()
df.info()
df.describe()

#Exploratory Data Analysis
sns.jointplot(df['Time on Website'],df['Yearly Amount Spent'])
sns.pairplot(df)

df.corr()
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)

#model

df.columns
x=df[[ 'Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

#

predictions=lm.predict(x_test)


plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

coeff = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff