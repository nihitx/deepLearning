import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

dataframe = pd.read_csv('bmi_life_expectancy.csv')
country = dataframe[['Country']]
life = dataframe[['Life expectancy']]
BMI = dataframe[['BMI']]


#train the model
regr = linear_model.LinearRegression()
regr.fit(BMI, life)
prediction = regr.predict(BMI)


#visualize the result
plt.scatter(BMI, life)
plt.plot(BMI, prediction)
plt.show()