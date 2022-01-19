
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("assign1of2.csv")

median_bedrooms=math.floor(df.bedrooms.median())

df.bedrooms=df.bedrooms.fillna(median_bedrooms)
linearRegression=LinearRegression()
linearRegression.fit(df[['area','bedrooms','age','car parking']],df.price)

print("Coefficient",linearRegression.coef_)
print("Intercept",linearRegression.intercept_)
print("Predicted value of 10000sqft, 2 bedroom, 4 years old and 1 car parking",linearRegression.predict([[10000,2,4,1]]))
print("Predicted value of 800 sqft, 2 bedrooms, 5 years old and 1 car parking",linearRegression.predict([[800,2,5,1]]))
