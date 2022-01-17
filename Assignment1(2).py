# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:24:46 2022

@author: Vinyas S
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("assign1of2.csv")
print(df)

median_bedrooms=math.floor(df.bedrooms.median())

df.bedrooms=df.bedrooms.fillna(median_bedrooms)

print(df)
linearRegression=LinearRegression()
linearRegression.fit(df[['area','bedrooms','age','car parking']],df.price)

print(linearRegression.coef_)
print(linearRegression.intercept_)
print(linearRegression.predict([[10000,2,4,1]]))
print(linearRegression.predict([[800,2,5,1]]))
