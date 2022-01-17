import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([1000,950,800,650,720,850, 1400, 1450, 1200, 1250, 900, 930, 820, 780, 980, 1050, 1280, 1320, 1430, 1100])
y=np.array([36,34, 30,28,30,31, 60,70,54, 65,37,40, 37,32, 35,43, 62, 67, 80, 5])
linearRegression=LinearRegression()
x=x.reshape(-1,1)
linearRegression.fit(x,y)
print(linearRegression.coef_)
print(linearRegression.intercept_)

y1=(linearRegression.coef_*985)+linearRegression.intercept_
print(y1)

y2=(linearRegression.coef_*1225)+linearRegression.intercept_
print(y2)
