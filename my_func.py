import numpy 
import pandas
import matplotlib.pyplot as plt
from math import sqrt

data = pandas.read_csv('home_dataset.csv')

data = data[(data['HousePrice'] > 500000)].copy()

#data standardization for HouseSize
mean_X = data['HouseSize'].mean()
st_dev_X = data['HouseSize'].std()
data['HouseSize_Scaled'] = (data['HouseSize'] - mean_X)/st_dev_X

#data standardization for HousePrice
mean_Y = data['HousePrice'].mean()
st_dev_Y = data['HousePrice'].std()
data['HousePrice_Scaled'] = (data['HousePrice'] - mean_Y)/st_dev_Y


def standardize(mean, st_dev, data):
    for i in range(len(data)):
        data.iloc[i].HouseSize = (data.iloc[i].HouseSize - mean) / st_dev
    return data


def error_function(m, b, points): 
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].HouseSize
        y = points.iloc[i].HousePrice
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L): 
    m_grad = 0
    b_grad = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].HouseSize_Scaled
        y = points.iloc[i].HousePrice_Scaled

        m_grad += -(2/n) * x * (y  - (m_now * x + b_now))
        b_grad += -(2/n) * (y  - (m_now * x + b_now))

    m = m_now - m_grad * L
    b = b_now - b_grad * L
    return m, b

m = 0
b = 0
L = 0.01
epochs = 300

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

X_plot = numpy.linspace(data['HouseSize_Scaled'].min(),
                        data['HouseSize_Scaled'].max(),
                        100)

# Вариант 1: Используя существующие данные (может быть ломаная)
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['HouseSize_Scaled'], data['HousePrice_Scaled'], color="blue")
plt.plot(data['HouseSize_Scaled'], m * data['HouseSize_Scaled'] + b, color="red")
plt.title("Using HouseSize_Scaled directly")

# Вариант 2: Используя linspace (всегда прямая)
plt.subplot(1, 2, 2)
X_line = numpy.linspace(data['HouseSize_Scaled'].min(), 
                        data['HouseSize_Scaled'].max(), 
                        100)
Y_line = m * X_line + b
plt.scatter(data['HouseSize_Scaled'], data['HousePrice_Scaled'], color="blue")
plt.plot(X_line, Y_line, color="red")
plt.title("Using linspace")

plt.show()