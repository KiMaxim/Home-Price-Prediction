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

class Scratch_Version:
    def __init__(self, data, L = 0.001):
        self.m = 0.0
        self.b = 0.0
        self.L = L
        self.epochs = epochs
        self.data = data

    def error_function(self): 
        total_error = 0
        for i in range(len(self.data)):
            x = self.data.iloc[i].HouseSize
            y = self.data.iloc[i].HousePrice
            total_error += (y - (self.m * x + self.b)) ** 2
        return total_error / float(len(self.data))

    def gradient_descent(self): 
        m_grad = 0
        b_grad = 0
        n = len(self.data)
        for i in range(n):
            x = self.data.iloc[i].HouseSize_Scaled
            y = self.data.iloc[i].HousePrice_Scaled

            m_grad += -(2/n) * x * (y  - (self.m * x +  self.b))
            b_grad += -(2/n) * (y  - (self.m * x + self.b))

        self.m -= self.L * m_grad
        self.b -= self.L * b_grad
    
epochs = int(input())
L = float(input())
Scratch_Model = Scratch_Version(data, L)
losses = []

for _ in range(epochs):
    Scratch_Model.gradient_descent()
    losses.append(Scratch_Model.error_function())

print(losses)

X_plot = numpy.linspace(data['HouseSize_Scaled'].min(),
                        data['HouseSize_Scaled'].max(),
                        100)

#Using a data directly
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['HouseSize_Scaled'], data['HousePrice_Scaled'], color="blue")
plt.plot(data['HouseSize_Scaled'], Scratch_Model.m * data['HouseSize_Scaled'] + Scratch_Model.b, color="red")
plt.title("Using HouseSize_Scaled directly")

#Using linspace 
plt.subplot(1, 2, 2)
X_line = numpy.linspace(data['HouseSize_Scaled'].min(), 
                        data['HouseSize_Scaled'].max(), 
                        100)

Y_line = Scratch_Model.m * X_line + Scratch_Model.b
plt.scatter(data['HouseSize_Scaled'], data['HousePrice_Scaled'], color="blue")
plt.plot(X_line, Y_line, color="red")
plt.title("Using linspace")

plt.show()