import numpy 
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('home_dataset.csv')

HouseSizes = data['HouseSize'].to_numpy().reshape(-1, 1)
HousePrices = data['HousePrice'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(HouseSizes, HousePrices, test_size=0.2, random_state=42)

# Create an object of the LinearRegression model and train (fit) it
# using the training feature values (x_train) and target values (y_train)
model = LinearRegression()
model.fit(x_train, y_train)

#Scatterplot of the provided dataset 
plt.scatter(HouseSizes, HousePrices, marker='o', color='blue') 
plt.title('House Prices vs House Size') 
plt.xlabel('House Sizes (ft.sq)') 
plt.ylabel('House Price ($)') 
plt.show()


#Linear Regression Graph:
predictions = model.predict(x_test)
plt.scatter(x_test, y_test, marker='^', color='blue', label='Actual Price')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted Price')
plt.title('House Prices vs House Size')
plt.xlabel('House Sizes (ft.sq)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()

