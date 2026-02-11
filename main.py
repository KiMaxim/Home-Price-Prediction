import numpy 
import pandas
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pandas.read_csv('home_dataset.csv')

# Filter out houses priced at or below 500k
data = data[(data['HousePrice'] > 500000)].copy()

# Standardize HouseSize (z-score normalization)
mean_X = data['HouseSize'].mean()
st_dev_X = data['HouseSize'].std()
data['HouseSize_Scaled'] = (data['HouseSize'] - mean_X)/st_dev_X

# Standardize HousePrice (z-score normalization)
mean_Y = data['HousePrice'].mean()
st_dev_Y = data['HousePrice'].std()
data['HousePrice_Scaled'] = (data['HousePrice'] - mean_Y)/st_dev_Y

class Scratch_Version:
    """Linear regression implemented from scratch using gradient descent."""

    def __init__(self, data, L = 0.001, epochs = 1000):
        self.m = 0.0  # slope
        self.b = 0.0  # intercept
        self.L = L    # learning rate
        self.epochs = epochs
        self.data = data

    def error_function(self): 
        """Compute Mean Squared Error over the dataset."""
        total_error = 0
        for i in range(len(self.data)):
            x = self.data.iloc[i].HouseSize
            y = self.data.iloc[i].HousePrice
            total_error += (y - (self.m * x + self.b)) ** 2
        return total_error / float(len(self.data))

    def gradient_descent(self): 
        """Perform one step of gradient descent, updating m and b."""
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

    def scratch_r2(self):
        """Compute R² score manually."""
        x = self.data['HouseSize_Scaled'].to_numpy()
        y = self.data['HousePrice_Scaled'].to_numpy()
        predictions = self.m  * x + self.b
        ss_res = numpy.sum((y - predictions) ** 2)  # sum of squares of residuals
        ss_tot = numpy.sum((y - numpy.mean(y)) ** 2)  # total sum of squares
        return 1 - (ss_res / ss_tot)


# Log training results to file
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Read hyperparameters from user input
epochs = int(input())
L = float(input())
Scratch_Model = Scratch_Version(data, L, epochs)
losses = []

# Training loop
for _ in range(Scratch_Model.epochs):
    Scratch_Model.gradient_descent()
    losses.append(Scratch_Model.error_function())

logging.info(f"Epochs: {Scratch_Model.epochs}, With Learning Rate: {Scratch_Model.L}")
logging.info(f"Final MSE: {losses[-1]:.6f}")
logging.info(f"m: {Scratch_Model.m:.4f}, b: {Scratch_Model.b:.4f}")
logging.info(f"All loses: {[round(l, 6) for l in losses]}")
logging.info("-" * 24)

# Prepare data for sklearn model (unscaled)
HouseSizes = data['HouseSize'].to_numpy().reshape(-1, 1)
HousePrices = data['HousePrice'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(HouseSizes, HousePrices, test_size=0.2, random_state=42)

# Train sklearn LinearRegression model
model = LinearRegression()
model.fit(x_train, y_train)

# Scatterplot of raw dataset
plt.scatter(HouseSizes, HousePrices, marker='o', color='blue') 
plt.title('House Prices vs House Size') 
plt.xlabel('House Sizes (ft.sq)') 
plt.ylabel('House Price ($)') 
plt.show()


plt.figure(figsize=(20, 5))

# Prepare sorted x-values for plotting the regression line
X_sorted = numpy.sort(data['HouseSize_Scaled'].to_numpy())
Y_line = Scratch_Model.m * X_sorted + Scratch_Model.b

# Scratch model on scaled data
plt.subplot(1, 2, 1)
plt.scatter(data['HouseSize_Scaled'], data['HousePrice_Scaled'], color="blue")
plt.plot(X_sorted, Y_line, color="green", label="Scratch Model (Original Scaled Data)")
plt.legend()
plt.title("Scratch Model (Scaled Data)")
plt.xlabel('House Size (Scaled)')
plt.ylabel('House Price (Scaled)')

# Sklearn model on original data
plt.subplot(1, 2, 2)
predictions = model.predict(x_test)
plt.scatter(x_test, y_test, marker='^', color='blue', label='Actual Price')
sorted_indices = x_test.flatten().argsort()
plt.plot(x_test[sorted_indices], predictions[sorted_indices], color='red', linewidth=2, label='Predicted Price')
plt.title('House Prices vs House Size')
plt.xlabel('House Sizes (ft.sq)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()

print(f"R² Score for Scratch Model: {Scratch_Model.scratch_r2():.4f}")
print(f"R² Score for Sklearn Model: {r2_score(y_test, predictions):.4f}")

print(f"MSE for Scratch Model: {losses[-1]:.6f}") # Note: MSE for scratch model is the last value in losses list, which is computed on RAW data
print(f"MSE for Sklearn Model: {mean_squared_error(y_test, predictions):.6f}")

