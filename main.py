import numpy
import pandas
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pandas.read_csv('home_dataset.csv')

# Keep only houses priced above $500k
data = data[data['HousePrice'] > 500_000].copy()

mean_X, std_X = data['HouseSize'].mean(), data['HouseSize'].std()
mean_Y, std_Y = data['HousePrice'].mean(), data['HousePrice'].std()

data['HouseSize_Scaled']  = (data['HouseSize']  - mean_X) / std_X
data['HousePrice_Scaled'] = (data['HousePrice'] - mean_Y) / std_Y


x_train, x_test, y_train, y_test = train_test_split(
    data['HouseSize_Scaled'].values,
    data['HousePrice_Scaled'].values,
    test_size=0.2,
    random_state=42,
)

# Full scaled arrays used for the scatter plot
HouseSizes  = data['HouseSize_Scaled'].to_numpy().reshape(-1, 1)
HousePrices = data['HousePrice_Scaled'].to_numpy()


class Scratch_Version:
    """Linear regression implemented from scratch using gradient descent."""

    x_train: numpy.ndarray
    y_train: numpy.ndarray
    x_test: numpy.ndarray
    y_test: numpy.ndarray
    L: float
    epochs: int

    def __init__(self, *args, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.m = 0.0
        self.b = 0.0 

    def gradient_descent(self, x, y):
        """Perform one gradient-descent step, updating slope (m) and intercept (b)."""
        n = len(x)
        y_pred  = self.m * x + self.b
        m_grad  = (-2 / n) * numpy.sum(x * (y - y_pred))
        b_grad  = (-2 / n) * numpy.sum(y - y_pred)
        # BUG FIX: was referencing global `L` instead of `self.L`
        self.m -= self.L * m_grad
        self.b -= self.L * b_grad

    def predict(self, x):
        return self.m * x + self.b

    def mse(self, x, y):
        """Mean squared error."""
        return numpy.mean((y - self.predict(x)) ** 2)

    def r2(self, x, y):
        """R² score."""
        y_pred = self.predict(x)
        rss = numpy.sum((y - y_pred) ** 2)
        tss = numpy.sum((y - numpy.mean(y)) ** 2)
        return 1 - rss / tss


epochs         = int(input("Epochs: "))
learning_rate  = float(input("Learning rate: "))

scratch_model = Scratch_Version(
    x_train=x_train, y_train=y_train,
    x_test=x_test,   y_test=y_test,
    L=learning_rate,
    epochs=epochs,
)


train_losses, test_losses = [], []

for _ in range(scratch_model.epochs):
    scratch_model.gradient_descent(scratch_model.x_train, scratch_model.y_train)
    train_losses.append(scratch_model.mse(scratch_model.x_train, scratch_model.y_train))
    test_losses.append(scratch_model.mse(scratch_model.x_test,  scratch_model.y_test))


sklearn_model = LinearRegression()
sklearn_model.fit(x_train.reshape(-1, 1), y_train)
sklearn_predictions = sklearn_model.predict(x_test.reshape(-1, 1))



# 1. Raw scatter plot
plt.figure()
plt.scatter(HouseSizes, HousePrices, marker='o', color='blue')
plt.title('House Prices vs House Size (Scaled)')
plt.xlabel('House Size (Scaled)')
plt.ylabel('House Price (Scaled)')
plt.show()

# BUG FIX: all subplots are now built before calling plt.show(),
# so the loss curve subplot is no longer orphaned in a separate figure.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2. Scratch model predictions
ax = axes[0]
sorted_idx = scratch_model.x_test.argsort()
ax.scatter(scratch_model.x_test, scratch_model.y_test,
           marker='^', color='blue', label='Actual Test Price')
ax.plot(scratch_model.x_test[sorted_idx],
        scratch_model.predict(scratch_model.x_test)[sorted_idx],
        color='red', linewidth=2, label='Scratch Predicted Price')
ax.set_title('Scratch Model — Test Predictions')
ax.set_xlabel('House Size (Scaled)')
ax.set_ylabel('House Price (Scaled)')
ax.legend()

# 3. Sklearn model predictions
ax = axes[1]
sorted_idx = x_test.argsort()
ax.scatter(x_test, y_test, marker='^', color='blue', label='Actual Price')
ax.plot(x_test[sorted_idx], sklearn_predictions[sorted_idx],
        color='red', linewidth=2, label='Sklearn Predicted Price')
ax.set_title('Sklearn Model — Test Predictions')
ax.set_xlabel('House Size (Scaled)')
ax.set_ylabel('House Price (Scaled)')
ax.legend()

# 4. Loss curves
ax = axes[2]
ax.plot(train_losses, label='Train Loss', color='blue')
ax.plot(test_losses,  label='Test Loss',  color='red')
ax.set_title('Loss Curves')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


# Metrics

print(f"R² — Scratch (train) : {scratch_model.r2(scratch_model.x_train, scratch_model.y_train):.4f}")
print(f"R² — Scratch (test)  : {scratch_model.r2(scratch_model.x_test,  scratch_model.y_test):.4f}")
print(f"R² — Sklearn (test)  : {r2_score(y_test, sklearn_predictions):.4f}")
print(f"MSE — Scratch (train): {train_losses[-1]:.6f}")
print(f"MSE — Sklearn (test) : {mean_squared_error(y_test, sklearn_predictions):.6f}")


#Logging management
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logging.info(f"Epochs: {scratch_model.epochs}, Learning Rate: {scratch_model.L}")
logging.info(f"Final Training MSE : {train_losses[-1]:.6f}")
logging.info(f"Final Test MSE     : {test_losses[-1]:.6f}")
logging.info(f"m: {scratch_model.m:.4f}, b: {scratch_model.b:.4f}")
logging.info(f"All training losses: {[round(l, 6) for l in train_losses]}")
logging.info(f"All test losses    : {[round(l, 6) for l in test_losses]}")
logging.info("-" * 40)