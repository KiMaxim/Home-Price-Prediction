"""
Home Price Prediction
Predicts house sale price from house size using three regression approaches:
  1. A from-scratch linear regression trained with gradient descent.
  2. Scikit-learn's LinearRegression.
  3. Scikit-learn's Ridge regression.

The user picks epochs / learning rate, can plot results, and can type in any
house size to get a price prediction back from each model.

Sources consulted (ideas were adapted, not copied verbatim):
https://www.kaggle.com/code/fareselmenshawii/linear-regression-from-scratch
https://www.w3schools.com/python/python_ml_linear_regression.asp
"""

import logging

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = 'home_dataset.csv'
PRICE_FLOOR = 500_000


# -------------------------- data + scaling helpers -------------------------- #

def load_data(path: str) -> pandas.DataFrame:
    """Load the CSV and drop rows below the price floor."""
    df = pandas.read_csv(path)
    mask = df['HousePrice'] > PRICE_FLOOR
    return df.loc[mask].copy()


def standardize(df: pandas.DataFrame) -> dict:
    """
    Z-score scale HouseSize and HousePrice and return the scaling stats so we
    can convert user input / model output back to real dollars and sq-ft later.
    """
    stats = {
        'size_mean':  df['HouseSize'].mean(),
        'size_std':   df['HouseSize'].std(),
        'price_mean': df['HousePrice'].mean(),
        'price_std':  df['HousePrice'].std(),
    }
    df['HouseSize_Scaled']  = (df['HouseSize']  - stats['size_mean'])  / stats['size_std']
    df['HousePrice_Scaled'] = (df['HousePrice'] - stats['price_mean']) / stats['price_std']
    return stats


# -------------------------- scratch implementation -------------------------- #

class ScratchLinearRegression:
    """Linear regression implemented from scratch using gradient descent."""

    def __init__(self, learning_rate: float, epochs: int) -> None:
        self.L = learning_rate
        self.epochs = epochs
        self.m = 0.0
        self.b = 0.0

    def _step(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        n = len(x)
        y_pred = self.m * x + self.b
        m_grad = (-2 / n) * numpy.sum(x * (y - y_pred))
        b_grad = (-2 / n) * numpy.sum(y - y_pred)
        self.m -= self.L * m_grad
        self.b -= self.L * b_grad

    def fit(self, x: numpy.ndarray, y: numpy.ndarray,
            x_test: numpy.ndarray, y_test: numpy.ndarray) -> dict:
        train_losses, test_losses = [], []
        for _ in range(self.epochs):
            self._step(x, y)
            train_losses.append(self.mse(x, y))
            test_losses.append(self.mse(x_test, y_test))
        return {'train_losses': train_losses, 'test_losses': test_losses}

    def predict(self, x):
        return self.m * x + self.b

    def mse(self, x, y) -> float:
        return float(numpy.mean((y - self.predict(x)) ** 2))

    def r2(self, x, y) -> float:
        y_pred = self.predict(x)
        rss = numpy.sum((y - y_pred) ** 2)
        tss = numpy.sum((y - numpy.mean(y)) ** 2)
        return float(1 - rss / tss)


# ----------------------------- metrics helpers ----------------------------- #

def collect_metrics(name: str, y_true, y_pred) -> dict:
    """Bundle the three rubric metrics into a dict for one model."""
    return {
        'name': name,
        'R2':   r2_score(y_true, y_pred),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MSE':  mean_squared_error(y_true, y_pred),
    }


def print_metrics_table(results: list) -> None:
    """Pretty-print metrics for every model in `results`."""
    print()
    print(f"{'Model':<22}{'R²':>10}{'MAE':>12}{'MSE':>12}")
    print("-" * 56)
    for r in results:
        print(f"{r['name']:<22}{r['R2']:>10.4f}{r['MAE']:>12.4f}{r['MSE']:>12.4f}")
    print()


# ------------------------------- plotting --------------------------------- #

def plot_everything(x_test, y_test,
                    scratch, sklearn_pred, ridge_pred,
                    train_losses, test_losses) -> None:
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    sorted_idx = x_test.argsort()

    # Scratch
    ax = axes[0, 0]
    ax.scatter(x_test, y_test, marker='^', color='blue', label='Actual')
    ax.plot(x_test[sorted_idx], scratch.predict(x_test)[sorted_idx],
            color='red', linewidth=2, label='Scratch Predicted')
    ax.set_title('Scratch Model')
    ax.set_xlabel('House Size (Scaled)')
    ax.set_ylabel('House Price (Scaled)')
    ax.legend()

    # Sklearn linear
    ax = axes[0, 1]
    ax.scatter(x_test, y_test, marker='^', color='blue', label='Actual')
    ax.plot(x_test[sorted_idx], sklearn_pred[sorted_idx],
            color='red', linewidth=2, label='Sklearn Predicted')
    ax.set_title('Sklearn Linear Regression')
    ax.set_xlabel('House Size (Scaled)')
    ax.set_ylabel('House Price (Scaled)')
    ax.legend()

    # Ridge
    ax = axes[1, 0]
    ax.scatter(x_test, y_test, marker='^', color='blue', label='Actual')
    ax.plot(x_test[sorted_idx], ridge_pred[sorted_idx],
            color='red', linewidth=2, label='Ridge Predicted')
    ax.set_title('Ridge Regression')
    ax.set_xlabel('House Size (Scaled)')
    ax.set_ylabel('House Price (Scaled)')
    ax.legend()

    # Loss curves
    ax = axes[1, 1]
    ax.plot(train_losses, label='Train Loss', color='blue')
    ax.plot(test_losses,  label='Test Loss',  color='red')
    ax.set_title('Scratch Model Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------- user interaction ----------------------------- #

def ask_float(prompt: str, default: float, must_be_positive: bool = True) -> float:
    """Re-prompt until the user gives a valid float (or hits enter for default)."""
    while True:
        raw = input(f"{prompt} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print(f"  '{raw}' isn't a number. Try again or press enter for {default}.")
            continue
        if must_be_positive and value <= 0:
            print(f"  Value has to be positive. Try again or press enter for {default}.")
            continue
        return value


def ask_int(prompt: str, default: int, must_be_positive: bool = True) -> int:
    """Re-prompt until the user gives a valid int (or hits enter for default)."""
    while True:
        raw = input(f"{prompt} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"  '{raw}' isn't an integer. Try again or press enter for {default}.")
            continue
        if must_be_positive and value <= 0:
            print(f"  Value has to be positive. Try again or press enter for {default}.")
            continue
        return value


def predict_dollars(size_sqft: float, scratch, sklearn_lin, ridge, stats: dict) -> dict:
    """Run a single user-supplied house size through all three models."""
    size_scaled = (size_sqft - stats['size_mean']) / stats['size_std']
    arr = numpy.array([[size_scaled]])

    def back_to_dollars(scaled_price):
        return scaled_price * stats['price_std'] + stats['price_mean']

    return {
        'Scratch':  float(back_to_dollars(scratch.predict(size_scaled))),
        'Sklearn':  float(back_to_dollars(sklearn_lin.predict(arr)[0])),
        'Ridge':    float(back_to_dollars(ridge.predict(arr)[0])),
    }


def interactive_prediction_loop(scratch, sklearn_lin, ridge, stats: dict) -> None:
    """Let the user type house sizes and get back dollar estimates."""
    exit_words = {"", "b", "back", "q", "quit", "exit", "menu"}
    print("\nEnter a house size in square feet to get price estimates.")
    print("Type 'back' (or just press enter) to return to the main menu.\n")
    while True:
        raw = input("House size (sq ft, or 'back'): ").strip().lower()
        if raw in exit_words:
            print("Returning to main menu.\n")
            return
        try:
            size = float(raw)
        except ValueError:
            print("  Please enter a number (e.g. 1800), or 'back' to return.")
            continue
        if size <= 0:
            print("  House size has to be positive.")
            continue

        predictions = predict_dollars(size, scratch, sklearn_lin, ridge, stats)
        print(f"  Predictions for a {size:.0f} sq-ft house:")
        for model_name, dollars in predictions.items():
            print(f"    {model_name:<10} ${dollars:,.0f}")
        print()


# -------------------------------- logging --------------------------------- #

def configure_logging() -> None:
    logging.basicConfig(
        filename='training_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def log_run(scratch, train_losses, test_losses, results) -> None:
    logging.info(f"Epochs: {scratch.epochs}, Learning Rate: {scratch.L}")
    logging.info(f"Final Training MSE : {train_losses[-1]:.6f}")
    logging.info(f"Final Test MSE     : {test_losses[-1]:.6f}")
    logging.info(f"m: {scratch.m:.4f}, b: {scratch.b:.4f}")
    for r in results:
        logging.info(f"{r['name']:<22} R²={r['R2']:.4f}  MAE={r['MAE']:.4f}  MSE={r['MSE']:.4f}")
    logging.info("-" * 40)


# --------------------------------- main ----------------------------------- #

def main() -> None:
    configure_logging()

    print("=" * 56)
    print(" Home Price Prediction — Linear vs Ridge vs Scratch")
    print("=" * 56)

    df = load_data(DATA_PATH)
    stats = standardize(df)
    print(f"Loaded {len(df)} houses priced above ${PRICE_FLOOR:,}.")

    epochs        = ask_int("Epochs for the scratch model", 1000)
    learning_rate = ask_float("Learning rate for the scratch model", 0.01)
    ridge_alpha   = ask_float("Ridge regularisation alpha", 1.0)

    split = train_test_split(
        df['HouseSize_Scaled'].values,
        df['HousePrice_Scaled'].values,
        test_size=0.2,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = (numpy.asarray(a) for a in split)

    # Scratch
    scratch = ScratchLinearRegression(learning_rate=learning_rate, epochs=epochs)
    losses = scratch.fit(x_train, y_train, x_test, y_test)
    train_losses, test_losses = losses['train_losses'], losses['test_losses']
    if numpy.isnan(train_losses[-1]) or numpy.isinf(train_losses[-1]):
        print("\n  Warning: the scratch model diverged (loss is NaN/inf).")
        print("  Try a smaller learning rate (e.g. 0.001 - 0.05).\n")

    # Sklearn linear
    sklearn_lin = LinearRegression()
    sklearn_lin.fit(x_train.reshape(-1, 1), y_train)
    sklearn_pred = sklearn_lin.predict(x_test.reshape(-1, 1))

    # Ridge
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(x_train.reshape(-1, 1), y_train)
    ridge_pred = ridge.predict(x_test.reshape(-1, 1))

    results = [
        collect_metrics('Scratch (test)',  y_test, scratch.predict(x_test)),
        collect_metrics('Sklearn (test)',  y_test, sklearn_pred),
        collect_metrics('Ridge (test)',    y_test, ridge_pred),
    ]
    print_metrics_table(results)
    log_run(scratch, train_losses, test_losses, results)

    # Menu loop — while + if/elif/else
    menu = {
        '1': 'Show plots',
        '2': 'Predict a house price from a size',
        '3': 'Re-print metrics',
        '4': 'Quit',
    }
    while True:
        print("What would you like to do?")
        for key, label in menu.items():
            print(f"  {key}. {label}")
        choice = input("Choice: ").strip()

        choice_lc = choice.lower()
        if choice == '1':
            plot_everything(x_test, y_test,
                            scratch, sklearn_pred, ridge_pred,
                            train_losses, test_losses)
        elif choice == '2':
            interactive_prediction_loop(scratch, sklearn_lin, ridge, stats)
        elif choice == '3':
            print_metrics_table(results)
        elif choice == '4' or choice_lc in ('q', 'quit', 'exit'):
            print("Goodbye.")
            break
        else:
            print(f"  '{choice}' isn't on the menu — pick 1–4.\n")


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Goodbye.")
