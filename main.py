"""
Home Price Prediction
---------------------
Predicts the sale price of a house from three features (HouseSize, Bedrooms,
Location) using two scikit-learn regression models:
  1. LinearRegression (closed-form OLS).
  2. Ridge regression (L2-regularised, user picks alpha).

Both models are wrapped in a single sklearn Pipeline that standardises the
numeric features and one-hot encodes the categorical Location feature, so
the user can type real-world inputs like "1800 sq ft, 3 bedrooms, Downtown"
and get a dollar prediction back.

Sources consulted (ideas were adapted, not copied verbatim):
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
  https://scikit-learn.org/stable/modules/compose.html
  https://www.w3schools.com/python/python_ml_linear_regression.asp
"""

import logging

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = 'home_dataset.csv'
PRICE_FLOOR = 500_000

NUMERIC_FEATURES = ['HouseSize', 'Bedrooms']
CATEGORICAL_FEATURES = ['Location']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'HousePrice'

VALID_LOCATIONS = ('Downtown', 'Suburb', 'Rural')


# ----------------------------- data loading ------------------------------- #

def load_data(path: str) -> pandas.DataFrame:
    """Load the CSV and drop unrealistic rows (price below the floor)."""
    df = pandas.read_csv(path)
    mask = df['HousePrice'] > PRICE_FLOOR
    return df.loc[mask].copy()


# --------------------------- model construction --------------------------- #

def build_pipeline(model) -> Pipeline:
    """
    Compose preprocessing + a regression model into a single estimator.
    StandardScaler for numeric columns, one-hot for the categorical Location.
    """
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),                       NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ])
    return Pipeline([('preprocess', preprocessor), ('regressor', model)])


# ----------------------------- metrics helpers ---------------------------- #

def collect_metrics(name: str, y_true, y_pred) -> dict:
    """Bundle the rubric metrics (R² + MAE) plus MSE into a dict."""
    return {
        'name': name,
        'R2':   r2_score(y_true, y_pred),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MSE':  mean_squared_error(y_true, y_pred),
    }


def print_metrics_table(results: list) -> None:
    print()
    print(f"{'Model':<18}{'R²':>10}{'MAE ($)':>16}{'MSE ($²)':>22}")
    print("-" * 66)
    for r in results:
        print(f"{r['name']:<18}{r['R2']:>10.4f}{r['MAE']:>16,.0f}{r['MSE']:>22,.0f}")
    print()


# ------------------------------- plotting --------------------------------- #

def plot_diagnostics(y_test, linear_pred, ridge_pred, df) -> None:
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    lo, hi = float(y_test.min()), float(y_test.max())

    # Predicted vs Actual — Linear
    ax = axes[0, 0]
    ax.scatter(y_test, linear_pred, color='blue', alpha=0.7)
    ax.plot([lo, hi], [lo, hi], color='red', linestyle='--', label='y = x')
    ax.set_title('Linear Regression — Predicted vs Actual')
    ax.set_xlabel('Actual price ($)')
    ax.set_ylabel('Predicted price ($)')
    ax.legend()

    # Predicted vs Actual — Ridge
    ax = axes[0, 1]
    ax.scatter(y_test, ridge_pred, color='green', alpha=0.7)
    ax.plot([lo, hi], [lo, hi], color='red', linestyle='--', label='y = x')
    ax.set_title('Ridge Regression — Predicted vs Actual')
    ax.set_xlabel('Actual price ($)')
    ax.set_ylabel('Predicted price ($)')
    ax.legend()

    # HouseSize vs HousePrice coloured by Location
    ax = axes[1, 0]
    colors = {'Downtown': 'red', 'Suburb': 'blue', 'Rural': 'green'}
    for loc in VALID_LOCATIONS:
        sub = df[df['Location'] == loc]
        ax.scatter(sub['HouseSize'], sub['HousePrice'],
                   color=colors[loc], label=loc, alpha=0.7)
    ax.set_title('HouseSize vs HousePrice by Location')
    ax.set_xlabel('House size (sq ft)')
    ax.set_ylabel('House price ($)')
    ax.legend()

    # Average HousePrice by bedroom count
    ax = axes[1, 1]
    by_bed = df.groupby('Bedrooms')['HousePrice'].mean()
    ax.bar(by_bed.index, by_bed.values, color='purple', alpha=0.7)
    ax.set_title('Average HousePrice by Bedroom Count')
    ax.set_xlabel('Bedrooms')
    ax.set_ylabel('Average price ($)')
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


# ---------------------------- user interaction ---------------------------- #

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


def ask_location() -> str:
    options_str = " / ".join(VALID_LOCATIONS)
    while True:
        raw = input(f"Location ({options_str}): ").strip().title()
        if raw in VALID_LOCATIONS:
            return raw
        print(f"  Pick one of: {options_str}")


def predict_dollars(size: float, bedrooms: int, location: str,
                    linear_model: Pipeline, ridge_model: Pipeline) -> dict:
    """Run a single user-supplied house through both models."""
    row = pandas.DataFrame([{
        'HouseSize': size,
        'Bedrooms':  bedrooms,
        'Location':  location,
    }])
    return {
        'Linear': float(linear_model.predict(row)[0]),
        'Ridge':  float(ridge_model.predict(row)[0]),
    }


def interactive_prediction_loop(linear_model: Pipeline, ridge_model: Pipeline) -> None:
    exit_words = {"", "b", "back", "q", "quit", "exit", "menu"}
    print("\nEnter the house features to get a price estimate.")
    print("Type 'back' at the size prompt to return to the main menu.\n")
    while True:
        raw = input("House size in sq ft (or 'back'): ").strip().lower()
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

        bedrooms = ask_int("Number of bedrooms", default=3)
        location = ask_location()

        predictions = predict_dollars(size, bedrooms, location,
                                      linear_model, ridge_model)
        print(f"  Predicted price for a {size:.0f} sq-ft, "
              f"{bedrooms}-bedroom {location} house:")
        for model_name, dollars in predictions.items():
            print(f"    {model_name:<8} ${dollars:,.0f}")
        print()


# -------------------------------- logging --------------------------------- #

def configure_logging() -> None:
    logging.basicConfig(
        filename='training_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def log_run(ridge_alpha: float, results: list) -> None:
    logging.info(f"Ridge alpha: {ridge_alpha}")
    for r in results:
        logging.info(f"{r['name']:<18} R²={r['R2']:.4f}  MAE={r['MAE']:.2f}  MSE={r['MSE']:.2f}")
    logging.info("-" * 40)


# --------------------------------- main ----------------------------------- #

def main() -> None:
    configure_logging()

    print("=" * 60)
    print(" Home Price Prediction — Linear vs Ridge Regression")
    print("=" * 60)

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} houses priced above ${PRICE_FLOOR:,}.")
    print(f"Features: {ALL_FEATURES}.  Target: {TARGET}.\n")

    ridge_alpha = ask_float("Ridge regularisation alpha", 1.0)

    X = df[ALL_FEATURES]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    y_test = numpy.asarray(y_test)

    linear_model = build_pipeline(LinearRegression())
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)

    ridge_model = build_pipeline(Ridge(alpha=ridge_alpha))
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    results = [
        collect_metrics('Linear (test)', y_test, linear_pred),
        collect_metrics('Ridge (test)',  y_test, ridge_pred),
    ]
    print_metrics_table(results)
    log_run(ridge_alpha, results)

    menu = {
        '1': 'Show diagnostic plots',
        '2': 'Predict a house price',
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
            plot_diagnostics(y_test, linear_pred, ridge_pred, df)
        elif choice == '2':
            interactive_prediction_loop(linear_model, ridge_model)
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
