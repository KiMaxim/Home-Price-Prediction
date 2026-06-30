"""
Home Price Prediction Project
-----------------------------
Predicts the price of a house from its size, number of bedrooms and
location. Compares three models: Linear Regression, Ridge Regression
and a small TensorFlow neural network.

Sources I used while learning the ideas:
https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/
https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/
https://www.w3schools.com/python/python_ml_linear_regression.asp
https://www.tensorflow.org/tutorials
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# settings
DATA_FILE = "home_dataset.csv"
PRICE_FLOOR = 500_000

NUMERIC_COLS = ["HouseSize", "Bedrooms"]
CATEGORY_COLS = ["Location"]
FEATURE_COLS = NUMERIC_COLS + CATEGORY_COLS
TARGET_COL = "HousePrice"

LOCATIONS = ["Downtown", "Suburb", "Rural"]

# neural network hyper-parameters
NN_EPOCHS = 200
NN_BATCH = 8
RANDOM_SEED = 42


# preprocessing section
def load_data(path):
    # read the CSV and drop rows that look like bad data
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Could not find the data file '{path}'. "
              f"Make sure it's in the same folder as this script.")
        raise SystemExit(1)
    except pd.errors.EmptyDataError:
        print(f"The data file '{path}' is empty.")
        raise SystemExit(1)
    except pd.errors.ParserError as err:
        print(f"Could not parse '{path}' as a CSV: {err}")
        raise SystemExit(1)
    df = df[df[TARGET_COL] > PRICE_FLOOR]
    return df


def make_preprocessor():
    # scale the numeric columns and one-hot encode the Location column
    return ColumnTransformer([
        ("scale", StandardScaler(), NUMERIC_COLS),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORY_COLS),
    ])


def make_regression_pipeline(regressor):
    # preprocessor + regressor; shared by Linear and Ridge branches
    return Pipeline([
        ("prep", make_preprocessor()),
        ("reg", regressor),
    ])


# neural network
def build_nn(input_dim):
    # small MLP: two hidden ReLU layers and one linear output for the price
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mae"],
    )
    return model


# metrics
def score_model(name, y_true, y_pred):
    # compute R2, MAE and MSE and return them as a dictionary
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"name": name, "R2": r2, "MAE": mae, "MSE": mse}


def print_results(results):
    print()
    print(f"{'Model':<14}{'R2':>10}{'MAE ($)':>16}{'MSE ($^2)':>22}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:<14}{r['R2']:>10.4f}{r['MAE']:>16,.0f}{r['MSE']:>22,.0f}")
    print()


def _plot_pred_vs_actual(ax, y_test, y_pred, title, color):
    # one model's predicted-vs-actual scatter with the perfect-fit reference line
    lo = float(np.min(y_test))
    hi = float(np.max(y_test))
    ax.scatter(y_test, y_pred, color=color, alpha=0.6, label="predicted")
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", label="perfect")
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title(title)
    ax.legend()


def show_plots(y_test, lin_pred, rid_pred, nn_pred, nn_history, df):
    _, axs = plt.subplots(2, 3, figsize=(16, 9))

    # one predicted-vs-actual scatter per model
    _plot_pred_vs_actual(axs[0, 0], y_test, lin_pred,
                         "Linear: Predicted vs Actual", "blue")
    _plot_pred_vs_actual(axs[0, 1], y_test, rid_pred,
                         "Ridge: Predicted vs Actual", "green")
    _plot_pred_vs_actual(axs[0, 2], y_test, nn_pred,
                         "Neural Net: Predicted vs Actual", "orange")

    # neural network training loss
    ax = axs[1, 0]
    ax.plot(nn_history.history["loss"], color="orange", label="train")
    if "val_loss" in nn_history.history:
        ax.plot(nn_history.history["val_loss"], color="red",
                linestyle="--", label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Neural Net training loss")
    ax.legend()

    # house size vs house price
    ax = axs[1, 1]
    colors = {"Downtown": "red", "Suburb": "blue", "Rural": "green"}
    for loc in LOCATIONS:
        sub = df[df["Location"] == loc]
        ax.scatter(sub["HouseSize"], sub["HousePrice"],
                   color=colors[loc], alpha=0.7, label=loc)
    ax.set_xlabel("House size (sq ft)")
    ax.set_ylabel("Price ($)")
    ax.set_title("Size vs Price by Location")
    ax.legend()

    # average price by bedroom count
    ax = axs[1, 2]
    by_bed = df.groupby("Bedrooms")["HousePrice"].mean()
    ax.bar(by_bed.index, by_bed.values, color="purple", alpha=0.7)
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Average price ($)")
    ax.set_title("Average price by bedroom count")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


# user input
def ask_float(prompt, default):
    # keep asking until the user gives us a positive number (blank = default)
    while True:
        raw = input(f"{prompt} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  That's not a number, try again.")
            continue
        if value <= 0:
            print("  The number has to be positive.")
            continue
        return value


def ask_int(prompt, default):
    while True:
        raw = input(f"{prompt} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  That's not a whole number, try again.")
            continue
        if value <= 0:
            print("  The number has to be positive.")
            continue
        return value


def ask_location():
    options = " / ".join(LOCATIONS)
    while True:
        raw = input(f"Location ({options}): ").strip().title()
        if raw in LOCATIONS:
            return raw
        print(f"  Please pick one of: {options}")


# interactive prediction loop
def predict_loop(lin_model, rid_model, nn_model, nn_pre, y_mean, y_std):
    print("\nEnter the house details and I'll predict the price.")
    print("Type 'back' at the size prompt to return to the main menu.\n")

    while True:
        raw = input("House size in sq ft (or 'back'): ").strip().lower()
        if raw in ("", "back", "b", "q", "quit", "exit"):
            print("Returning to main menu.\n")
            return

        try:
            size = float(raw)
        except ValueError:
            print("  Please enter a number (e.g. 1800).")
            continue
        if size <= 0:
            print("  The size has to be positive.")
            continue

        bedrooms = ask_int("Number of bedrooms", 3)
        location = ask_location()

        # build a 1-row DataFrame so the sklearn pipelines see the right columns
        row = pd.DataFrame([{
            "HouseSize": size,
            "Bedrooms": bedrooms,
            "Location": location,
        }])

        lin_price = float(lin_model.predict(row)[0])
        rid_price = float(rid_model.predict(row)[0])

        # the neural net needs the preprocessed row, and its output is scaled
        nn_row = nn_pre.transform(row).astype("float32")
        nn_scaled = float(nn_model.predict(nn_row, verbose=0).flatten()[0])
        nn_price = nn_scaled * y_std + y_mean

        predictions = {
            "Linear": lin_price,
            "Ridge": rid_price,
            "Neural Net": nn_price,
        }
        print(f"  Predicted price for a {size:.0f} sq-ft, "
              f"{bedrooms}-bedroom {location} house:")
        for name, price in predictions.items():
            print(f"    {name:<11} ${price:,.0f}")
        print()


# main program
def main():
    print("=" * 60)
    print(" Home Price Prediction")
    print(" Linear vs Ridge vs Neural Network")
    print("=" * 60)

    # load the data
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} houses priced above ${PRICE_FLOOR:,}.")
    print(f"Features: {FEATURE_COLS}.  Target: {TARGET_COL}.\n")

    # let the user pick the ridge regularisation strength
    ridge_alpha = ask_float("Ridge alpha (regularisation strength)", 1.0)

    # split into features (X) and target (y), then into train/test sets
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED,
    )
    y_test = np.asarray(y_test)

    # linear regression
    lin_model = make_regression_pipeline(LinearRegression())
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)

    # ridge regression
    rid_model = make_regression_pipeline(Ridge(alpha=ridge_alpha))
    rid_model.fit(X_train, y_train)
    rid_pred = rid_model.predict(X_test)

    # fit the preprocessor once and reuse it
    nn_pre = make_preprocessor()
    X_train_nn = nn_pre.fit_transform(X_train).astype("float32")
    X_test_nn = nn_pre.transform(X_test).astype("float32")

    # normalise the target so the network trains in a sensible range
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std == 0:
        y_std = 1.0
    y_train_nn = ((y_train - y_mean) / y_std).astype("float32")

    print(f"\nTraining neural network ({NN_EPOCHS} epochs)... ", end="", flush=True)
    nn_model = build_nn(X_train_nn.shape[1])
    nn_history = nn_model.fit(
        X_train_nn, y_train_nn,
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH,
        validation_split=0.1,
        verbose=0,
    )
    print("done.")

    # the NN predicts in the scaled range, so convert back to dollars
    nn_pred_scaled = nn_model.predict(X_test_nn, verbose=0).flatten()
    nn_pred = nn_pred_scaled * y_std + y_mean

    # collect and display metrics
    results = [
        score_model("Linear", y_test, lin_pred),
        score_model("Ridge", y_test, rid_pred),
        score_model("Neural Net", y_test, nn_pred),
    ]
    print_results(results)

    # main menu
    menu = {
        "1": "Show diagnostic plots",
        "2": "Predict a house price",
        "3": "Show metrics again",
        "4": "Quit",
    }
    while True:
        print("What would you like to do?")
        for key, label in menu.items():
            print(f"  {key}. {label}")
        choice = input("Choice: ").strip().lower()

        if choice == "1":
            show_plots(y_test, lin_pred, rid_pred, nn_pred, nn_history, df)
        elif choice == "2":
            predict_loop(lin_model, rid_model, nn_model, nn_pre, y_mean, y_std)
        elif choice == "3":
            print_results(results)
        elif choice == "4" or choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print(f"  '{choice}' isn't on the menu, please pick 1-4.\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Goodbye!")
