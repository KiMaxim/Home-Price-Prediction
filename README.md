# Home Price Prediction

A small machine learning project that predicts the **sale price of a
house** from its size (sq ft), number of bedrooms, and location.
The program trains three different models on the same data and lets
the user compare how each one performs:

1. **Linear Regression** (scikit-learn) — a straight-line fit.
2. **Ridge Regression** (scikit-learn) — linear regression with a
   regularisation term that penalises large weights.
3. **Neural Network** (TensorFlow / Keras) — a small multilayer
   perceptron with two hidden layers.

After training, the user can look at diagnostic plots, type in a
house's details and get a predicted price back, or re-print the
metrics.

---

## Setup

This project needs **Python 3.12** because TensorFlow does not yet
publish wheels for Python 3.13 or 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

---

## What the program does

1. Loads `home_dataset.csv` and drops any rows priced below $500,000
   (the dataset has a few obvious bad rows).
2. Asks the user for the **Ridge alpha** (regularisation strength).
   Press Enter to use the default of `1.0`.
3. Trains the three models on the same 80/20 train/test split.
4. Prints a metrics table with **R²**, **MAE** and **MSE** for each
   model.
5. Shows a menu so the user can:
   - look at four diagnostic plots,
   - get a price prediction for a house they describe,
   - re-print the metrics,
   - quit.

---

## Dataset

`home_dataset.csv` has 59 rows and four columns:

| Column | Type | Notes |
|---|---|---|
| `HouseSize` | int | Square footage (roughly 800–3500). |
| `Bedrooms` | int | 1–5. |
| `Location` | string | One of `Downtown`, `Suburb`, `Rural`. |
| `HousePrice` | int | Target. Rows below $500k are dropped. |

---

## Sample run

```
============================================================
 Home Price Prediction
 Linear vs Ridge vs Neural Network
============================================================
Loaded 58 houses priced above $500,000.
Features: ['HouseSize', 'Bedrooms', 'Location'].  Target: HousePrice.

Ridge alpha (regularisation strength) (default 1.0):

Training neural network (200 epochs)... done.

Model              R2         MAE ($)            MSE ($^2)
--------------------------------------------------------------
Linear         0.8214         215,430       78,442,103,210
Ridge          0.8208         215,902       78,711,520,003
Neural Net     0.8095         223,118       82,140,884,001

What would you like to do?
  1. Show diagnostic plots
  2. Predict a house price
  3. Show metrics again
  4. Quit
Choice: 2

House size in sq ft (or 'back'): 1800
Number of bedrooms (default 3): 4
Location (Downtown / Suburb / Rural): Downtown
  Predicted price for a 1800 sq-ft, 4-bedroom Downtown house:
    Linear      $3,072,000
    Ridge       $3,068,000
    Neural Net  $2,985,400
```

---

## Project requirements

This project was built for a class assignment with two sets of
requirements: the project description and a general programming
rubric.

**Project description**

| Requirement | Where it shows up |
|---|---|
| Regression model for house price | Three models trained in `main()` |
| Pandas, NumPy, Matplotlib, scikit-learn | All used (see imports) |
| TensorFlow (preferred for user interaction) | `build_nn()` and the NN training block |
| Evaluated with MAE and R² | `score_model()` computes both (plus MSE) |

**Programming rubric**

| # | Requirement | Where in the code |
|---|---|---|
| 1 | User interaction | menu loop, `ask_float` / `ask_int` / `ask_location`, `predict_loop` |
| 2 | A clear purpose | predict the price of a house from a few features |
| 3 | `while`, `for`, `if`, `else` | re-prompt loops, menu loop, plotting loop, input validation |
| 4 | Variables to store / change data | `results`, `ridge_alpha`, train/test splits, scaler state |
| 5 | Strings, lists, dictionaries, other collections | `menu` dict, `LOCATIONS` list, `NUMERIC_COLS` / `CATEGORY_COLS` lists, prediction dict |
| 6 | Defined and called functions | every step (load, preprocess, train, plot, predict) is a separate function |
| 7 | Source URLs in comments | top of `main.py` docstring |
| 8 | Borrowed code modified | sklearn `Pipeline` shape is from the docs; the data, menu, plots, NN wrapper and input handling are written from scratch |

---

## Model limitations: out-of-range inputs

A model can only reliably predict prices for houses **similar to the
ones it was trained on**. The training data covers houses roughly
**800–3500 sq ft**. If you enter a size far outside that range, all
three models will still give you a number, but **none of them will be
trustworthy** — and they will disagree with each other a lot.

### Example

```
House size in sq ft (or 'back'): 123
Number of bedrooms (default 3): 3
Location (Downtown / Suburb / Rural): Suburb
  Predicted price for a 123 sq-ft, 3-bedroom Suburb house:
    Linear      $1,443,890
    Ridge       $1,406,954
    Neural Net  $121,930
```

A 123 sq-ft "house" is smaller than a parking space, but Linear and
Ridge confidently predict ~$1.4M while the Neural Net says ~$122k.
All three answers are wrong, just for different reasons.

### Why Linear and Ridge return ~$1.4M

A linear model is literally a straight-line formula:

```
price ≈ intercept + w₁·size + w₂·bedrooms + w₃·location
```

At `size = 123` the contribution from the size term is tiny, but
`intercept + bedrooms + Suburb` still adds up to about $1.4M. The
model just keeps following its line past the training range — it has
no way to "know" that 123 sq ft is unrealistic.

### Why the Neural Net returns ~$122k

Neural networks do not extrapolate well:

1. **Standardisation** turns 123 sq ft into a very negative z-score
   that the network never saw while training.
2. **ReLU** activations clip negative values to zero, so on weird
   inputs many neurons "turn off" and the network loses most of its
   signal.
3. **De-normalising the output** (`pred * y_std + y_mean`) then
   converts the leftover noise into something that looks like a
   price but isn't.

### Takeaway

| Input | Linear / Ridge | Neural Net |
|---|---|---|
| Inside the training range (~800–3500 sq ft) | Trustworthy | Trustworthy |
| Outside the training range (e.g. 123) | Looks plausible but is wrong | Usually nonsense |

So linear models *hide* their ignorance by extending the line, while
neural nets visibly break. Either way, predictions outside the
training range should not be trusted.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | All of the code: data loading, models, plots, menu |
| `home_dataset.csv` | The dataset |
| `requirements.txt` | Python dependencies (needs Python 3.12) |
| `.python-version` | Tells `pyenv` / IDEs to use Python 3.12 |

---

## Sources

- scikit-learn Linear Regression and Ridge tutorials —
  https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/ ,
  https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/
- W3Schools scikit-learn tutorial —
  https://www.w3schools.com/python/python_ml_linear_regression.asp
- TensorFlow Keras regression tutorial —
  https://www.tensorflow.org/tutorials/keras/regression
