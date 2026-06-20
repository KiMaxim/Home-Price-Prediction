# Home Price Prediction

A regression project that predicts the sale price of a house from its
square footage, comparing **three models** side-by-side:

1. A **linear regression implemented from scratch** with gradient descent
   (no ML library — just NumPy).
2. **scikit-learn's `LinearRegression`** (closed-form OLS).
3. **scikit-learn's `Ridge`** (L2-regularised regression).

The program is fully interactive: the user picks training hyperparameters,
chooses what to do from a menu, and can type any house size to get a
predicted dollar price back from each model.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

The program will:

1. Load `home_dataset.csv` and filter houses above $500,000.
2. Prompt for **epochs**, **learning rate**, and **Ridge alpha** (press
   enter for sensible defaults).
3. Train all three models and print a metrics table (R² / MAE / MSE).
4. Open a menu so the user can plot results, ask for price predictions,
   re-print metrics, or quit.
5. Write a per-run log to `training_log.txt`.

---

## How the project meets the assignment specification

| Specification | Where it lives in the code |
|---|---|
| **Build a regression model to predict sale price from features** | `ScratchLinearRegression`, `LinearRegression`, `Ridge` are all trained in `main()` |
| **Tools: Python, Pandas, scikit-learn (Linear + Ridge), NumPy, Matplotlib** | All five used — see `import` block at the top of `main.py` |
| **A trained model evaluated with MAE and R²** | `collect_metrics()` returns R², MAE, and MSE for every model; printed by `print_metrics_table()` |
| **Project outcome shown to the user** | Metrics table printed after training, side-by-side plot from menu option 1, dollar predictions from menu option 2 |

---

## How the project meets each rubric item

| # | Rubric requirement | Where / how |
|---|---|---|
| 1 | Program has interaction with the user (prompt + input + outcome shown) | `ask_int` / `ask_float` collect hyperparameters; menu in `main()` accepts choices; `interactive_prediction_loop` lets the user type house sizes and prints dollar estimates back |
| 2 | Has a purpose (practical or fun) | Practical: estimate a home's sale price from its size, and compare a hand-written model against industry-standard ones |
| 3 | Uses `while`, `for`, `if`, `else` | `while` — main menu loop, prediction loop, `ask_*` re-prompt loops. `for` — epoch loop in `fit()`, metric/menu rendering. `if` / `elif` / `else` — input validation, menu dispatch, NaN/inf check after training |
| 4 | Uses variables to retain and alter data | Model parameters `self.m`, `self.b`; running `train_losses` / `test_losses`; scaler `stats`; per-run `results` |
| 5 | Uses strings, lists, dictionaries, and other collections | `dict` — `stats`, `menu`, per-model metric record, `predictions`. `list` — `train_losses`, `test_losses`, `results`. `set` — `exit_words` for fast membership check. f-strings everywhere |
| 6 | Defines and calls functions | `load_data`, `standardize`, `collect_metrics`, `print_metrics_table`, `plot_everything`, `ask_float`, `ask_int`, `predict_dollars`, `interactive_prediction_loop`, `configure_logging`, `log_run`, `main`, plus the `ScratchLinearRegression` class with `_step` / `fit` / `predict` / `mse` / `r2` |
| 7 | Identifies sources in comments | Top-of-file docstring lists the URLs of every reference consulted |
| 8 | Modifies any borrowed code to make it original | The scratch-model derivation was adapted from the Kaggle reference, but the class structure, hyperparameter handling, NaN-divergence detection, and integration with sklearn/Ridge are original |

---

## Code organization

Every concern lives in a small, single-purpose function — no top-level
script soup, no copy-pasted blocks:

```
load_data / standardize          → data prep
ScratchLinearRegression          → from-scratch model
collect_metrics / print_metrics  → evaluation
plot_everything                  → matplotlib visualization
ask_int / ask_float              → robust numeric input
predict_dollars                  → un-scale predictions to real $
interactive_prediction_loop      → per-house-size prediction REPL
configure_logging / log_run      → run logging
main                             → orchestration + menu
```

Identifier names are descriptive (`learning_rate`, `train_losses`,
`back_to_dollars`, `size_scaled`), repetition is factored out
(`ask_int` / `ask_float` share the same re-prompt loop pattern, the
three model panels in `plot_everything` use the same plotting recipe),
and the control flow mirrors the structure of the problem (`while` for
loops that depend on user state, `for` for fixed iteration, `if`/`elif`
for menu dispatch).

---

## Sample interaction

```
========================================================
 Home Price Prediction — Linear vs Ridge vs Scratch
========================================================
Loaded 49 houses priced above $500,000.
Epochs for the scratch model (default 1000): 2000
Learning rate for the scratch model (default 0.01):
Ridge regularisation alpha (default 1.0):

Model                        R²         MAE         MSE
--------------------------------------------------------
Scratch (test)           0.7421      0.4123      0.2987
Sklearn (test)           0.7423      0.4119      0.2985
Ridge (test)             0.7420      0.4124      0.2988

What would you like to do?
  1. Show plots
  2. Predict a house price from a size
  3. Re-print metrics
  4. Quit
Choice: 2

Enter a house size in square feet to get price estimates.
Type 'back' (or just press enter) to return to the main menu.

House size (sq ft, or 'back'): 1800
  Predictions for a 1800 sq-ft house:
    Scratch    $2,431,000
    Sklearn    $2,432,000
    Ridge      $2,430,000
```

---

## Robustness

The program is defensive about every external interaction:

- **Bad numeric input** (e.g. `"abc"` for epochs) is re-prompted, not
  silently defaulted.
- **Non-positive hyperparameters** (epochs ≤ 0, LR ≤ 0) are rejected.
- **Out-of-menu choices** print a helpful message and re-show the menu.
- **Diverged training** (NaN/inf loss from too-large LR) prints a
  warning suggesting a smaller learning rate.
- **Ctrl-C / Ctrl-D** prints `"Interrupted. Goodbye."` instead of a
  traceback.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | All code: data prep, models, evaluation, plotting, menu |
| `home_dataset.csv` | Input data — columns `HouseSize`, `HousePrice` |
| `requirements.txt` | Python dependencies |
| `training_log.txt` | Auto-appended per-run log (epochs, LR, final losses, metrics) |

---

## Future work

- Add more features (bedrooms, location) when richer data is available;
  the scaler/prediction pipeline already supports multi-feature input
  with minimal change.
- K-fold cross-validation instead of a single 80/20 split.
- Hyperparameter search over `learning_rate` and `alpha`.

---

## Sources

Referenced for ideas only — code in this repo is original:

- Linear regression from scratch (Kaggle):
  https://www.kaggle.com/code/fareselmenshawii/linear-regression-from-scratch
- scikit-learn linear regression tutorial (W3Schools):
  https://www.w3schools.com/python/python_ml_linear_regression.asp
- scikit-learn `Ridge` API reference:
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Z-score standardisation (Wikipedia):
  https://en.wikipedia.org/wiki/Standard_score
