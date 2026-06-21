# Home Price Prediction

A regression project that predicts the **sale price of a house** from its
**square footage, number of bedrooms, and location**, comparing two
scikit-learn models:

1. **`LinearRegression`** — ordinary least squares.
2. **`Ridge`** — L2-regularised regression (user picks `alpha`).

Both models share a single preprocessing pipeline: `StandardScaler` for the
numeric features (`HouseSize`, `Bedrooms`) and `OneHotEncoder` for the
categorical `Location`. The user can train the models, see diagnostic
plots, and type in a house's features to get a predicted price back.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

The program will:

1. Load `home_dataset.csv` and filter houses above $500,000.
2. Prompt for the **Ridge alpha** (press enter for the default of `1.0`).
3. Train both models and print a metrics table with **R²**, **MAE**, and
   **MSE** for each.
4. Open a menu so the user can show diagnostic plots, ask for price
   predictions, re-print metrics, or quit.
5. Append a per-run record to `training_log.txt`.

---

## Dataset

`home_dataset.csv` — 59 rows, four columns:

| Column | Type | Notes |
|---|---|---|
| `HouseSize` | int | Square footage. |
| `Bedrooms` | int | 1–5. |
| `Location` | string | One of `Downtown`, `Suburb`, `Rural`. |
| `HousePrice` | int | Target. Rows below $500k are dropped at load time. |

---

## How the project meets the specification

| Specification line | Where in the project |
|---|---|
| "Build a regression model to predict sale price from features (e.g. square footage, number of bedrooms, location)" | `home_dataset.csv` has all three features; `build_pipeline()` wires them into both models |
| "Tools & Technologies: Python, Pandas, scikit-learn (Linear Regression, Ridge Regression), NumPy, Matplotlib" | All five used — see the `import` block at the top of `main.py` |
| "A trained regression model, evaluated using metrics like MAE and R² score" | `collect_metrics()` returns R² + MAE (+ MSE); `print_metrics_table()` prints them per model |
| "…to show how well it predicts prices" | Menu option 1 plots Predicted vs Actual diagnostics; menu option 2 prints a dollar prediction for user-supplied features |

---

## How the project meets the assignment rubric

| # | Rubric requirement | Where / how |
|---|---|---|
| 1 | Program interacts with the user (prompt + input + outcome shown) | `ask_float`/`ask_int`/`ask_location` collect input with re-prompt; main menu accepts choices; `interactive_prediction_loop` lets the user type a house's features and prints dollar estimates back |
| 2 | Has a purpose | Practical: estimate a home's sale price from its size, bedrooms, and location, and compare a regularised vs unregularised model |
| 3 | Uses `while`, `for`, `if`, `else` | `while` — main menu, prediction loop, all `ask_*` re-prompt loops. `for` — metric/menu rendering, location loop in plotting. `if` / `elif` / `else` — input validation, menu dispatch |
| 4 | Uses variables to retain and alter data | `results` list, scaler/encoder fitted state inside the `Pipeline`, per-run `ridge_alpha` |
| 5 | Uses strings, lists, dictionaries, other collections | `dict` — per-model metric record, `menu`, `predictions`, `colors`. `list` — `NUMERIC_FEATURES`, `ALL_FEATURES`, `results`. `tuple` — `VALID_LOCATIONS`. `set` — `exit_words` for fast membership. f-strings everywhere |
| 6 | Defines and calls functions | `load_data`, `build_pipeline`, `collect_metrics`, `print_metrics_table`, `plot_diagnostics`, `ask_float`, `ask_int`, `ask_location`, `predict_dollars`, `interactive_prediction_loop`, `configure_logging`, `log_run`, `main` |
| 7 | Identifies sources in comments | Top-of-file docstring lists the URLs of every reference consulted |
| 8 | Modifies any borrowed code to make it original | Standard sklearn snippets (`ColumnTransformer`, `Pipeline`) are referenced for shape only; the dataset, menu, prediction loop, plotting layout, validation, and logging are original |

---

## Code organization

Every concern lives in a small, single-purpose function — no top-level
script soup, no copy-pasted blocks:

```
load_data                       → CSV load + filter
build_pipeline                  → scaler + one-hot + regressor
collect_metrics / print_metrics → evaluation
plot_diagnostics                → matplotlib visualisation
ask_float / ask_int / ask_location → robust user input with re-prompt
predict_dollars                 → wrap a single house in a DataFrame and predict
interactive_prediction_loop     → per-house prediction REPL
configure_logging / log_run     → run logging
main                            → orchestration + menu
```

Identifier names are descriptive (`ridge_alpha`, `linear_pred`,
`predict_dollars`, `valid_locations`), repetition is factored out
(`ask_int` / `ask_float` / `ask_location` share the same re-prompt
pattern, the two diagnostic panels share the same plotting recipe), and
the control flow mirrors the structure of the problem (`while` for loops
that depend on user state, `for` for fixed iteration, `if`/`elif` for
menu dispatch).

---

## Sample interaction

```
============================================================
 Home Price Prediction — Linear vs Ridge Regression
============================================================
Loaded 58 houses priced above $500,000.
Features: ['HouseSize', 'Bedrooms', 'Location'].  Target: HousePrice.

Ridge regularisation alpha (default 1.0):

Model                     R²         MAE ($)              MSE ($²)
------------------------------------------------------------------
Linear (test)         0.8214         215,430      78,442,103,210
Ridge  (test)         0.8208         215,902      78,711,520,003

What would you like to do?
  1. Show diagnostic plots
  2. Predict a house price
  3. Re-print metrics
  4. Quit
Choice: 2

Enter the house features to get a price estimate.
Type 'back' at the size prompt to return to the main menu.

House size in sq ft (or 'back'): 1800
Number of bedrooms (default 3): 4
Location (Downtown / Suburb / Rural): Downtown
  Predicted price for a 1800 sq-ft, 4-bedroom Downtown house:
    Linear   $3,072,000
    Ridge    $3,068,000
```

---

## Robustness

The program is defensive about every external interaction:

- **Bad numeric input** (e.g. `"abc"` for alpha or bedrooms) is
  re-prompted, not silently defaulted.
- **Non-positive hyperparameters** (alpha ≤ 0, bedrooms ≤ 0) are rejected.
- **Unknown location** (`"forest"`) is re-prompted with the valid set.
- **Out-of-menu choices** print a helpful message and re-show the menu.
- **`Ctrl-C` / `Ctrl-D`** print `"Interrupted. Goodbye."` instead of a
  traceback.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | All code: data prep, models, evaluation, plotting, menu |
| `home_dataset.csv` | Input data — `HouseSize`, `Bedrooms`, `Location`, `HousePrice` |
| `requirements.txt` | Python dependencies |
| `training_log.txt` | Auto-appended per-run log (alpha, R²/MAE/MSE per model) |

---

## Sources

Referenced for ideas only — code in this repo is original:

- scikit-learn `Ridge` API:
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- scikit-learn `ColumnTransformer` / `Pipeline` guide:
  https://scikit-learn.org/stable/modules/compose.html
- W3Schools scikit-learn linear regression tutorial:
  https://www.w3schools.com/python/python_ml_linear_regression.asp
