import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 1. Load the "notebook" of 8,760 hourly records
df = pd.read_csv("data.csv")

# 2. Separate the inputs (situation + settings) from the answer (PUE)
X = df[["it_kw", "wet_bulb", "chw", "fan"]]
y = df["pue"]

# 3. Split by time: first 70% to learn from, last 30% as the "exam"
split = int(len(df) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Scale inputs so they're all on a similar size
#    (it_kw is in the thousands, fan is between 0 and 1)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # learn the scaling from training data only
X_test_s = scaler.transform(X_test)         # apply the same scaling to the exam data

# 5. Fit the straight-line model and test it on the exam data
model = LinearRegression()
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

# 6. Score it: MAE = the average size of its mistakes, in PUE units
print("Baseline model MAE:", mean_absolute_error(y_test, pred))

# For comparison: the score if it just guessed the average every time
guess = np.full(len(y_test), y_train.mean())
print("Just-guess-the-average MAE:", mean_absolute_error(y_test, guess))