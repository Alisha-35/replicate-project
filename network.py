import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

torch.manual_seed(0)   # makes the random starting weights the same every run

# 1. Load the data and split it the same way as the baseline
df = pd.read_csv("data.csv")
X = df[["it_kw", "wet_bulb", "chw", "fan"]].values
y = df[["pue"]].values

split = int(len(df) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Scale the inputs AND the answer so everything is a similar size
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

# 3. Convert to the format torch uses (tensors)
Xtr = torch.tensor(x_scaler.transform(X_train), dtype=torch.float32)
ytr = torch.tensor(y_scaler.transform(y_train), dtype=torch.float32)
Xte = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32)

# 4. Build the network: 4 inputs -> 5 layers of 50 nodes -> 1 output (PUE)
#    Tanh is the "bend" that lets each layer learn curves, not just straight lines
model = nn.Sequential(
    nn.Linear(4, 50),  nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 1),
)

loss_fn = nn.MSELoss()                                   # how we measure mistakes during training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # the thing that nudges the weights

def test_mae():
    # Ask the network to predict the exam data and score it in real PUE units
    model.eval()
    with torch.no_grad():
        p = model(Xte).numpy()
    p = y_scaler.inverse_transform(p)
    return mean_absolute_error(y_test, p)

# 5. Training: show the network the data in small batches, many times over
for epoch in range(1, 61):
    model.train()
    order = torch.randperm(len(Xtr))             # shuffle the rows
    for i in range(0, len(Xtr), 64):             # 64 rows at a time
        idx = order[i:i + 64]
        pred = model(Xtr[idx])                   # network guesses
        loss = loss_fn(pred, ytr[idx])           # measure how wrong it was
        optimizer.zero_grad()
        loss.backward()                          # work out which weights caused the error
        optimizer.step()                         # nudge those weights
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: test MAE = {test_mae():.5f}")

# 6. Final score, plus save everything for the optimizer step
print("\nNeural network MAE:", test_mae())
print("Baseline MAE was:   0.00431")

torch.save(model.state_dict(), "model.pt")
with open("scalers.pkl", "wb") as f:
    pickle.dump((x_scaler, y_scaler), f)
print("Saved model.pt and scalers.pkl")
