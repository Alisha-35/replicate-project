import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# 1. Rebuild the same network shape and load what it learned
model = nn.Sequential(
    nn.Linear(4, 50),  nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 1),
)
model.load_state_dict(torch.load("model.pt"))
model.eval()
with open("scalers.pkl", "rb") as f:
    x_scaler, y_scaler = pickle.load(f)

# 2. The simulator's real physics (our "ground truth" for checking)
def pue_model(it_kw, wet_bulb, chw, fan):
    cond_temp = wet_bulb + (8 - 5*fan)
    lift = cond_temp - chw
    cop = np.maximum(2.0, 9.0 - 0.18*lift)
    chiller = it_kw / cop
    tower_fan = 0.03 * it_kw * fan**3
    pump = 0.02 * it_kw
    other = 0.05 * it_kw
    server_fans = 0.01 * it_kw * (1 + 0.05*(chw-7)**2)
    return (it_kw + chiller + tower_fan + pump + other + server_fans) / it_kw

# 3. Take the "exam" period (last 30% of the year)
df = pd.read_csv("data.csv")
test = df[int(len(df) * 0.7):].reset_index(drop=True)

# 4. The menu of settings to try: 8 water temps x 8 fan speeds = 64 combos
#    (stays inside the ranges the network saw during training)
chw_options = np.linspace(7, 14, 8)
fan_options = np.linspace(0.3, 1.0, 8)
grid = np.array([(c, f) for c in chw_options for f in fan_options])

# 5. For every hour: try all 64 combos, let the network predict PUE, pick the lowest
best_chw, best_fan = [], []
for _, row in test.iterrows():
    inputs = np.column_stack([
        np.full(len(grid), row["it_kw"]),
        np.full(len(grid), row["wet_bulb"]),
        grid[:, 0],
        grid[:, 1],
    ])
    x = torch.tensor(x_scaler.transform(inputs), dtype=torch.float32)
    with torch.no_grad():
        preds = y_scaler.inverse_transform(model(x).numpy())
    best = grid[np.argmin(preds)]
    best_chw.append(best[0])
    best_fan.append(best[1])

test["best_chw"] = best_chw
test["best_fan"] = best_fan

# 6. Check both choices with the real physics (no noise)
test["pue_operator"] = pue_model(test["it_kw"], test["wet_bulb"], test["chw"], test["fan"])
test["pue_optimized"] = pue_model(test["it_kw"], test["wet_bulb"], test["best_chw"], test["best_fan"])

# 7. Results
op = test["pue_operator"].mean()
opt = test["pue_optimized"].mean()
print(f"Average PUE with the random operator settings: {op:.4f}")
print(f"Average PUE with the optimizer's settings:     {opt:.4f}")

# Compare the overhead (the part above 1.0), which is the real saving
saving = ((op - 1) - (opt - 1)) / (op - 1) * 100
print(f"Cooling and overhead energy reduced by:        {saving:.1f}%")

test.to_csv("results.csv", index=False)
print("Saved results.csv")