import numpy as np, pandas as pd
rng = np.random.default_rng(42)

N = 365 * 24                                   # 1 year, one row per hour
hour = np.arange(N) % 24
day = np.arange(N) / 24

# Wet-bulb temp: warm summers, cool winters, a bit warmer in the afternoon
wet_bulb = 15 + 8*np.sin(2*np.pi*(day-100)/365) + 2*np.sin(2*np.pi*(hour-9)/24) + rng.normal(0, 1, N)

# Server load: busier in the daytime, plus noise
it_kw = 6000 + 800*np.sin(2*np.pi*(hour-9)/24) + rng.normal(0, 200, N)

# Settings the "operator" used (random, so the data has variety)
chw = np.clip(10 + rng.normal(0, 1.5, N), 7, 14)          # cooling water temp (C)
fan = np.clip(rng.uniform(0.3, 1.0, N), 0.3, 1.0)         # fan speed (0.3 to 1.0)

def pue_model(it_kw, wet_bulb, chw, fan):
    cond_temp = wet_bulb + (8 - 5*fan)                    # faster fan -> water closer to wet-bulb
    lift = cond_temp - chw                                # temperature gap the chiller works against
    cop = np.maximum(2.0, 9.0 - 0.18*lift)                # chiller efficiency
    chiller = it_kw / cop
    tower_fan = 0.03 * it_kw * fan**3                     # fan power (cube of speed)
    pump = 0.02 * it_kw
    other = 0.05 * it_kw
    server_fans = 0.01 * it_kw * (1 + 0.05*(chw-7)**2)    # warmer water makes server fans work harder
    return (it_kw + chiller + tower_fan + pump + other + server_fans) / it_kw

pue = pue_model(it_kw, wet_bulb, chw, fan) + rng.normal(0, 0.003, N)
pd.DataFrame({"it_kw": it_kw, "wet_bulb": wet_bulb, "chw": chw, "fan": fan, "pue": pue}).to_csv("data.csv", index=False)
print(pd.DataFrame({"pue": pue}).describe())