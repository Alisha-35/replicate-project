# Replicating Google's data center cooling optimizer

A simulated data center, a neural network that predicts PUE, and an optimizer that searches for better settings. All savings are simulated, not measured in a real plant.

## How to run
1. py datacenter.py   (simulates one year of hourly data)
2. py baseline.py     (linear regression baseline)
3. py network.py      (trains the neural network)
4. py optimizer.py    (finds better settings, creates results.csv)

Requires Python with pandas, numpy, torch, scikit-learn and matplotlib.
