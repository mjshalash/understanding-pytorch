import numpy as np

# Data Generation
# Maintain rand seed so x can be reproduced and create 100 points for feature x
# labels a = 1 and b = 2
np.random.seed(42)
x = np.random.rand(100, 1)

y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffle the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Use first 80 for training
train_idx = idx[:80]

# Use remaining for testing
test_idx = idx[80:]

# Generate test and train sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

### Stopped at Step 2: Compute the Gradients ###
