import numpy as np

# Assuming 400 features (pixels) for simplicity
num_features = 400

# Initialize weights with small random values from a Gaussian distribution
# weights = np.random.normal(loc=0, scale=0.01, size=num_features)
weights = np.random.rand(400)
print(weights)

a = [1,2]
b = [3,4]

print(np.dot(a,b))

colors = ['⬜️', '🟥', '🟨', '🟦', '🟩']
encoded_array = np.zeros((20, 20, len(colors)), dtype=int)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(encoded_array.flatten())
# Initialize bias
# bias = np.random.rand(1)
# print(bias)