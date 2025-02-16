import numpy as np

# Load the two .npy files
obs1 = np.load("obs1.npy")
obs2 = np.load("obs2.npy")

# Compare the arrays
are_equal = np.array_equal(obs1, obs2)

if are_equal:
    print("The contents of obs1.npy and obs2.npy are identical.")
else:
    print("The contents of obs1.npy and obs2.npy are different.")
    
    # If they're different, you might want to see where they differ
    if obs1.shape == obs2.shape:
        differences = np.where(obs1 != obs2)
        print(f"Differences found at indices: {differences}")
        print(f"Values in obs1: {obs1[differences]}")
        print(f"Values in obs2: {obs2[differences]}")
    else:
        print(f"The arrays have different shapes: obs1 {obs1.shape}, obs2 {obs2.shape}")
