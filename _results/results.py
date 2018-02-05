import numpy as np

yval = np.load("processed_data.npy")

new = np.ndarray((4,4))


print("Bi-directional confidence")
print(np.round(yval, 3))
print("")

for i in range(yval.shape[0]):
    adjust = yval[i][i]
    for j in range(yval.shape[1]):
        yval[i][j] = yval[i][j]/adjust

yval = np.round(yval, 3)

print("Bi-directional confidence regularized (pred/correctpred)")
print(yval)