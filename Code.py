
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Read data from CSV
df = pd.read_csv("data.csv")
X = df["X"].values.reshape(-1, 1)
y = df["y"].values

# Perform SVR
svr = SVR(kernel="linear")
svr.fit(X, y)
y_pred = svr.predict(X)

# Visualize the results
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Support Vector Regression")
plt.show()
