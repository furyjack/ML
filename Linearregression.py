from sklearn.linear_model import LinearRegression
import numpy as np
# Training data
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# error
print(np.mean((model.predict(X)-y)**2))

