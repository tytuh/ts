import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

st.title("Logistic Regression with Gradient Descent")

# Generate toy dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, n_samples=1000, random_state=42)

# Initialize logistic regression model
model = LogisticRegression(fit_intercept=True, solver='sag', max_iter=1000)

# Add interactive inputs to the app
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=1.0, step=0.001, value=0.1)
num_iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=1000, step=1, value=100)

# Fit model using gradient descent
for i in range(num_iterations):
    model.coef_ += learning_rate * model.coef_
    model.intercept_ += learning_rate * model.intercept_
    model.coef_ = model.coef_ - learning_rate * model.coef_
    model.intercept_ = model.intercept_ - learning_rate * model.intercept_
    model.fit(X, y)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('X')
plt.ylabel('Y')

st.pyplot()