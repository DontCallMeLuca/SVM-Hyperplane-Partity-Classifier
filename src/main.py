import pandas as pd
import numpy as np
import random

def image_data(X, i):
	"""
	Takes X, extracts and reshapes the i-th observation for visualization.

	Parameters:
		X (np.ndarray): Represents the data matrix (X ∈ R^Nxp), where each row contains pixel data for an image.
		i (int): Represents the index of the one observation we want to visualize.

	Returns:
		np.ndarray: A 28x28 matrix with pixel values clipped to [0, 255] representing the darkness.
	"""
	reshaped_matrix = 255 - X[i].reshape(28, 28)
	return reshaped_matrix


def indicator(d, v):
	"""
	Converts class labels into binary labels based on membership in the specified classes, creating vector y

	Parameters:
		d (np.ndarray): Vector of class labels. d ∈ R^N
		v (list or np.ndarray): Classes belonging here are assigned value 1. v ∈ R^m

	Returns:
		np.ndarray: Binary vector y with classes with value 1 if in v, -1 otherwise.
	"""
	return np.where(np.isin(d, v), 1, -1)


def separate_wbar(wbar):
	"""
	Added this to separate the first element (bias) from the rest (weights) in a vector.

	Parameters:
		wbar (np.ndarray): Combined vector [w0, w1]. w0 ∈ R and w1 ∈ R^p

	Returns:
		tuple: w0 (float) and w1 (np.ndarray)
	"""
	w0 = wbar[0]
	w1 = wbar[1:]
	return w0, w1


def func_loss(wbar, mu, X, y):
	"""
	Computes soft-margin SVM loss function.

	Parameters:
		wbar (np.ndarray): Combined vector [w0, w1]. wbar ∈ R^p+1
		mu (float): Regularization parameter. mu ∈ R
		X (np.ndarray): Data matrix. X ∈ R^Nxp
		y (np.ndarray): Label vector. y ∈ R^N

	Returns:
		float: The loss function value.
	"""
	w0, w1 = separate_wbar(wbar)
	w_plus = np.array(w0 + np.dot(X, w1))
	loss_function = mu * np.dot(w1, w1) + np.maximum (0, 1 - (y * w_plus)).mean() 
	return loss_function


def grad_loss(wbar, mu, X, y):
	"""
	Computes subgradient of the SVM loss function.

	Parameters:
		wbar (np.ndarray): Combined vector [w0, w1]. wbar ∈ R^p+1
		mu (float): Regularization parameter. mu ∈ R
		X (np.ndarray): Data matrix. X ∈ R^Nxp
		y (np.ndarray): Label vector. y ∈ R^N

	Returns:
		np.ndarray: The subgradient vector of the loss function.
	"""
	w0, w1 = separate_wbar(wbar)
	p = w1.shape[0]
	summation = np.zeros(p + 1)

	for i in range(y.shape[0]):
		w_plus = w0 + np.dot(w1, X[i])
		if 1 - y[i] * w_plus >= 0:
			summation += (-y[i] * np.hstack((1, X[i])))
		else:
			summation += 0

	r = 2 * mu * np.hstack((0, w1)) + (1/y.shape[0]) * summation
	return r


def svm(alpha, epsilon, mu, X, y):
	"""
	Trains a soft-margin SVM on entire data set using subgradient descent.

	Parameters:
		alpha (float): Step size for gradient descent updates. alpha ∈ R
		epsilon (float): Convergence tolerance for the loss function. epsilon ∈ R
		mu (float): Regularization parameter. mu ∈ R
		X (np.ndarray): Data matrix. X ∈ R^Nxp
		y (np.ndarray): Label vector. y ∈ R^N

	Returns:
		np.ndarray: Matrix of iterates (p+1 x iterations).
	"""
	wbar_curr = np.zeros(X.shape[1] + 1)
	wbar_prev = np.zeros(X.shape[1] + 1)
	k = 0
	iterates = np.empty((X.shape[1] + 1, 0))

	while k <= 1 or np.abs(func_loss(wbar_curr, mu, X, y) \
		- func_loss(wbar_prev, mu, X, y)) > epsilon:
		wbar_next = wbar_curr - alpha * grad_loss(wbar_curr, mu, X, y)
		k += 1
		iterates = np.column_stack((iterates, wbar_next))
		wbar_prev = wbar_curr
		wbar_curr = wbar_next

	return iterates


def svm_accuracy(wbar, X, y):
	"""
	Computes the accuracy of the SVM.

	Parameters:
		wbar (np.ndarray): Combined vector [w0, w1]. wbar ∈ R^p+1
		X (np.ndarray): Data matrix. X ∈ R^Nxp
		y (np.ndarray): Labels vector. y ∈ R^N

	Returns:
		float: Accuracy of SVM, which is the fraction of correctly classified observations.
	"""
	w0, w1 = separate_wbar(wbar)
	accuracy_counter = np.mean((np.sign(w0 + X.dot(w1))) == y)
	return accuracy_counter


def train_test_split(X, y, beta):
	"""
	Separates the data into two subsets: training data and testing data.

	Parameters:
		X (np.ndarray): Data matrix. X ∈ R^Nxp
		y (np.ndarray): Labels vector. y ∈ R^N
		beta (float): Value between 0 and 1, representing the fraction of dataset to use for training. Beta ∈ R

	Returns:
		tuple: X_train, y_train, X_test, y_test
	"""
	N_train = int(beta * y.shape[0])
	N_test = y.shape[0] - int(beta * y.shape[0])
	train_indices = random.sample(range(y.shape[0]), N_train)
	test_indices = random.sample(range(y.shape[0]), N_test)
	X_train = X[train_indices, :]
	y_train = y[train_indices]
	X_test = X[test_indices, :]
	y_test = y[test_indices]
	return X_train, y_train, X_test, y_test


def accuracies(X, y):
	"""
	Computes in-sample and out-of-sample accuracies for the resulting model.

	Parameters:
		X (np.ndarray): Data matrix. X ∈ R^Nxp
		y (np.ndarray): Labels vector. y ∈ R^N

	Returns:
		tuple: Contains the in-sample and out-of-sample accuracies, in that order.
	"""
	X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
	trained_svm = svm(1e-5, 1e-2, 1e-1, X_train, y_train)
	accuracy_insample = svm_accuracy(trained_svm[:, -1], X_train, y_train)
	accuracy_outofsample = svm_accuracy(trained_svm[:, -1], X_test, y_test)
	return accuracy_insample, accuracy_outofsample


if __name__ == '__main__':
	df = pd.read_csv("./data/handwriting.csv")
	d = df.iloc[:, 0].values
	X = df.iloc[:, 1:].values
	y = indicator(d, [0, 1])
	accuracy_insample, accuracy_outofsample = accuracies(X, y)
	print(f"In-sample accuracy: {accuracy_insample:.4f}")
	print(f"Out-of-sample accuracy: {accuracy_outofsample:.4f}")