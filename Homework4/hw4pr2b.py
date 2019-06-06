"""
Aom Pongpiriyakarn
apongpiriyakarn@hmc.edu
June 5, 2019
MATH189R Big Data Analytics Homework 4 Problem 2b
"""

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time

def NLL(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix
			4) reg, the parameter for regularization

		This function calculates and returns negative log likelihood for
		softmax regression.
	"""
	# Find the negative log likelihood for softmax regression
	mu = X @ W 
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis = 1).reshape(-1, 1)
	groundTruth = y * np.log(prob)
	NLL = -groundTruth.sum(axis = 1).sum() + reg * np.diag(W.T @ W).sum()
	return NLL

def grad_softmax(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix
			4) reg, the parameter for regularization

		This function calculates and returns the gradient of W for softmax
		regression.
	"""
	# Find the gradient of softmax regression with respect to W
	mu = X @ W
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	grad = X.T @ (prob - y) + reg * W
	return grad

def predict(X, W):
	"""	This function takes in two arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) W, a weight matrix

		This function returns the predicted labels y_pred with
		dimension m x 1
	"""
	# Obtain the array of predicted label y_pred using X, and Weight given
	mu = X @ W
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	y_pred = np.argmax(prob, axis=1).reshape(-1, 1)
	return y_pred

def get_accuracy(y_pred, y):
	"""	This function takes in two arguments:
			1) y_pred, the predicted label of data with dimension m x 1
			2) y, the true label of data with dimension m x 1

		This function calculates and returns the accuracy of the prediction
	"""
	diff = (y_pred == y).astype(int)
	accu = 1. * diff.sum() / len(y)
	return accu

def grad_descent(X, y, reg=0.0, lr=1e-5, eps=1e-6, max_iter=500, print_freq=20):
	"""	This function takes in seven arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of data with dimension m x 1
			3) reg, the parameter for regularization
			4) lr, the learning rate
			5) eps, the threshold of the norm for the gradients
			6) max_iter, the maximum number of iterations
			7) print_freq, the frequency of printing the report

		This function returns W, the optimal weight by gradient descent,
		and nll_list, the corresponding learning objectives.
	"""
	# get the shape of the data, and initialize nll_list
	m, n = X.shape
	k = y.shape[1]
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, k))
	W_grad = np.ones((n, k))

	print('\n==> Running gradient descent...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	# Run gradient descent algorithms
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# calculate negative log likelihood and append value in nll_list
		nll = NLL(X, y, W, reg=reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)
		# Calculate gradient for W and update it
		W_grad = grad_softmax(X, y, W, reg=reg)
		W -= lr * W_grad
		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1

	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} seconds'.format(\
			t=t_end - t_start))

	return W, nll_list

def accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list):
	"""	This function takes in five arguments:
			1) X_train, the training data with dimension m x (n + 1)
			2) y_train_OH, the label of training data with dimension m x 1
			3) X_test, the validation data with dimension m x (n + 1)
			4) y_test, the label of validation data with dimension m x 1
			5) lambda_list, a list of different regularization paramters that
							we want to test

		This function generates a plot of accuracy of prediction vs lambda and
		returns the regularization parameter that maximizes the accuracy,
		reg_opt.
	"""
	# Initialize the list of accuracy
	accu_list = []

	# Find corresponding accuracy values for each parameter
	for reg in lambda_list:
		# Run gradient descent with each parameter to obtain the optimal weight
		W, nll_list = grad_descent(X_train, y_train_OH, reg=reg, lr=2e-5, print_freq=50)
		# Predicted the label using the weights
		y_pred = predict(X_test, W)
		# Calculate accuracy
		accuracy = get_accuracy(y_pred, y_test)
		accu_list.append(accuracy)
		print('-- Accuracy is {:2.4f} for lambda = {:2.2f}'.format(accuracy, reg))

	# Plot accuracy vs lambda
	print('==> Printing accuracy vs lambda...')
	plt.style.use('ggplot')
	plt.plot(lambda_list, accu_list)
	plt.title('Accuracy versus Lambda in Softmax Regression')
	plt.xlabel('Lambda')
	plt.ylabel('Accuracy')
	plt.savefig('hw4pr2b_lva.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	# Find the optimal lambda that maximizes the accuracy
	opt_lambda_index = np.argmax(accu_list)
	reg_opt = lambda_list[opt_lambda_index]
	return reg_opt

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	df_train = data.df_train
	df_test = data.df_test

	X_train = data.X_train
	y_train = data.y_train
	X_test = data.X_test
	y_test = data.y_test

	# stacking an array of ones
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# one hot encoder
	enc = OneHotEncoder()
	y_train_OH = enc.fit_transform(y_train.copy()).astype(int).toarray()
	y_test_OH = enc.fit_transform(y_test.copy()).astype(int).toarray()

	# =============STEP 1: Accuracy versus lambda=================
	print('\n\n==> Step 1: Finding optimal regularization parameter...')

	lambda_list = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
	reg_opt = accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list)

	print('\n-- Optimal regularization parameter is {:2.2f}'.format(reg_opt))

	# =============STEP 2: Convergence plot=================
	# run gradient descent to get the nll_list
	W_gd, nll_list_gd = grad_descent(X_train, y_train_OH, reg=reg_opt,\
	 	max_iter=1500, lr=2e-5, print_freq=100)

	print('\n==> Step 2: Plotting convergence plot...')

	# set up style for the plot
	plt.style.use('ggplot')

	# generate the convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	# add legend, title, etc and save the figure
	plt.title('Convergence Plot on Softmax Regression with $\lambda = {:2.2f}$'.format(reg_opt))
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2b_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')
