"""
Aom Pongpiriyakarn
apongpiriyakarn@hmc.edu
May 22, 2019
CS189r Big Data Analytics Homework 2 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def linreg(X, y, reg=0.0):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) reg, the parameter for regularization

		This function calculates and returns the optimal weight matrix, W_opt.
	"""
	# solve for W_opt
	dim_col = np.eye(X.shape[1])
	dim_col[0, 0] = 0
	W_opt = np.linalg.solve(X.T @ X + reg * dim_col, X.T @ y)
	return W_opt

def predict(W, X):
	"""	This function takes in two arguments:
			1) W, a weight matrix with bias
			2) X, the data with dimension m x (n + 1)

		This function calculates and returns the predicted label, y_pred.
	"""
	return X * W

def find_RMSE(W, X, y):
	"""	This function takes in three arguments:
			1) W, a weight matrix with bias
			2) X, the data with dimension m x (n + 1)
			3) y, the label of the data with dimension m x 1

		This function calculates and returns the root mean-squared error, RMSE
	"""
	# Solve for the root mean-squared error, RMSE
	diff_y = y - predict(W, X)
	dim_row = X.shape[0]
	mse = np.linalg.norm(diff_y, 2) ** 2 / dim_row
	rmse = np.sqrt(mse)
	return rmse

def RMSE_vs_lambda(X_train, y_train, X_val, y_val):
	"""	This function takes in four arguments:
			1) X_train, the training data with dimension m x (n + 1)
			2) y_train, the label of training data with dimension m x 1
			3) X_val, the validation data with dimension m x (n + 1)
			4) y_val, the label of validation data with dimension m x 1

		This function generates a plot of RMSE vs lambda and returns the
		regularization parameter that minimizes RMSE, reg_opt.
	"""
	# Generate a list of RMSE, RESE_list
	RMSE_list = []
	reg_list = []
	W_list = []
	reg_list = np.random.uniform(0.0, 150.0, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = r) for r in reg_list]
	for i in range(len(reg_list)):
		W_opt = W_list[i]
		RMSE_list.append(find_RMSE(W_opt, X_val, y_val))

	# Set up plot style
	plt.style.use('ggplot')

	# Plot RMSE vs lambda
	RMSE_vs_lambda_plot, = plt.plot(reg_list, RMSE_list)
	plt.setp(RMSE_vs_lambda_plot, color='red')
	plt.title('RMSE vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('RMSE')
	plt.savefig('RMSE_vs_lambda.png', format='png')
	plt.close()
	print('==> Plotting completed.')

	# Find reg_opt, the regularization value that minimizes RMSE
	reg_opt = reg_list[np.argmin(RMSE_list)]
	return reg_opt

def norm_vs_lambda(X_train, y_train, X_val, y_val):
	"""	This function takes in four arguments:
			1) X_train, the training data with dimension m x (n + 1)
			2) y_train, the label of training data with dimension m x 1
			3) X_val, the validation data with dimension m x (n + 1)
			4) y_val, the label of validation data with dimension m x 1

		This function generates a plot of norm of the weights vs lambda.
	"""
	# Generate a list of norm, norm_list
	reg_list = []
	W_list = []
	norm_list = []
	reg_list = np.random.uniform(0.0, 150.0, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = r) for r in reg_list]

	# Calculate the norm of each weight
	norm_list = [np.linalg.norm(W, 2) for W in W_list]

	# Set up plot style
	plt.style.use('ggplot')

	# Plot norm vs lambda
	norm_vs_lambda_plot, = plt.plot(reg_list, norm_list)
	plt.setp(norm_vs_lambda_plot, color='blue')
	plt.title('norm vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('norm')
	plt.savefig('norm_vs_lambda.png', format='png')
	plt.close()
	print('==> Plotting completed.')

def linreg_no_bias(X, y, reg = 0.0):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) reg, the parameter for regularization

		This function calculates and returns the optimal weight matrix, W_opt
		and bias, b_opt seperately
	"""
	t_start = time.time()

	# Find the numerical solution in part d
	# Solve for W_opt, and b_opt
	dim_row = X.shape[0]
	mat_I = np.eye(dim_row)
	mat_ones = np.ones(dim_row) 
	eq = X.T @ (mat_I - mat_ones / dim_row)
	mat_I = np.eye(eq.shape[0])
	W_opt = np.linalg.solve(eq @ X + reg * mat_I, eq @ y)
	b_opt = sum((y - X @ W_opt)) / dim_row

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} seconds'.format(\
			t=t_end - t_start))

	return b_opt, W_opt

def grad_descent(X_train, y_train, X_val, y_val, reg=0.0, lr_W=2.5e-12, \
 		lr_b=0.2, max_iter=150, eps=1e-6, print_freq=25):
	"""	This function takes in ten arguments:
			1) X_train, the training data with dimension m x (n + 1)
			2) y_train, the label of training data with dimension m x 1
			3) X_val, the validation data with dimension m x (n + 1)
			4) y_val, the label of validation data with dimension m x 1
			5) reg, the parameter for regularization
			6) lr_W, the learning rate for weights
			7) lr_b, the learning rate for bias
			8) max_iter, the maximum number of iterations
			9) eps, the threshold of the norm for the gradients
			10) print_freq, the frequency of printing the report

		This function returns W, the optimal weight, and b, the bias by
		gradient descent.
	"""
	m_train, n = X_train.shape
	m_val = X_val.shape[0]

	# initialize the weights and bias and their corresponding gradients
	W = np.zeros((n, 1))	# weights
	W_grad = np.ones_like(W)# gradients of W
	b = 0. 					# bias
	b_grad = 1.				# gradients of b

	print('==> Running gradient descent...')

	# Run the gradient descent algorithm followed steps below
	#	1) Calculate the training RMSE and validation RMSE at each iteration,
	#      and append these values to obj_train and obj_val respectively
	#	2) Calculate the gradient for W and b as W_grad and b_grad
	#	3) Upgrade W and b
	#	4) Keep iterating while the number of iterations is less than the
	#	   maximum and the gradient is larger than the threshold
	obj_train = []
	obj_val = []
	iter_num = 0

	t_start = time.time()

	# start iteration for gradient descent
	while np.linalg.norm(W_grad) > eps and np.linalg.norm(b_grad) > eps \
		and iter_num < max_iter:

		# calculate norms
		rmse_train = np.sqrt(np.linalg.norm((X_train @ W).reshape((-1,1)) + b - y_train)\
			** 2/ m_train)
		obj_train.append(rmse_train)
		val_rmse = np.sqrt(np.linalg.norm((X_val @ W).reshape((-1, 1)) + b - y_val)\
			** 2 / m_val)
		obj_val.append(val_rmse)

		# calculate gradient
		mat_I = np.eye(n)
		W_grad = ((X_train.T @ X_train + reg * mat_I) @ W + X_train.T @ (b - y_train)) / m_train
		b_grad = (sum(X_train @ W) - sum(y_train) + b * m_train) / m_train

		# update weights and bias
		W -= lr_W * W_grad
		b -= lr_b * b_grad

		# print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration{} - training rmse {: 4.4f} - gradient norm {: 4.4E}'.format(\
				iter_num + 1, rmse_train, np.linalg.norm(W_grad)))

		# goes to next iteration
		iter_num += 1

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} seconds'.format(\
			t=t_end - t_start))

	# Set up plot style
	plt.style.use('ggplot')

	# generate convergence plot
	train_rmse_plot, = plt.plot(range(iter_num), obj_train)
	plt.setp(train_rmse_plot, color='red')
	val_rmse_plot, = plt.plot(range(iter_num), obj_val)
	plt.setp(val_rmse_plot, color='green')
	plt.legend((train_rmse_plot, val_rmse_plot), \
		('Training RMSE', 'Validation RMSE'), loc='best')
	plt.title('RMSE vs iteration')
	plt.xlabel('iteration')
	plt.ylabel('RMSE')
	plt.savefig('convergence.png', format='png')
	plt.close()
	print('==> Plotting completed.')

	return b, W

if __name__ == '__main__':

	print('==> Loading data...')

	# Read data
	df = pd.read_csv('https://math189sp19.github.io/data/online_news_popularity.csv', \
		sep=', ', engine='python')

	# split the data frame by type: training, validation, and test
	train_pct = 2.0 / 3
	val_pct = 5.0 / 6

	df['type'] = ''
	df.loc[:int(train_pct * len(df)), 'type'] = 'train'
	df.loc[int(train_pct * len(df)) : int(val_pct * len(df)), 'type'] = 'val'
	df.loc[int(val_pct * len(df)):, 'type'] = 'test'

	# extracting columns into training, validation, and test data
	X_train = np.array(df[df.type == 'train'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_train = np.array(np.log(df[df.type == 'train'].shares)).reshape((-1, 1))

	X_val = np.array(df[df.type == 'val'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_val = np.array(np.log(df[df.type == 'val'].shares)).reshape((-1, 1))

	X_test = np.array(df[df.type == 'test'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_test = np.array(np.log(df[df.type == 'test'].shares)).reshape((-1, 1))

	# Stack a column of ones to the feature data, X_train, X_val and X_test
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_val = np.hstack((np.ones_like(y_val), X_val))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# Convert data to matrix
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_val = np.matrix(X_val)
	y_val = np.matrix(y_val)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)

	print('==> Step 1: RMSE vs lambda...')

	# Find the optimal regularization parameter
	reg_opt = RMSE_vs_lambda(X_train, y_train, X_val, y_val)
	print('==> The optimal regularization parameter is {reg: 4.4f}.'.format(\
		reg=reg_opt))

	# Find the optimal weights and bias for future use in step 3
	W_with_b_1 = linreg(X_train, y_train, reg=reg_opt)
	b_opt_1 = W_with_b_1[0]
	W_opt_1 = W_with_b_1[1: ]

	# Report the RMSE with the found optimal weights on validation set
	val_RMSE = find_RMSE(W_with_b_1, X_val, y_val)
	print('==> The RMSE on the validation set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(\
		RMSE=val_RMSE))

	# Report the RMSE with the found optimal weights on test set
	test_RMSE = find_RMSE(W_with_b_1, X_test, y_test)
	print('==> The RMSE on the test set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(\
		RMSE=test_RMSE))

	print('\n==> Step 2: Norm vs lambda...')
	norm_vs_lambda(X_train, y_train, X_val, y_val)

	# From here on, we will strip the columns of ones for all data
	X_train = X_train[:, 1:]
	X_val = X_val[:, 1:]
	X_test = X_test[:, 1:]

	# Compare the result with the one from step 1
	# The difference in norm should be a small scalar (i.e, 1e-10)
	print('\n==> Step 3: Linear regression without bias...')
	b_opt_2, W_opt_2 = linreg_no_bias(X_train, y_train, reg=reg_opt)

	# difference in bias
	diff_bias = np.linalg.norm(b_opt_2 - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

	# difference in weights
	diff_W = np.linalg.norm(W_opt_2 -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))

	print('\n==> Step 4: Gradient descent')
	b_gd, W_gd = grad_descent(X_train, y_train, X_val, y_val, reg=reg_opt)

	# Compare the result from the one from step 1
	# Difference in bias
	diff_bias = np.linalg.norm(b_gd - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

	# difference in weights
	diff_W = np.linalg.norm(W_gd -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))
