"""
Aom Pongpiriyakarn 
apongpiriyakarn@hmc.edu
May 23, 2019
CS189r: Big Data Analytics Homework 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

	# part c: Plot data and the optimal linear fit

	# load the four data points of tihs problem
	X = np.array([0, 2, 3, 4])
	y = np.array([1, 3, 6, 8])

	# plot four data points on the plot
	plt.style.use('ggplot')
	plt.plot(X, y, 'ro')

	# replace the m_opt and b_opt with the solution you obtained from
	# part (a), note that y = mx + b
	m_opt = 62./35  # theta_1
	b_opt = 18./35  # theta_0

	# generate 100 points along the line of optimal linear fit.
	X_space = []
	y_space = []
	X_space = np.linspace(-1, 5, 100).reshape(-1, 1)
	y_space = (m_opt * X_space + b_opt).reshape(-1, 1)

	# plot the optimal learn fit you obtained and save it to your current folder
	plt.plot(X_space, y_space)
	plt.savefig('hw1pr2c.png', format='png')
	plt.close()


	# part d: Optimal linear fit with random data points

	# variables to start with
	mu, sigma, sampleSize = 0, 1, 100

	# Generate white Gaussian noise
	noise = []
	noise = np.random.normal(mu, sigma, sampleSize).reshape(-1, 1)

	# generate y-coordinate of the 100 points with noise
	y_space_rand = np.zeros(len(X_space))
	y_space_rand = m_opt * X_space + b_opt + noise

	# calculate the new parameters for optimal linear fit using the
	# 100 new points generated above
	X_space_stacked = X_space	# need to be replaced following hint 1 and 2
	W_opt = None
	X_space_stacked = np.hstack((np.ones_like(y_space), X_space))
	W_opt = np.linalg.solve(X_space_stacked.T @ X_space_stacked,
		X_space_stacked.T @ y_space_rand)

	# get the new m, and new b from W_opt obtained above
	b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)

	# Generate the y-coordinate of 100 points with the new parameters
	# obtained
	y_pred_rand = []
	y_pred_rand = np.array([m_rand_opt * x + b_rand_opt for x in X_space])\
				  .reshape(-1,1)

	# generate plot
	# plot original data points and line
	plt.plot(X, y, 'ro')
	orig_plot, = plt.plot(X_space, y_space, 'r')

	# plot the generated 100 points with white gaussian noise and the new line
	plt.plot(X_space, y_space_rand, 'bo')
	rand_plot, = plt.plot(X_space, y_pred_rand, 'b')

	# set up legend and save the plot to the current folder
	plt.legend((orig_plot, rand_plot), \
		('original fit', 'fit with noise'), loc = 'best')
	plt.savefig('hw1pr2d.png', format='png')
	plt.close()
