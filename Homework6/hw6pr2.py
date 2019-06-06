"""
Aom Pongpiriyakarn
apongpiriyakarn@hmc.edu
June 5, 2019
MATH189R Big Data Analytics Homework 6 Problem 2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import urllib

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading image data...')
	img = ndimage.imread(urllib.request.urlopen('http://i.imgur.com/X017qGH.jpg'), flatten=True)

	# Shuffle the image
	shuffle_img = img.copy().flatten()
	np.random.shuffle(shuffle_img)

	# reshape the shuffled image
	shuffle_img = shuffle_img.reshape(img.shape)

	# =============STEP 1: RUNNING SVD ON IMAGES=================
	print('==> Running SVD on images...')

	# SVD on img and shuffle_img
	U, S, V = np.linalg.svd(img)
	U_s, S_s, V_s = np.linalg.svd(shuffle_img)

	# =============STEP 2: SINGULAR VALUE DROPOFF=================
	print('==> Singular value dropoff plot...')
	k = 100
	plt.style.use('ggplot')

	# Generate singular value dropoff plot
	orig_S_plot, = plt.plot(S[:k], 'b')
	shuf_S_plot, = plt.plot(S_s[:k], 'r')

	plt.legend((orig_S_plot, shuf_S_plot), \
		('original', 'shuffled'), loc = 'best')
	plt.title('Singular Value Dropoff for Clown Image')
	plt.ylabel('singular values')
	plt.savefig('dropoff.png', format='png')
	plt.close()

	# =============STEP 3: RECONSTRUCTION=================
	print('==> Reconstruction with different ranks...')
	rank_list = [2, 10, 20]
	plt.subplot(2, 2, 1)
	plt.imshow(img, cmap='Greys_r')
	plt.axis('off')
	plt.title('Original Image')

	# Generate reconstruction images for each of the rank values
	for index in range(len(rank_list)):
		k = rank_list[index]
		plt.subplot(2, 2, 2 + index)
		img_recons = U[:, :k] @ np.diag(S)[:k, :k] @ V[:k, :]
		plt.imshow(img_recons, cmap='Greys_r')
		plt.title('Rank {} Approximation'.format(k))
		plt.axis('off')

	plt.tight_layout()
	plt.savefig('reconstruction.png', format='png')
	plt.close()
