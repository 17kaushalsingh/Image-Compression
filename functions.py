'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg

def read_image():
	img = cv2.imread(path) 

	# Convert the image from BGR to RGB
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Scaling the image so that the values are in the range of 0 to 1
	img = img / 255.0


	return img

def initialize_means(img, clusters):
	# reshaping it or flattening it into a 2d matrix
	points = img.reshape((-1, img.shape[2]))
	m, n = points.shape

	# clusters is the number of clusters
	# or the number of colors that we choose.

	# means is the array of assumed means or centroids.
	means = np.zeros((clusters, n))

	# random initialization of means.
	for i in range(clusters):
		rand_indices = np.random.choice(m, size=10, replace=False)
		means[i] = np.mean(points[rand_indices], axis=0)

	return points, means

def distance(x1, y1, x2, y2):
	dist = np.square(x1 - x2) + np.square(y1 - y2)
	dist = np.sqrt(dist)
	return dist

def k_means(points, means, clusters):
	iterations = 10 # the number of iterations
	m, n = points.shape

	# these are the index values that correspond to the cluster to
	# which each pixel belongs to.
	index = np.zeros(m)

	# k-means algorithm.
	while iterations > 0:
		for j in range(m):
			# initialize minimum value to a large value
			min_dist = float('inf')
			temp = None

			for k in range(clusters):
				x1, y1 = points[j, 0], points[j, 1]
				x2, y2 = means[k, 0], means[k, 1]

				if distance(x1, y1, x2, y2) <= min_dist:
					min_dist = distance(x1, y1, x2, y2)
					temp = k
					index[j] = k

		for k in range(clusters):
			cluster_points = points[index == k]
			if len(cluster_points) > 0:
				means[k] = np.mean(cluster_points, axis=0)

		iterations -= 1

	return means, index

def compress_image(means, index, img):
	# recovering the compressed image by assigning each pixel to its corresponding centroid.
	centroid = np.array(means)
	recovered = centroid[index.astype(int), :]

	# getting back the 3d matrix (row, col, rgb(3))
	recovered = recovered.reshape(img.shape)

	# plotting the compressed image.

	# saving the compressed image.
	cv2.imwrite('compressed_' + str(clusters) + '_colors.png',recovered)
	return recovered # Fixed indentation

def plot_image(Orginal,Compressed):
    fig,ax = plt.subplots(1,2)
    Img1 = ax[0].imshow(Orginal)
    Img2 = ax[1] .imshow(Compressed)
    for i in range(2):
        title = ["Orginal","Compressed"]
        ax[i].set_title(title[i])

'''

# /mnt/data/functions.py

import numpy as np
import cv2
from sklearn.cluster import KMeans

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("The provided path is not an image file or the file does not exist.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def compress_image(image, n_colors):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = new_colors.reshape(image.shape)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    return compressed_image

def save_image(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def kmeans_image_compression(image_path, n_colors=16, output_path="compressed_image.png"):
    image = load_image(image_path)
    compressed_image = compress_image(image, n_colors)
    save_image(compressed_image, output_path)
    return output_path
