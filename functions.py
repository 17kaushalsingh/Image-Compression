# import numpy as np
# import cv2
# from sklearn.cluster import KMeans

# def load_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("The provided path is not an image file or the file does not exist.")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image

# def compress_image(image, n_colors):
#     data = image.reshape((-1, 3))
#     kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(data)
#     new_colors = kmeans.cluster_centers_[kmeans.labels_]
#     compressed_image = new_colors.reshape(image.shape)
#     compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
#     return compressed_image

# def save_image(image, path):
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(path, image)

# def kmeans_image_compression(image_path, n_colors=16, output_path="compressed_image.png"):
#     image = load_image(image_path)
#     compressed_image = compress_image(image, n_colors)
#     save_image(compressed_image, output_path)
#     return output_path

'''
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def load_image(image):
    image = np.array(image)
    return image

def compress_image(image, n_colors):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = new_colors.reshape(image.shape)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed_image)

def save_image(image, path):
    image.save(path)

def kmeans_image_compression(image_path, n_colors=16, output_path="compressed_image.png"):
    image = load_image(image_path)
    compressed_image = compress_image(image, n_colors)
    save_image(compressed_image, output_path)
    return output_path
'''
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def load_image(image):
    image = np.array(image)
    return image

def compress_image(image, n_colors):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = new_colors.reshape(image.shape)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed_image)

def save_image(image, path):
    image.save(path)

def kmeans_image_compression(image_path, n_colors=16, output_path="compressed_image.png"):
    image = load_image(image_path)
    compressed_image = compress_image(image, n_colors)
    save_image(compressed_image, output_path)
    return output_path
