import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------- Custom KMeans --------------------
class KMeansClustering:
    def __init__(self, k=3):
        # Number of clusters to find
        self.k = k
        # Centroids will be initialized during fit()
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        # Compute Euclidean distance between a data point and all centroids
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=300):
        # Randomly initialize centroids within the data range
        self.centroids = np.random.uniform(
            np.amin(X, axis=0), 
            np.amax(X, axis=0), 
            size=(self.k, X.shape[1])
        )

        for _ in range(max_iterations):
            y = []  # Cluster assignment list

            # Step 1: Assign each point to the closest centroid
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)  # Closest cluster
                y.append(cluster_num)

            y = np.array(y)  # Convert to NumPy array for indexing

            # Step 2: Group points by cluster
            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))  # Get all indices for cluster i

            # Step 3: Recompute centroids as the mean of assigned points
            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    # If no points assigned to this cluster, keep old centroid
                    cluster_centers.append(self.centroids[i])
                else:
                    # Compute new centroid (mean of all points in the cluster)
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            # Step 4: Check for convergence (no significant change in centroids)
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break  # Centroids are stable, stop iterating
            else:
                self.centroids = np.array(cluster_centers)  # Update centroids

        return y  # Return final cluster assignments
# --------------------------------------------------------------------------

# Load Image
img = cv2.imread("/home/kailinazx/Documents/Computer Vision/Kmeans-Clustering-Segmentation/test-img/74.jpg")  # Use your image file path
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300))

# Preprocess
pixels = img.reshape((-1, 3)).astype(np.float32)

# Apply Custom KMeans
kmeans = KMeansClustering(k=5)  # You may change k based on your image
labels = kmeans.fit(pixels)
labels = labels.reshape((img.shape[:2]))

# Show each cluster
for i in range(kmeans.k):
    mask = (labels == i)
    cluster_img = np.zeros_like(img)
    cluster_img[mask] = img[mask]

    plt.figure()
    plt.imshow(cluster_img)
    plt.title(f'Cluster {i}')
    plt.axis('off')

plt.show()
