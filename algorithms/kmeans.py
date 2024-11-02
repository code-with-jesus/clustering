import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Randomly initialize centroids
        np.random.seed(42)  # For reproducibility
        initial_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[initial_indices]
        
        for i in range(self.max_iter):
            # Assign each point to the nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Calculate new centroids
            new_centroids = self._calculate_centroids(X)
            
            # Check for convergence (if centroids do not change)
            if np.all(new_centroids == self.centroids):
                print(f"Converged after {i+1} iterations")
                break
            
            # Update centroids
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # Calculate distances between each point and each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Return the index of the closest centroid
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X):
        # Calculate the mean of points assigned to each centroid
        return np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def predict(self, X):
        # Predict the nearest cluster for each data point
        return self._assign_clusters(X)