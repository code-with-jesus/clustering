import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from algorithms.kmeans import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Main program
if __name__ == "__main__":
    matplotlib.use("TkAgg") 

    # Generate sample data
    np.random.seed(0)
    X = np.random.rand(100, 2)

    # Instantiate and fit KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap="viridis", marker="o", edgecolor="k")
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red", marker="x", s=100, label="Centroids")
    plt.legend()
    plt.savefig("output.png")
    plt.show()
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)

    print("TF-IDF matrix shape:", X_train_tfidf.shape)

