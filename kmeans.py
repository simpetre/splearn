import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

class Kmeans(object):

    def __init__(self, n_clusters=8, max_iters=100):
        """Perform the Kmeans clustering algorithm on an input set X.
        INPUT: n_clusters: desired number of centroids - manually chosen by user
        max_iters: prevents infinite looping if solution not converging
        OUTPUT: None
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters

        self.centroids_ = None
        self.labels_ = None

    def initialize_centroids(self, X):
        """Define initial state of centroids, based on user preference. X is a
        matrix of n_item rows and n_features columns
        INPUT: features matrix X
        OUTPUT: None
        """
        self.centroids_ = np.array(random.sample(X, k=self.n_clusters))


    def fit(self, X):
        """Define initial state of centroids, based on user preference.
        INPUT: features matrix X
        OUTPUT: Numpy array of closest centroids for each item in X
        """
        closest_centroid = np.empty(shape=(X.shape[0], 1))
        update_closest_centroid = np.empty(shape=(X.shape[0], 1))
        # Initialise closest centroid and centroid. Empty matrices should
        # be arbitrarily different and force at least one loop
        self.initialize_centroids(X)

        iters = 0
        # Loop until either two incremental loops produce no change or stop
        # at max_iters
        while ((closest_centroid != update_closest_centroid).any()
                and (iters < self.max_iters)):
            update_closest_centroid = closest_centroid
            distmat = self.update_dist_matrix(X)
            closest_centroid = np.argmin(distmat, axis=1)
            self.update_centroids(distmat, closest_centroid)
            iters += 1
        self.labels_ = closest_centroid

    def update_dist_matrix(self, X):
         """Update distances to centroids for each row (i.e. sample) in matrix X
         INPUT: X, centroids matrix
         OUTPUT: Updated numpy distance array
         """
         return cdist(X, self.centroids_)

    def update_centroids(self, distmat, closest_centroid):
         """Change positions of centroids based on new clusters
         INPUT: X, centroids matrix, matrix containing closest centroid for each
         item
         OUTPUT: None
         """
         # Loop through and assign a new centroid for each cluster
         for i, centroid in enumerate(self.centroids_):
             self.centroids_[i] = np.mean(X[closest_centroid==i], axis=0)

if __name__ == '__main__':
    # Test algorithm on sklearn's Iris dataset
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    model = Kmeans(n_clusters=2)
    model.fit(X)

    fig, ax = plt.subplots()
    ax = plt.scatter(X[:, 0], X[:, 1])
    cents = np.array(model.centroids_)
    ax = plt.scatter(cents[:, 0], cents[:, 1], s=20, color='red')
    fig.show()
