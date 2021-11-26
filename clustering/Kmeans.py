import numpy as np
import matplotlib.pyplot as plt

from clustering.utils import euclidean_distance


class KMeans:
    
    def __init__(self , K : int = 5 , epochs : int = 100 ,random_state : int = 5 ,verbose : bool = False) -> None:
    
        """
            Randomly pick k points at first as centroids
            iteratively check the euclidean distance of each point with the centroid and find the nearest centroid cluster
            find the mean of all the cluster , make the centroid to move to the mean of the current cluster.
            if centroids doesnot change (no convergence) then break the loop or else loop untill max iter given
        """
    
        self.K = K
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        self.clusters = [[] for _ in range(K)] # creating list for each cluster to append sample indexes
        self.centroids = []  # k centroids

    def predict(self , X):
        
        self.X = X
        self.n_samples ,_ = X.shape

        np.random.seed(self.random_state)
        random_index_centroids = np.random.choice(self.n_samples , self.K , replace=False)
        self.centroids = [X[idx] for idx in random_index_centroids]

        for _ in range(self.epochs):
            
            self.clusters = self._make_clusters()
            old_centroids = self.centroids
            self._update_centroids()
            if self._is_converged(old_centroids):
                break
    
        return self._get_cluster_labels()

    def _get_cluster_labels(self):

        #give each sample its cluster index
        groups = np.random.randn(self.n_samples)

        for cluster_idx , cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                groups[sample_idx] = cluster_idx

        return groups

    def _make_clusters(self):

        clusters = [[] for _ in range(self.K)]
        for idx , sample in enumerate(self.X):
            centroid_idx = self._get_nearest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _get_nearest_centroid(self , sample):
        all_distances = [euclidean_distance(sample , centroid) for centroid in self.centroids]

        return np.argmin(all_distances)

    def _update_centroids(self):
        
        #find the mean of all the sample in the current cluster and update the centroid to the mean
        self.centroids = [np.mean(self.X[samples_idx] , axis=0) for samples_idx in self.clusters]
        
    def _is_converged(self , old_centroids):

        # if euclidean distance between old and updated centroids are same , the centroids are converged
        total_distances = [euclidean_distance(old_centroids[i] , self.centroids[i]) for i in range(self.K)]
        return sum(total_distances) == 0
    
    def plot(self):
        _, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="red", linewidth=5)

        plt.show()

