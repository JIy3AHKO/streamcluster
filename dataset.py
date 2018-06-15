import numpy as np

class ClusterConfig:

    def __init__(self, mean, std, dimensions, weight):
        self._mean = mean
        self._std = std
        self._dimensions = dimensions
        self.weight = weight

    def sample(self, count=1):
        return self._sample() if count == 1 else np.array([self._sample() for _ in range(count)])

    def evlove(self, strength):
        self._mean += np.random.normal(0, strength, self._dimensions)
        self._std += np.random.normal(0, strength, self._dimensions)
        self.weight += np.random.normal(0, strength)

    def _sample(self):
        return np.random.normal(self._mean, self._std, self._dimensions)

    def __repr__(self):
        return "Cluster: mean {} std {} weight {} dims {}".format(self._mean, self._std, self.weight, self.weight)

class DataStream:

    def __init__(self,
                 clusters,
                 dimensions,
                 evolve_speed,
                 evolve_strength,
                 cluster_size=None,
                 seed=None):
        self.cluster_size = cluster_size or (np.ones(clusters) / clusters)
        self.seed = seed
        self.evolve_strength = evolve_strength
        self.n_clusters = clusters
        self.n_dimensions = dimensions
        self.evolve_speed = evolve_speed
        if seed:
            np.random.seed(seed)

        self.cluster_means = np.random.normal(0, 100, (clusters, dimensions))
        self.cluster_stds = np.abs(np.random.normal(0, 10, (clusters, dimensions)))

    def __iter__(self):
        counter = 0
        while True:
            cluster_to_sample = np.random.choice(np.arange(self.n_clusters), p=self.cluster_size)
            counter += 1
            if counter % self.evolve_speed == 0:
                self._evolve()
            yield self.sample_cluster(cluster_to_sample), cluster_to_sample

    def _evolve(self):
        pass

    def sample_cluster(self, cluster_idx):
        return np.random.normal(self.cluster_means[cluster_idx], self.cluster_stds[cluster_idx])