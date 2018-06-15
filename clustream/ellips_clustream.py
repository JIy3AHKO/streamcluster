import numpy as np
from .base_clustream import CluStream, MicroCluster
from scipy.spatial.distance import mahalanobis, euclidean


class EllipseMicroCluster(MicroCluster):
    def distance_to_cluster(self, other):
        sigma0 = self.std(mean=False) + 1e-7
        sigma1 = other.std(mean=False) + 1e-7
        mu1 = other.mean()
        k = mu1.shape[0]

        # KL-divergence for multinominal normal distribution with diagonal covariation matrix case
        return 1/2 * (np.linalg.norm(sigma0 / sigma1, ord=1) + self.distance_to_sample(mu1) + np.log(np.prod(sigma1) / np.prod(sigma0)) - k)

    def distance_to_sample(self, sample):
        sample = np.squeeze(sample)
        return mahalanobis(self.mean(), sample, np.diag(self.std(mean=False) ** 2)) if self.n > 1 else euclidean(self.mean(), sample)


class EllipseCluStream(CluStream):

    def __init__(self, n_microclusters, **kwargs):
        super().__init__(n_microclusters, **kwargs)
        self._micro_cluster_type = EllipseMicroCluster

    def is_outlier(self, distances, closest_id):
        std = self.micro_clusters[closest_id].std() if self.micro_clusters[closest_id].n != 1 else distances[closest_id]

        if distances[closest_id] > std * self.distance_threshold:
            return -1
        else:
            return closest_id

    def macro_clusters(self, n_clusters, max_iters=1000):
        data = np.array([k.mean() for k in self.micro_clusters])
        counts = np.array([k.n for k in self.micro_clusters])
        centroids = np.random.choice(np.arange(len(data)), n_clusters, p=counts / counts.sum(), replace=False)
        centroids = data[centroids]
        for i in range(max_iters):
            distances = np.array([[((sample - centroid) ** 2).sum() for centroid in centroids] for sample in data])

            C = np.argmin(distances, axis=1)

            distances = np.min(distances, axis=1)
            prev = centroids.copy()
            centroids = np.array(
                [np.sum(data[C == k] * counts[C == k, np.newaxis], axis=0) / (counts[C == k, np.newaxis].sum()) for k in
                 range(n_clusters)])

            for i in range(n_clusters):
                if len(data[C == i]) == 0:
                    farest = np.argmax(distances)
                    centroids[i] = data[farest]
                    distances[farest] = -1
            diff = np.mean([euclidean(prev[i], centroids[i]) for i in range(n_clusters)])
            if diff < 1e-5:
                break

        self.macro_centroids = centroids
        distances = np.array([[((sample - centroid) ** 2).sum() for centroid in centroids] for sample in data])

        C = np.argmin(distances, axis=1)

        self.micro_clusters_clusters = C

        return centroids

    def transform(self, data):
        tmp = self.macro_centroids
        self.macro_centroids = None
        mc = super().transform(data)
        self.macro_centroids = tmp
        macro_c = self.micro_clusters_clusters[mc]
        return macro_c