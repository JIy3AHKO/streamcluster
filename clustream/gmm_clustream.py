from pomegranate import GeneralMixtureModel, MultivariateGaussianDistribution
from sklearn.mixture.gaussian_mixture import GaussianMixture
import numpy as np

from .base_clustream import CluStream, MicroCluster


class GmmMicroCluster(MicroCluster):
    pass


class GmmCluStream(CluStream):
    def __init__(self, n_microclusters, **kwargs):
        super().__init__(n_microclusters, **kwargs)
        self.gmm_model = None

    def calculate_distances(self, sample):
        cluster_means = np.array([m.mean() for m in self.micro_clusters])
        cluster_means -= sample
        d = np.linalg.norm(cluster_means, axis=1)
        return d

    def find_closest_clusters(self):
        cluster_means = np.array([m.mean() for m in self.micro_clusters])
        d_matrix = np.linalg.norm(cluster_means[..., None] - cluster_means[..., None].T, axis=1, ord=2)
        np.fill_diagonal(d_matrix, d_matrix.max() + 1)
        cluster_1, cluster_2 = np.unravel_index(np.argmin(d_matrix), d_matrix.shape)
        return cluster_1, cluster_2

    def macro_clusters(self, n_clusters, max_iters=1000, **kwargs):
        data = np.concatenate([m.sample(np.int32(np.ceil(m.n / 100))) for m in self.micro_clusters])

        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(data)
        self.macro_centroids = gmm.means_
        self.gmm_model = gmm

    def transform(self, data):
        return self.gmm_model.predict(data)