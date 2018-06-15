from collections import deque
from multiprocessing import Pool

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.stats import norm


class MicroCluster:
    LAST_ID = 0

    def __init__(self, points, timestamps):
        self.data_squares = np.sum(points ** 2, axis=0)
        self.data_sum = np.sum(points, axis=0)
        self.n = len(points)
        self.ts_squares = np.sum(timestamps ** 2)
        self.ts_sum = np.sum(timestamps)
        self.id = MicroCluster.LAST_ID
        MicroCluster.LAST_ID += 1
        self.id_list = []

    def __add__(self, other):
        new_microcluster = MicroCluster(np.array([[]]), np.array([]))
        new_microcluster.data_squares = self.data_squares + other.data_squares
        new_microcluster.data_sum = self.data_sum + other.data_sum
        new_microcluster.n = self.n + other.n
        new_microcluster.ts_squares = self.ts_squares + other.ts_squares
        new_microcluster.ts_sum = self.ts_sum + other.ts_sum
        new_microcluster.id_list = self.id_list
        new_microcluster.id_list.extend([self.id, other.id])
        new_microcluster.id_list.extend(other.id_list)

        return new_microcluster

    def __sub__(self, other):
        new_microcluster = MicroCluster(np.array([[]]), np.array([]))
        new_microcluster.data_squares = self.data_squares - other.data_squares
        new_microcluster.data_sum = self.data_sum - other.data_sum
        new_microcluster.n = self.n - other.n
        new_microcluster.ts_squares = self.ts_squares - other.ts_squares
        new_microcluster.ts_sum = self.ts_sum - other.ts_sum
        new_microcluster.id_list = self.id_list
        new_microcluster.id_list.extend([self.id, other.id])
        new_microcluster.id_list.extend(other.id_list)

        return new_microcluster

    def get_percentile(self, m):
        if self.n <= 2 * m:
            return self.ts_sum / self.n
        else:
            return norm(self.mean('timestamp'), self.std('timestamp')).ppf(m / (2 * self.n))

    def std(self, mode='data', mean=True):
        data = self.data_squares if mode == 'data' else self.ts_squares
        stds_squared = data / self.n - self.mean(mode) ** 2
        if mean:
            stds_squared = np.mean(stds_squared)
        return np.sqrt(np.abs(stds_squared) )

    def mean(self, mode='data'):
        data = self.data_sum if mode == 'data' else self.ts_sum
        return data / self.n

    def distance_to_sample(self, sample):
        return np.sqrt(np.sum((self.mean() - sample) ** 2))

    def distance_to_cluster(self, other):
        return np.sqrt(np.sum((self.mean() - other.mean()) ** 2))

    def __repr__(self):
        return "Microcluster: N {} Mean {} Std {}".format(self.n, self.mean(), self.std())

    def sample(self, shape=None):
        return np.random.normal(self.mean(), self.std(mean=False), (shape, self.data_squares.shape[0]))

class Snapshot:
    def __init__(self):
        self.microclusters = []

    def append(self, item):
        self.microclusters.append(item)

    def __getitem__(self, item):
        return self.microclusters[item]

    def __setitem__(self, key, value):
        self.microclusters[key] = value

    def __delitem__(self, key):
        del self.microclusters[key]

    def __iter__(self):
        return iter(self.microclusters)

    def __sub__(self, other):
        for i, microcluster in enumerate(self.microclusters):
            ids = set(microcluster.id_list)
            for mc in other:
                if ids & set(mc.id_list):
                    self.microclusters[i] = self.microclusters[i] - mc
        return self

    def __len__(self):
        return len(self.microclusters)


class PyramidalTimeFrame:

    def __init__(self, alpha, limit_power):
        self.alpha = alpha
        self.limit = alpha ** limit_power + 1
        self.queues = []

    def append(self, snapshot, timestamp):
        if self.alpha ** len(self.queues) <= timestamp:
            self.queues.append(deque())
        order = self._get_order(timestamp)
        self.queues[order].append((timestamp, snapshot))

        if len(self.queues[order]) > self.limit:
            self.queues[order].popleft()

    def _get_order(self, timestamp):
        order = 0
        while timestamp % self.alpha == 0:
            timestamp /= self.alpha
            order += 1
        return order

    def get_clusters(self, timestamp, time_horizon):
        current_snapshot = self._get_closest_snapshot(timestamp)
        closest_snapshot = self._get_closest_snapshot(timestamp - time_horizon)

        return current_snapshot - closest_snapshot

    def _get_closest_snapshot(self, timestamp):
        closest_distance = np.abs(timestamp - self.queues[0][-1][0])
        closest_snapshot = self.queues[0][-1][1]
        for queue in self.queues:
            for t, snapshot in queue:
                dist = np.abs(timestamp - t)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_snapshot = snapshot

        return closest_snapshot


class CluStream(object):

    def __init__(self, n_microclusters,
                 distance_threshold=25,
                 time_threshold=100,
                 avg_time_points=10,
                 pyramidal_alpha=2,
                 pyramidal_granularity=1,
                 buffer_size=100):
        self.n_microclusters = n_microclusters
        self.micro_clusters = Snapshot()
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_timestamps = []
        self.initialized = False
        self.snapshots = PyramidalTimeFrame(pyramidal_alpha, pyramidal_granularity)
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold
        self.avg_time_points = avg_time_points
        self.macro_centroids = None
        self._micro_cluster_type = MicroCluster
        self._n_deletions = 0
        self._n_merges = 0

    def is_base(self):
        return type(self) == CluStream

    def get_micro_cluster_centroids(self):
        return np.array([m.mean() for m in self.micro_clusters])

    def fit_sample(self, sample, timestamp):
        if not self.initialized:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(sample)
                self.buffer_timestamps.append(timestamp)
            else:
                self.initialize()
        else:
            mc = self._micro_cluster_type(np.array([sample]), np.array([timestamp]))
            cluster_id = self._find_cluster(sample)

            if cluster_id == -1:
                self.micro_clusters.append(mc)
                self._reduce_clusters(timestamp)
            else:
                self.micro_clusters[cluster_id] = self.micro_clusters[cluster_id] + mc

            self.snapshots.append(self.micro_clusters, timestamp)

    def initialize(self):
        self.buffer = np.array(self.buffer)
        self.buffer_timestamps = np.array(self.buffer_timestamps)
        km = KMeans(n_clusters=self.n_microclusters)
        km.fit(self.buffer)
        clusters = km.predict(self.buffer)

        for i in range(self.n_microclusters):
            self.micro_clusters.append(self._micro_cluster_type(self.buffer[clusters == i], self.buffer_timestamps[clusters == i]))

        self.initialized = True

    def calculate_distances(self, sample):
        def distance_to_cluster(cluster_index):
            cluster = self.micro_clusters[cluster_index]
            return cluster.distance_to_sample(sample)

        d = list(map(distance_to_cluster, range(len(self.micro_clusters))))

        return d

    def is_outlier(self, distances, closest_id):
        std = self.micro_clusters[closest_id].std() if self.micro_clusters[closest_id].n != 1 else distances[closest_id]

        if distances[closest_id] > std * self.distance_threshold:
            return -1
        else:
            return closest_id

    def _find_cluster(self, sample):

        if self.is_base():
            cluster_means = np.array([m.mean() for m in self.micro_clusters])
            cluster_means -= sample
            d = np.linalg.norm(cluster_means, axis=1)
        else:
            d = self.calculate_distances(sample)

        closest_id = np.argmin(d)
        closest_id = self.is_outlier(d, closest_id)

        return closest_id

    def _reduce_clusters(self, timestamp):
        for i, mc in enumerate(self.micro_clusters):
            if mc.get_percentile(self.avg_time_points) < timestamp - self.time_threshold:
                del self.micro_clusters[i]
                self._n_deletions += 1
                return
        self._merge_clusters()

    def find_closest_clusters(self):
        cluster_1 = 0
        cluster_2 = 1
        min_distance = self.micro_clusters[cluster_1].distance_to_cluster(self.micro_clusters[cluster_2])
        for i in range(len(self.micro_clusters)):
            for j in range(i + 1, len(self.micro_clusters)):
                d = self.micro_clusters[i].distance_to_cluster(self.micro_clusters[j])
                if d < min_distance:
                    cluster_1 = i
                    cluster_2 = j
                    min_distance = d

        return cluster_1, cluster_2

    def _merge_clusters(self):
        self._n_merges += 1

        if self.is_base():
            cluster_means = np.array([m.mean() for m in self.micro_clusters])
            d_matrix = np.linalg.norm(cluster_means[..., None] - cluster_means[..., None].T, axis=1, ord=2)
            np.fill_diagonal(d_matrix, d_matrix.max() + 1)
            cluster_1, cluster_2 = np.unravel_index(np.argmin(d_matrix), d_matrix.shape)
        else:
            cluster_1, cluster_2 = self.find_closest_clusters()

        self.micro_clusters[cluster_1] = self.micro_clusters[cluster_1] + self.micro_clusters[cluster_2]
        del self.micro_clusters[cluster_2]

    def macro_clusters(self, n_clusters, max_iters=1000):
        data = np.array([k.mean() for k in self.micro_clusters])
        counts = np.array([k.n for k in self.micro_clusters])
        centroids = np.random.choice(np.arange(len(data)), n_clusters, p=counts / counts.sum(), replace=False)
        centroids = data[centroids]
        for i in range(max_iters):
            distances = np.array([[((sample - centroid) ** 2).sum() for centroid in centroids] for sample in data])
            C = np.argmin(distances, axis=1)

            distances = np.min(distances, axis=1)
            centroids = np.array(
                [np.sum(data[C == k] * counts[C == k, np.newaxis], axis=0) / (counts[C == k, np.newaxis].sum()) for k in
                 range(n_clusters)])

            for i in range(n_clusters):
                if len(data[C == i]) == 0:
                    farest = np.argmax(distances)
                    centroids[i] = data[farest]
                    distances[farest] = -1

        self.macro_centroids = centroids

        return centroids

    def transform(self, data):
        if self.macro_centroids is None:
            self.macro_centroids = np.array([m.mean() for m in self.micro_clusters])
        return pairwise_distances_argmin(data, self.macro_centroids)
