import argparse

import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dataset import DataStream
from clustream import CluStream, EllipseCluStream, GmmCluStream


def preproc_categortical(df, column):
    return df.drop(column, axis=1)


def calculate_ssq(data, clusters, centroids):
    ssq = []
    for c in np.unique(clusters):
        ssq.append(np.sum((data[clusters == c] - centroids[c]) ** 2))

    return np.mean(ssq)

def calculate_metrics(algo, data_part):
    macro_centroids = algo.macro_clusters(args.c)
    clusters = algo.transform(data_part)
    ssq = calculate_ssq(data_part, clusters, algo.macro_centroids)
    try:
        silh = silhouette_score(data_part, clusters, sample_size=10000)
    except:
        silh = -1
    print('Metrics at', len(data_part))
    print('SSQ:', ssq)
    print('silhouette score:', silh)
    print()


algos = {
    'naive': CluStream,
    'ellipse': EllipseCluStream,
    'gmm': GmmCluStream,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int, default=10)
    parser.add_argument('-n', nargs='*', type=float, default=0.33)
    parser.add_argument('-c', type=int, default=5)
    parser.add_argument('-t', type=int, default=100000)
    parser.add_argument('-a', type=str, choices=list(algos.keys()), default='naive')
    parser.add_argument('-d', type=str, choices=['kdd99', 'gen', 'kdd98'], default='kdd99')
    parser.add_argument('--sigma', type=float, default=512)
    parser.add_argument('--buffer', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw', action='store_true')

    args = parser.parse_args()
    checkpoints = []
    np.random.seed(args.seed)
    if args.d == 'kdd99':
        data = pd.read_csv('data/kddcup.data.corrected', header=None)
        N = max(args.n)
        checkpoints = [int(n * len(data)) for n in args.n]
        data = data[:int(N * len(data))]
        labels = data[41]
        data.drop(41, axis=1, inplace=True)
        data = preproc_categortical(data, 1)
        data = preproc_categortical(data, 2)
        data = preproc_categortical(data, 3)
        data = preproc_categortical(data, 6)
        data = preproc_categortical(data, 11)
        data = preproc_categortical(data, 20)
        data = preproc_categortical(data, 21)
    elif args.d == 'kdd98':
        data = pd.read_csv('data/cup98LRN.txt')
        for c in data.columns:
            if data[c].dtype != float:
                data.drop(c, axis=1, inplace=True)
        data.fillna(method='backfill', inplace=True)
        data = data[data.columns[:42]]
    elif args.d == 'gen':
        ds = DataStream(args.c, 2, 1, 1, seed=args.seed)
        s = iter(ds)
        data = [next(s) for _ in range(int(10e5 * max(args.n)))]

        labels = [x[1] for x in data]
        data = [x[0] for x in data]
        data = pd.DataFrame(data)

    if not args.raw:
        from sklearn.preprocessing import StandardScaler

        data = StandardScaler().fit_transform(data)
        p = PCA(2)
        p.fit(data)
        l1_errors = np.mean(np.abs(data - p.inverse_transform(p.transform(data))), axis=1)
        max_error = np.percentile(l1_errors, (len(data) - 2) / len(data) * 100)

        data = data[l1_errors < max_error]
    else:
        data = data.values


    a = algos[args.a](args.r * args.c, buffer_size=args.buffer, time_threshold=args.t, distance_threshold=args.sigma)

    for i, row in tqdm.tqdm(enumerate(data), total=len(data)):
        a.fit_sample(np.array(row), i)
        if i in checkpoints:
            calculate_metrics(a, data[:i])


    macro_centroids = a.macro_clusters(args.c)

    ssqs = []
    silhs = []
    n_r = 10 if args.d != 'gen' else 1
    for i in range(n_r):
        clusters = a.transform(data)
        ssq = calculate_ssq(data, clusters, a.macro_centroids)
        silh = silhouette_score(data, clusters, sample_size=10000)
        print('SSQ:', ssq)
        print('silhouette score:', silh)
        ssqs.append(ssq)
        silhs.append(silh)
    print('AVG. SSQ:', np.mean(ssqs))
    print('AVG. silhouette score:',  np.mean(silhs))

    if args.d == 'gen':
        plt.scatter(data[:, 0], data[:, 1], c=clusters)
        micro = a.get_micro_cluster_centroids()
        macro = a.macro_centroids

        plt.scatter(micro[:, 0], micro[:, 1], 5, c='red')
        plt.scatter(macro[:, 0], macro[:, 1], 10, c='blue')

        plt.show()
    filename = str(args) + '.txt'

