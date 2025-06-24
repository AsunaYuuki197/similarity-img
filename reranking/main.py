import os
import time

import torch
import numpy as np
from pandas import read_pickle

from revisitop import configdataset, compute_map_and_print
from reranker.qe_reranking import aqe_reranking
from reranker.cas_reranking import cas_reranking

def main():
    dataset = 'roxford5k'
    # dataset = 'rparis6k'
    data_root = '../dataset'
    feature_name = 'delg_r50' # This give high mAP in roxford 
    # feature_name = 'gl18-tl-resnet101-gem-w'

    cfg = configdataset(dataset, data_root)
    feature = read_pickle(os.path.join(data_root, dataset, 'features/{}.pkl'.format(feature_name)))

    qvecs = feature['query']
    vecs = feature['db']

    start = time.time()
    scores = np.dot(qvecs, vecs.T) # Cosine similarity
    ranks = np.argsort(-scores, axis=1).T # Sort to get top
    compute_map_and_print(dataset, ranks, cfg['gnd']) # calculate mean Average Precision metrics and mP@K
    print("Time executed (secs):", time.time() - start)

    
    start = time.time()
    scores = aqe_reranking(qvecs, vecs) # Using Average Query Expansion reranking algo
    ranks = np.argsort(-scores, axis=1).T # Sort to get top
    compute_map_and_print(dataset, ranks, cfg['gnd']) # calculate mean Average Precision metrics and mP@K
    print("Time executed (secs):", time.time() - start)


    start = time.time()
    dist = cas_reranking(qvecs, vecs, metric='euclidean', k1=6, k2=60, k3=70, k4=7, k5=80, device=torch.device('cpu')) # Using Cluster-Aware Similarity Algo
    ranks = np.argsort(dist, axis=1).T # Sort to get top
    compute_map_and_print(dataset, ranks, cfg['gnd']) # calculate mean Average Precision metrics and mP@K
    print("Time executed (secs):", time.time() - start)


if __name__ == '__main__':
    main()