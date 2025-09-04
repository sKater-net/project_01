import torch
import numpy as np
from scipy.spatial.distance import cdist

def calculate_map(qB, rB, query_L, retrieval_L):
    num_query = qB.shape[0]
    map = 0.0
    
    for i in range(num_query):
        gnd = (np.dot(query_L[i], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
            
        hamm = cdist(qB[i].reshape(1, -1), rB, metric='hamming')
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map += np.mean(count / tindex)
        
    return map / num_query

def build_hypergraph_edges(features, k=5, method='knn'):
    if method == 'knn':
        return knn_hypergraph(features, k)
    else:
        raise ValueError("Unknown hypergraph method")

def knn_hypergraph(features, k=5):
    n = features.shape[0]
    dists = torch.cdist(features, features)
    H = torch.zeros(n, n)
    
    for i in range(n):
        _, indices = torch.topk(dists[i], k=k+1, largest=False)
        H[i, indices[1:]] = 1.0
        
    return H