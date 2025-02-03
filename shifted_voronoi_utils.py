import torch
import numpy as np
import sklearn
def two_frame_knn(pc1, pc2):
    distances_1 = sklearn.metrics.pairwise.euclidean_distances(pc1, pc2)
    #arr_non_zero_min = np.where(distances_1 == 0, np.inf, distances_1)
    # Find the row-wise minimum, ignoring zeros
    p2vmap = np.argmin(distances_1, axis=1)
    v2pmap = np.argmin(distances_1, axis=0)
    # Replace np.inf with np.nan or another placeholder if no non-zero elements exist in a row
    #row_min[np.isinf(row_min)] = np.nan
    return p2vmap, v2pmap


def two_frame_knn_(pc1, pc2):
    distances_1 = sklearn.metrics.pairwise.euclidean_distances(pc1, pc2)
    arr_non_zero_min = np.where(distances_1 == 0, np.inf, distances_1)
    # Find the row-wise minimum, ignoring zeros
    row_min = np.argmin(arr_non_zero_min, axis=1)
    # Replace np.inf with np.nan or another placeholder if no non-zero elements exist in a row
    #row_min[np.isinf(row_min)] = np.nan
    return row_min



def farthest_point_sample(xyz, npoint):
    # orig
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    print("---------\n fps call ----------- \n")
    if xyz.ndim == 2:
        xyz = np.expand_dims(xyz, axis = 0)
    xyz = torch.from_numpy(xyz)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    centroids = centroids.numpy()
    distance = distance.numpy()
    farthest = farthest.numpy()
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    batch_indices = batch_indices.numpy()
    for i in range(npoint):
        if i%5000 == 0:
            print(i, npoint)
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.numpy()
        #dist = torch.abs(torch.sum(xyz, -1) - torch.sum(centroid, -1))
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        #farthest = farthest[1]
        #farthest = np.max(distance, -1)[1]

    return centroids