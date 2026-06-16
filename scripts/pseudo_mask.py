import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import center_of_mass

def get_centroids(masks):
    """
    Compute centroids of binary instance segmentation masks.
    
    Parameters:
        masks (numpy array): (H, W, N) binary mask for N instances.
    
    Returns:
        List of (x, y) centroids for each instance.
    """
    centroids = []
    for i in range(masks.shape[-1]):
        y, x = center_of_mass(masks[:, :, i])  # Compute centroid
        centroids.append((x, y))  # Store (x, y) as tuple
    return np.array(centroids)

def compute_cost_matrix(att_centroids, mask_centroids):
    """
    Compute Euclidean distance cost matrix between attention maps and instance masks.
    
    Parameters:
        att_centroids (numpy array): (N, 2) centroid coordinates of attention maps.
        mask_centroids (numpy array): (M, 2) centroid coordinates of instance masks.
    
    Returns:
        cost_matrix (numpy array): (N, M) distance matrix.
    """
    N, M = len(att_centroids), len(mask_centroids)
    cost_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            cost_matrix[i, j] = np.linalg.norm(att_centroids[i] - mask_centroids[j])  # Euclidean distance
    return cost_matrix

def match_attention_to_masks(att_maps, instance_masks):
    """
    Match attention maps to instance masks using bipartite matching.
    
    Parameters:
        att_maps (numpy array): (H, W, N) attention maps.
        instance_masks (numpy array): (H, W, M) binary instance masks.
    
    Returns:
        matches (dict): Mapping from attention index to instance index.
    """
    # Get centroids
    att_centroids = get_centroids(att_maps)
    mask_centroids = get_centroids(instance_masks)

    # Compute cost matrix
    cost_matrix = compute_cost_matrix(att_centroids, mask_centroids)

    print(cost_matrix)
    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create match dictionary
    matches = {att_idx: mask_idx for att_idx, mask_idx in zip(row_ind, col_ind)}
    return matches

# Example usage
H, W, N, M = 128, 128, 64, 30  # Image size, 64 attention maps, 30 instance masks
att_maps = np.random.rand(H, W, N)  # Random attention maps (normalized)
instance_masks = np.random.randint(0, 2, (H, W, M))  # Random binary instance masks

matches = match_attention_to_masks(att_maps, instance_masks)
print("Attention-to-Mask Matching:", matches)

# obj_mask_dir = '/media/hcis-s15/ssd2/data_collection/ap_Town01/t1/variant_scenario/1/mask/object/00001006.npy' # (c, h, w)
