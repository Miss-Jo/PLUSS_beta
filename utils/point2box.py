"""
Point-2-Box Conversion Module
Converts point prompts to pseudo-bounding boxes using DBSCAN-like clustering
"""

import torch
import numpy as np
from sklearn.cluster import DBSCAN


def point_to_box(points, labels=None, radius=None, min_pts=3, image_shape=None):
    """
    Convert point prompts to pseudo-bounding boxes (Algorithm 1 in paper)
    
    Uses DBSCAN clustering to group points and generate boxes
    Filters out noise points before clustering
    
    Args:
        points: Point coordinates [N, 2] in (x, y) format
        labels: Point labels [N] (1 for positive, 0 for negative)
        radius: Neighborhood radius (R in paper), default is 1/8 of image diagonal
        min_pts: Minimum points for core point (default=3)
        image_shape: (H, W) for computing default radius
        
    Returns:
        boxes: Pseudo-bounding boxes [M, 4] in (x, y, w, h) format
        point_clusters: Cluster assignment for each point
    """
    if points is None or len(points) == 0:
        return np.array([]), np.array([])
    
    # Convert to numpy if tensor
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if labels is not None and torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Filter to only positive points if labels provided
    if labels is not None:
        positive_mask = labels > 0
        points = points[positive_mask]
    
    if len(points) == 0:
        return np.array([]), np.array([])
    
    # Set default radius if not provided
    if radius is None:
        if image_shape is not None:
            H, W = image_shape
            diagonal = np.sqrt(H**2 + W**2)
            radius = diagonal / 8  # Empirically set to 1/8 of diagonal
        else:
            # Estimate from point distribution
            radius = np.std(points) * 2
    
    # Step 1: Categorize points into core, border, and noise
    # Core points: N_R(X_i) > min_pts
    # Border points: N_R(X_i) <= min_pts AND X_i in U_R(core_pts)
    # Noise points: N_R(X_i) <= min_pts AND X_i NOT in U_R(core_pts)
    
    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=radius, min_samples=min_pts).fit(points)
    cluster_labels = clustering.labels_
    
    # Filter noise points (label = -1)
    valid_mask = cluster_labels != -1
    valid_points = points[valid_mask]
    valid_labels = cluster_labels[valid_mask]
    
    if len(valid_points) == 0:
        return np.array([]), np.array([])
    
    # Step 2: Generate bounding boxes for each cluster
    unique_clusters = np.unique(valid_labels)
    boxes = []
    
    for cluster_id in unique_clusters:
        cluster_mask = valid_labels == cluster_id
        cluster_points = valid_points[cluster_mask]
        
        # Get bounding box: (x_min, y_min, x_max, y_max)
        x_min = cluster_points[:, 0].min()
        y_min = cluster_points[:, 1].min()
        x_max = cluster_points[:, 0].max()
        y_max = cluster_points[:, 1].max()
        
        # Convert to (x, y, w, h) format
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        
        boxes.append([x, y, w, h])
    
    boxes = np.array(boxes)
    
    return boxes, cluster_labels


def visualize_point_clustering(points, labels, boxes, image_shape, save_path=None):
    """
    Visualize point clustering and generated boxes
    
    Args:
        points: Point coordinates [N, 2]
        labels: Cluster labels [N]
        boxes: Generated boxes [M, 4]
        image_shape: (H, W)
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Set image extent
    H, W = image_shape
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Invert y-axis
    ax.set_aspect('equal')
    
    # Plot points with colors based on cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            color = 'black'
            marker = 'x'
        else:
            marker = 'o'
        
        mask = labels == label
        cluster_points = points[mask]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  c=[color], marker=marker, s=50, alpha=0.6,
                  label=f'Cluster {label}' if label != -1 else 'Noise')
    
    # Draw bounding boxes
    for box in boxes:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, 
                                 linewidth=2, 
                                 edgecolor='red', 
                                 facecolor='none')
        ax.add_patch(rect)
    
    ax.legend()
    ax.set_title('Point Clustering and Box Generation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


class Point2BoxConverter:
    """
    Converter class for point-to-box transformation
    Handles batch processing and caching
    """
    
    def __init__(self, radius=None, min_pts=3, image_size=(512, 512)):
        self.radius = radius
        self.min_pts = min_pts
        self.image_size = image_size
        
    def __call__(self, points, labels=None):
        """
        Convert points to boxes
        
        Args:
            points: List of point arrays [N_i, 2] for each image in batch
            labels: List of label arrays [N_i] for each image (optional)

        Returns:
            boxes: List of pseudo-bounding boxes for each image
        """
        # Handle list of arrays (batch from forward_point_branch)
        if isinstance(points, list):
            batch_boxes = []
            for i in range(len(points)):
                pts = points[i]  # numpy array [N, 2]
                lbs = labels[i] if labels is not None else None  # numpy array [N]

                # Convert single image's points to boxes
                boxes, _ = point_to_box(
                    pts, lbs,
                    radius=self.radius,
                    min_pts=self.min_pts,
                    image_shape=self.image_size
                )
                batch_boxes.append(boxes)
            return batch_boxes

        # Handle single tensor (backward compatibility)
        if isinstance(points, torch.Tensor):
            if points.dim() == 3:
                # Batch processing [B, N, 2]
                batch_boxes = []
                for i in range(points.shape[0]):
                    pts = points[i]
                    lbs = labels[i] if labels is not None else None
                    boxes, _ = point_to_box(
                        pts, lbs, 
                        radius=self.radius,
                        min_pts=self.min_pts,
                        image_shape=self.image_size
                    )
                    batch_boxes.append(boxes)
                return batch_boxes
        
        # Single sample
        boxes, _ = point_to_box(
            points, labels,
            radius=self.radius,
            min_pts=self.min_pts,
            image_shape=self.image_size
        )
        
        return boxes
