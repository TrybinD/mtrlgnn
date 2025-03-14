import re
import torch


def parse_checkpoint(filename):
    match = re.match(r"epoch=(\d+)-step=(\d+)\.ckpt", filename)
    if match:
        epoch, step = map(int, match.groups())
        return epoch, step
    return -1, -1


def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled


def dist_to_coords(distance_matrix, n_components=2):
    """
    Converts a distance matrix (PyTorch tensor) to coordinates using Multidimensional Scaling (MDS).

    Args:
        distance_matrix (torch.Tensor): A square tensor representing distances (NxN).
        n_components (int): Number of dimensions for output coordinates.

    Returns:
        torch.Tensor: Coordinates tensor of shape (N, n_components).
    """
    N = distance_matrix.size(0)

    # Centering matrix
    J = torch.eye(N) - torch.ones((N, N)) / N

    # Apply double centering
    B = -0.5 * J @ (distance_matrix**2) @ J

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(B)

    # Select top n_components
    idx = torch.argsort(eigvals, descending=True)[:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Calculate coordinates
    coords = eigvecs * eigvals.sqrt().unsqueeze(0)

    return coords
