import numpy as np


def linear_weight(x, overlap, mask_edges=[True, True], linspace_axis=1, margin=0.01):
    if (linspace_axis == 0):
        x = x.T
    start_mask = x < overlap
    n_overlap = start_mask[0].sum()
    end_mask = np.zeros_like(start_mask)
    end_mask[:, -n_overlap:] = 1

    x_start = np.clip((x - margin * overlap) / ((1 - 2 * margin) * overlap), 0, 1)
    x_end = np.zeros_like(x_start)
    x_end[:, -n_overlap:] = 1 - x_start[:, :n_overlap]

    start_mask = start_mask if mask_edges[0] else np.zeros_like(x)
    end_mask = end_mask if mask_edges[1] else np.zeros_like(x)
    x_start = start_mask * x_start
    x_end = end_mask * x_end
    x_middle = 1 - (start_mask + end_mask)
    blend_weights = x_start + x_middle + x_end
    if (linspace_axis == 0):
        blend_weights = blend_weights.T
    return blend_weights

# Set mask sides clockwise starting from the top: T, R, B, L
def grid_blend_mask(resolution, overlap, mask_sides=[True, True, True, True], margin=0.01):
    X, Y = np.meshgrid(*[np.linspace(0, 1, resolution + 1)[:-1] + 0.5 / resolution] * 2)
    return linear_weight(X, overlap, mask_edges=[mask_sides[3], mask_sides[1]], margin=margin) * linear_weight(Y, overlap, mask_edges=[mask_sides[0], mask_sides[2]], linspace_axis=0, margin=margin)
