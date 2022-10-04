from cellpose import metrics as cp_metrics
from cellpose import utils as cp_utils
from cellstitch.alignment import *
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError
from skimage.measure import marching_cubes, mesh_surface_area


#--------------------
# Helper functions
#--------------------

def get_num_cells(masks):
    return len(np.unique(masks)[1:])


def get_avg_vol(masks):
    total_vol = (masks != 0).sum()
    n = get_num_cells(masks)
    return total_vol / n


def sample_indices(masks, n=50):
    """Randomly sample instances from the list of input 3D masks"""
    indices = []
    for mask in masks:
        n = min(get_num_cells(mask), n)
        assert n > 0, "Empty masks"
        indices.append(np.random.choice(np.unique(mask)[1:], n))
    return indices
    

def match_lbls(source_mask, target_mask, sc_lbl):
    """
    Find matched label between source mask & target mask with largest IoU area,
    Return target label
    """
    coords = np.nonzero(source_mask == sc_lbl)
    cand_lbls, counts = np.unique(target_mask[coords], return_counts=True)  # Candidate labels

    if len(cand_lbls) == 0:
        tg_lbl = -1
    else:
        # First check whether argmax matched lbl has IoU exceeding threshold
        if cand_lbls[0] == 0: # Ignore label 0
            cand_lbls = cand_lbls[1:]
            counts = counts[1:]
        
        max_intersect = counts.max()
        tg_lbl = cand_lbls[counts.argmax()]
        
    return tg_lbl


def _compute_convex_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    # Reference:
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    """    
    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"
    
    points = np.argwhere(image).astype(np.int16)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d
    
    mask = np.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:,:,0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask


def _calc_surface_area(mask):
    verts, faces, _, _ = marching_cubes(mask)
    return mesh_surface_area(verts, faces)


#--------------------
# Evaluation metrics
#--------------------

def average_precision(masks_true, masks_pred, threshold):
    iou = cp_metrics._intersection_over_union(masks_true, masks_pred)
    lbls_true, lbls_pred = np.unique(masks_true)[1:, ].tolist(), np.unique(masks_pred)[1:, ].tolist()

    tp = 0
    for lbl_true in lbls_true:
        for lbl_pred in lbls_pred:
            if iou[lbl_true][lbl_pred] >= threshold:
                tp += 2  # count the number of matched cells

    fp = len(lbls_pred) - tp / 2
    fn = len(lbls_true) - tp / 2

    ap = tp / (tp + fp + fn)

    return [ap, tp, fp, fn]  # return as list for easy convert to dataframe


def avg_symmetric_surf_dist(mask, pred, mask_lbls, pred_lbls):
    """
    Calculate Average Symmetric Surface Distance for each paired mask label & prediction
     (Randomly sample 100 instances: presampled as `mask_lbls` & `pred_lbls`)
    """
    
    # Helper functions
    def _calc_assd(mask, pred):
        # Take outline 
        mask_outline = cp_utils.masks_to_outlines(mask)
        pred_outline = cp_utils.masks_to_outlines(pred)

        # Calculate ASSD for each outline pixel
        m1 = np.ones_like(mask, dtype=np.uint8)
        m1[pred_outline] = 0
        m2 = np.ones_like(pred, dtype=np.uint8)
        m2[mask_outline] = 0

        dist1 = ndi.distance_transform_edt(m1)[np.nonzero(mask_outline)]
        dist2 = ndi.distance_transform_edt(m2)[np.nonzero(pred_outline)]
        assd = np.concatenate([dist1, dist2]).mean()

        return assd
    
    assd = []
    for sc_lbl, tg_lbl in zip(mask_lbls, pred_lbls):
        if tg_lbl != -1:
            assd.append(_calc_assd(mask == sc_lbl, pred == tg_lbl))
    return np.mean(assd) if len(assd) > 0 else 0

    
def compactness_convexity_ae(mask, pred, mask_lbls, pred_lbls, eps=1e-10):
    """
    Calculate Absolute Error of the following metrics between mapped (ground-truth, predicted) masks:
     - (1). Compactness
     - (2). Convexity
     (Randomly sample 100 instances: presampled as `mask_lbls` & `pred_lbls`)
    """    
    comp_abs_errors = np.zeros(len(mask_lbls)) # compactness errors
    conv_abs_errors = np.zeros(len(mask_lbls)) # convexity errors
    
    for i, (mask_lbl, pred_lbl) in enumerate(zip(mask_lbls, pred_lbls)):
        if pred_lbl == -1:
            continue
            
        mask_bin = (mask == mask_lbl).astype(np.uint8)
        pred_bin = (pred == pred_lbl).astype(np.uint8)
        
        try:
            # Compactness 
            vm, am = mask_bin.sum(), _calc_surface_area(mask_bin)
            vp, ap = pred_bin.sum(), _calc_surface_area(pred_bin)
            cm = 36*np.pi * vm**2 / (am**3+eps)
            cp = 36*np.pi * vp**2 / (ap**3+eps)
            comp_abs_errors[i] = np.abs(cp-cm) / cm
        
            # Convexity
            mask_bin_ch = _compute_convex_hull(mask_bin)
            pred_bin_ch = _compute_convex_hull(pred_bin)
        
            am_ch = _calc_surface_area(mask_bin_ch)
            ap_ch = _calc_surface_area(pred_bin_ch)
            cm = am_ch / (am+eps)
            cp = ap_ch / (ap+eps)
            conv_abs_errors[i] = np.abs(cp-cm) / cm
        except QhullError as qe:
            print("2D mask is detected, can't compute 3D convex hull, passed")
            pass
        
    return {'Compactness': comp_abs_errors.mean(), 'Convexity': conv_abs_errors.mean()}
