# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib
from scipy import ndimage

LABEL_NCR, LABEL_ED, LABEL_ET = 1, 2, 3


def _largest_cc(mask):
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def _zscore_in(vol, mask):
    vals = vol[mask]
    mu, sd = vals.mean(), vals.std() + 1e-6
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vol[mask] - mu) / sd
    return out


def load_modal(path_or_obj):
    img = nib.load(path_or_obj) if not isinstance(path_or_obj, nib.Nifti1Image) else path_or_obj
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    data = np.clip(data, 0, None)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    return data, zooms


def segment_case(vols, zooms):
    t1n, t1c, t2w, t2f = vols
    shape = t1n.shape

    brain = t1n > np.percentile(t1n, 30)
    brain = ndimage.binary_closing(brain, iterations=3)
    brain = ndimage.binary_fill_holes(brain)
    brain = _largest_cc(brain)
    brain = ndimage.binary_dilation(brain, iterations=2)

    z_t1n = _zscore_in(t1n, brain)
    z_t1c = _zscore_in(t1c, brain)
    z_t2f = _zscore_in(t2f, brain)

    csf = brain & (z_t1n < -1.2) & (z_t2f < 1.2)
    wt = z_t2f > 1.0
    wt = wt & ~csf
    wt = ndimage.binary_closing(wt, iterations=2)
    wt = ndimage.binary_opening(wt, iterations=1)
    wt = _largest_cc(wt)
    wt = ndimage.binary_fill_holes(wt)

    wt_fill = ndimage.binary_fill_holes(wt)
    if wt_fill.sum() < 50:
        return np.zeros(shape, dtype=np.uint8), brain
    wt_dil = ndimage.binary_dilation(wt_fill, iterations=4)

    et = wt_dil & ((z_t1c > 1.6) | ((z_t1c - z_t1n) > 1.3))
    et = ndimage.binary_opening(et, iterations=1)
    labeled, n = ndimage.label(et)
    if n > 1:
        sizes = ndimage.sum(et, labeled, range(1, n + 1))
        keep = labeled == (np.argmax(sizes) + 1)
        small = et & ~keep
        if sizes.max() < 200:
            for i in np.argsort(sizes)[::-1]:
                comp = labeled == (i + 1)
                if sizes[i] >= 60:
                    keep = keep | comp
                else:
                    break
        et = keep | (small & wt_fill)
    et = et & wt_dil
    if et.sum() < 30:
        et = np.zeros(shape, dtype=bool)

    if et.sum() > 0:
        et_dil = ndimage.binary_dilation(et, iterations=2)
        ncr = et_dil & (z_t1c < 0.2) & wt_fill
        ncr = ndimage.binary_opening(ncr, iterations=1)
        tc_seed = ndimage.binary_dilation(et | ncr, iterations=3)
    else:
        ncr = np.zeros(shape, dtype=bool)
        tc_seed = ndimage.binary_dilation(wt_fill & (z_t1c > 0.5), iterations=3)
        if tc_seed.sum() == 0:
            tc_seed = ndimage.binary_erosion(wt_fill, iterations=6)

    ed = wt_fill & ~tc_seed
    ed = ndimage.binary_opening(ed, iterations=1)
    ncr_final = ncr & wt_fill

    ed = _drop_small(ed, 80)
    et = _drop_small(et, 60)
    ncr_final = _drop_small(ncr_final, 60)

    labels = np.zeros(shape, dtype=np.uint8)
    labels[ed] = LABEL_ED
    labels[ncr_final] = LABEL_NCR
    labels[et] = LABEL_ET
    return labels, brain


def _drop_small(mask, min_voxels):
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(n):
        if sizes[i] >= min_voxels:
            keep |= labeled == (i + 1)
    return keep


def extract_features(labels, vols, zooms, case_name):
    t1n, t1c, t2w, t2f = vols
    vox_cc = np.prod(zooms) / 1000.0
    regions = {
        'WT': (labels > 0).astype(bool),
        'TC': np.isin(labels, [LABEL_NCR, LABEL_ET]),
        'ET': (labels == LABEL_ET),
        'NCR': (labels == LABEL_NCR),
        'ED': (labels == LABEL_ED),
    }
    out = {
        'case_name': case_name,
        'voxel_spacing_mm': [round(z, 2) for z in zooms],
        'volume_shape': list(labels.shape),
        'regions': {},
        'modalities': ['t1n', 't1c', 't2w', 't2f'],
    }
    for name, m in regions.items():
        n = int(m.sum())
        entry = {'voxels': n, 'volume_cm3': round(n * vox_cc, 2)}
        if n > 0:
            idx = np.array(np.nonzero(m))
            com = idx.mean(axis=1)
            entry['centroid_voxel'] = [int(round(c)) for c in com]
            extent = (idx.max(axis=1) - idx.min(axis=1)) * np.array(zooms)
            entry['max_diameter_mm'] = int(round(float(np.linalg.norm(extent))))
            for mname, vol in zip(['t1n', 't1c', 't2w', 't2f'], [t1n, t1c, t2w, t2f]):
                vals = vol[m]
                entry[f'{mname}_intensity'] = {
                    'mean': round(float(vals.mean()), 1),
                    'std': round(float(vals.std()), 1),
                }
        out['regions'][name] = entry
    if out['regions']['WT']['voxels'] > 0 and out['regions']['TC']['voxels'] > 0:
        out['ratios'] = {
            'TC_over_WT': round(out['regions']['TC']['volume_cm3'] / max(out['regions']['WT']['volume_cm3'], 1e-6), 3),
            'ET_over_TC': round(out['regions']['ET']['volume_cm3'] / max(out['regions']['TC']['volume_cm3'], 1e-6), 3),
        }
    return out


def render_overlays(labels, vols, save_prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    t2f, t1c = vols[3], vols[1]
    cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.29, 0.25, 0.85), (0.16, 0.93, 0.35, 0.6), (1.0, 0.87, 0.2, 0.85)])

    nz = np.array(np.nonzero(labels > 0))
    if nz.size == 0:
        center = [s // 2 for s in labels.shape]
    else:
        center = [int(c) for c in nz.mean(axis=1)]

    def norm(img):
        lo, hi = np.percentile(img, [1, 99.5])
        return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    paths = {}
    for view, axis, base in [('axial', 2, t2f), ('coronal', 1, t1c), ('sagittal', 0, t1c)]:
        n_slices = 9 if view == 'axial' else 5
        length = labels.shape[axis]
        span = min(70, length)
        lo_lim = max(0, center[axis] - span // 2)
        hi_lim = min(length, lo_lim + span)
        step = max(1, (hi_lim - lo_lim) // n_slices)
        pos = sorted(set(np.clip(np.arange(lo_lim, hi_lim, step)[:n_slices], 0, length - 1)))
        if center[axis] not in pos:
            pos[len(pos) // 2] = center[axis]
            pos = sorted(set(pos))

        fig, axes = plt.subplots(1, len(pos), figsize=(2.1 * len(pos), 2.3))
        if len(pos) == 1:
            axes = [axes]
        for ax, p in zip(axes, pos):
            if axis == 2:
                b, lab = norm(base[:, :, p]), labels[:, :, p]
                ax.imshow(b, cmap='gray')
                ax.imshow(lab, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
            elif axis == 1:
                b, lab = norm(base[:, p, ::-1]), labels[:, p, ::-1]
                ax.imshow(b, cmap='gray')
                ax.imshow(lab, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
            else:
                b, lab = norm(base[p, ::-1, :]), labels[p, ::-1, :]
                ax.imshow(b, cmap='gray')
                ax.imshow(lab, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
            ax.axis('off')
        plt.tight_layout(pad=0.3)
        path = f'{save_prefix}-{view}.png'
        fig.savefig(path, dpi=110, bbox_inches='tight', facecolor='#ffffff')
        plt.close(fig)
        paths[view] = path
    return paths


def run_pipeline(nii_objects, case_name, save_prefix):
    import os
    low_mem = os.environ.get('RACORE_LOW_MEM', '') == '1'

    vols, zooms = [], None
    for obj in nii_objects:
        d, z = load_modal(obj)
        vols.append(d)
        if zooms is None:
            zooms = z

    if low_mem:
        vols = [ndimage.zoom(v, 0.5, order=1) for v in vols]
        zooms = [z * 2 for z in zooms]
        import gc
        gc.collect()

    labels, brain = segment_case(vols, zooms)
    features = extract_features(labels, vols, zooms, case_name)
    overlay_paths = render_overlays(labels, vols, save_prefix)
    n_tumor = int((labels > 0).sum())
    return features, overlay_paths, n_tumor, brain.sum()