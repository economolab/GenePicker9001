#%%
 
import traceback
from pathlib import Path
 
import cv2
import matplotlib.pyplot as plt
import nd2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.segmentation import morphological_chan_vese, watershed
from skimage.transform import downscale_local_mean
from tqdm import tqdm
 
# ---------------------------------------------------------------- config
INPUT_DIR = Path(r'F:\2026-11-03_contra_WGA_HCR_r1')          # folder containing the .nd2 files
KEYWORD = 'overview'                  # only files whose name contains this (case-insensitive)
PATHS = sorted(p for p in INPUT_DIR.glob('*.nd2') if KEYWORD.lower() in p.name.lower())
 
DS = 4
PLOT_STEPS = False    # show intermediate figures interactively (one file at a time)
 
 
# ---------------------------------------------------------------- helpers
def largest(m):
    lab = measure.label(m)
    if lab.max() > 1:
        m = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return m
 
 
def show(img, msk, title, color="r", pct=(1, 99)):
    # US Letter size is 8.5 by 11 inches
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.imshow(img, cmap="gray",
              vmin=np.percentile(img, pct[0]), vmax=np.percentile(img, pct[1]))
    ax.contour(msk, levels=[0.5], colors=color, linewidths=1.5)
    ax.set_title(title)
    ax.axis("off")
    plt.show()
 
 
def pca_angle(mask):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = evecs[:, np.argmax(evals)]
    return (np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90
 
 
def rotate_by(arr, angle, flags=cv2.INTER_NEAREST):
    H, W = arr.shape[:2]
    cx, cy = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nW / 2 - cx
    M[1, 2] += nH / 2 - cy
    return cv2.warpAffine(arr, M, (nW, nH), flags=flags)
 
 
# ---------------------------------------------------------------- pipeline stages
def run_chan_vese(image, max_iter=1000, rel_tol=0.05, abs_tol=50, patience=10, desc=None):
    ls, prev_changed, quiet = "checkerboard", None, 0
    with tqdm(total=max_iter, desc=desc, leave=False) as pbar:
        for i in range(max_iter):
            new = morphological_chan_vese(image, num_iter=1, init_level_set=ls, smoothing=1)
            info = {"area_frac": f"{new.mean():.4f}"}
            if i > 0:
                changed = int((new != ls).sum())
                if prev_changed is not None:
                    thresh = max(abs_tol, rel_tol * max(changed, prev_changed))
                    quiet = quiet + 1 if abs(changed - prev_changed) <= thresh else 0
                    info.update(changed=changed, quiet=f"{quiet}/{patience}")
                prev_changed = changed
            ls = new
            pbar.set_postfix(info)
            pbar.update(1)
            if quiet >= patience:
                break
    return ls.astype(bool)
 
 
def cleanup(mask, image):
    m = mask if image[mask].mean() > image[~mask].mean() else ~mask
    m = largest(m)
    max_size = np.sum(m) * 0.01
    m = morphology.remove_small_holes(m, max_size=max_size)  # keep large holes
    m = filters.rank.modal(m.astype(np.uint8), morphology.disk(2)).astype(bool)
    return largest(m)
 
 
def remove_tendrils(m, dt):
    r = 0.01 * dt.max()  # tune once; should transfer between images
 
    core = largest(dt > r)
    opened = ndi.distance_transform_edt(~core) <= r  # used only as a selector
 
    resid = m & ~opened
    dist = ndi.distance_transform_edt(~opened)
    lab_r = measure.label(resid)
    if lab_r.max() > 0:
        depth = ndi.maximum(dist, lab_r, index=np.arange(1, lab_r.max() + 1))
        keep = np.flatnonzero(np.atleast_1d(depth) <= 0.6 * r) + 1  # shallow = corner slivers
        m_clean = opened | np.isin(lab_r, keep)
    else:
        m_clean = opened
    return largest(m_clean)
 
 
def watershed_refine(image, m_clean, dt, n_rounds=5):
    W = 0.01 * dt.max()
    grad = filters.sobel(filters.gaussian(image, 1, preserve_range=True))
 
    refined = m_clean.copy()
    for k in range(n_rounds):
        din = ndi.distance_transform_edt(refined)
        dout = ndi.distance_transform_edt(~refined)
 
        markers = np.zeros(refined.shape, np.int32)
        markers[dout > W] = 1  # definitely background
        markers[din > W] = 2   # definitely object
 
        new = largest(watershed(grad, markers) == 2)
        print(f"  round {k}: {int((new != refined).sum())} px changed")
        refined = new
    return refined
 
 
def crop_to_mask(image_rotated, mask_rotated, pad):
    ys, xs = np.nonzero(mask_rotated)
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, mask_rotated.shape[0])
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, mask_rotated.shape[1])
    return image_rotated[y0:y1, x0:x1], mask_rotated[y0:y1, x0:x1], (y0, y1, x0, x1)
 
 
# ---------------------------------------------------------------- per-file driver
def process_file(path):
    stem = path.stem
 
    def step(img, msk, title, **kw):
        if PLOT_STEPS:
            show(img, msk, f"{stem}: {title}", **kw)
 
    # load
    full = np.fliplr(nd2.imread(str(path))).astype(np.float32)
    image = downscale_local_mean(full, DS)
    del full
 
    # chan-vese
    mask = run_chan_vese(image, desc=stem)
    step(image, mask, "1. Chan-Vese")
 
    # cleanup at coarse resolution
    m = cleanup(mask, image)
    step(image, m, "2. cleanup")
 
    # remove tendrils / hanging chunk, keep sharp corners
    dt = ndi.distance_transform_edt(m)
    m_clean = remove_tendrils(m, dt)
    step(image, m_clean, "3. tendrils removed")
 
    # watershed refinement
    refined = watershed_refine(image, m_clean, dt)
    step(image, refined, "4. watershed refined", color="w")
 
    # rotate to principal axis
    angle = pca_angle(refined)
    mask_rotated = rotate_by(refined.astype(np.uint8), angle)
    image_rotated = rotate_by(image, angle, flags=cv2.INTER_LINEAR)
    step(image_rotated, mask_rotated, f"5. rotated {angle:.1f} deg", color="w")
 
    # crop
    pad = int(0.10 * dt.max())  # px, coarse-image coordinates
    image_crop, mask_crop, box = crop_to_mask(image_rotated, mask_rotated, pad)
    step(image_crop, mask_crop, "6. cropped", color="w", pct=(10, 99))
 
    return {"image": image_crop, "mask": mask_crop, "angle": angle, "crop_box": box}
 
 
#%% run all files
print(f"{len(PATHS)} file(s) found")
 
results, failed = {}, {}
for p in tqdm(PATHS, desc="files"):
    try:
        results[p.stem] = process_file(p)
    except Exception as e:  # one bad file shouldn't kill the whole batch
        traceback.print_exc()
        failed[p.stem] = e
 
print(f"done: {len(results)} ok, {len(failed)} failed")
 
 
#%% final figure: US Letter, two images per row
ncols = 2
nrows = -(-len(results) // ncols)  # ceil
fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 11), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, (name, r) in zip(axes.ravel(), results.items()):
    img = r["image"]
    ax.imshow(img, cmap="gray",
              vmin=np.percentile(img, 10), vmax=np.percentile(img, 99))
    ax.contour(r["mask"], levels=[0.5], colors="w", linewidths=1)
    ax.set_title(name, fontsize=8)
fig.tight_layout()
plt.show()
plt.savefig("overviews.png", dpi=1000)
