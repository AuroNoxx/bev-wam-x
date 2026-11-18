"""Refine the width and top-edge alignment of two detected vehicle boxes at native scale."""
import cv2
import numpy as np
from .scale_track import image


def match_pair(path, anchor, earlier_path, candidates):
    current = image(path)
    earlier = image(earlier_path)
    a = np.asarray(anchor)
    size = a[2:] - a[:2]
    pivot = np.array([(a[0] + a[2])/2, a[1]])
    start = np.maximum(0, np.floor(a[:2]-20).astype(int))
    end = np.minimum(current.shape[::-1], np.ceil(a[2:]+20).astype(int))
    crop = current[start[1]:end[1],start[0]:end[0]]
    mask = np.zeros_like(crop)
    lo = np.rint(a[:2]+.12*size-start).astype(int)
    hi = np.rint(a[2:]-.12*size-start).astype(int)
    cv2.rectangle(mask, tuple(lo), tuple(hi), 255, -1)
    points = cv2.goodFeaturesToTrack(crop, 180, .01, 4, mask=mask)
    if points is None or len(points) < 6:
        return None
    best = None
    for cand in candidates:
        b = np.asarray(cand['box'])
        extent = b[2:] - b[:2]
        ratio = extent[0]/size[0]
        other = np.array([(b[0]+b[2])/2, b[1]])
        if not .25 <= ratio <= 4 or max(abs(other-pivot)/np.maximum(size, extent)) > .85:
            continue
        shift = other + ratio*(start-pivot)
        seed = np.array([[ratio,0,shift[0]],[0,ratio,shift[1]]])
        # Output pixels have the original current-image pixel spacing.
        aligned = cv2.warpAffine(earlier, seed, crop.shape[::-1],
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        lk = dict(winSize=(21,21), maxLevel=2)
        dest, ok, error = cv2.calcOpticalFlowPyrLK(crop, aligned, points, None, **lk)
        back, rev, _ = cv2.calcOpticalFlowPyrLK(aligned, crop, dest, None, **lk)
        prev = dest[:,0] * ratio + shift
        valid = ((ok & rev).ravel().astype(bool) & (error.ravel() < 20)
                 & (np.linalg.norm((back-points)[:,0],axis=1) < 1.5)
                 & (prev >= b[:2]).all(axis=1) & (prev <= b[2:]).all(axis=1))
        if valid.sum() < 6:
            continue
        src = points[valid,0] + start
        transform, keep = cv2.estimateAffinePartial2D(src, prev[valid], method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.5)
        if transform is None:
            continue
        inside = keep.ravel().astype(bool)
        count = int(inside.sum())
        spread = np.ptp(src[inside],axis=0)/size if count else [0.,0.]
        scale = np.linalg.norm(transform[:,0])
        if count < 6 or spread[0] < .18 or spread[1] < .08 or not .65 <= scale/ratio <= 1.55:
            continue
        rank = count * min(1., spread[0]) * min(1., spread[1])
        if best is None or rank > best[0]:
            best = (rank,cand['box'],transform,count)
    return best
