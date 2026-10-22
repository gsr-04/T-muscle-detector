"""Snap approximate boundaries onto the creases and steps actually visible."""

from __future__ import annotations

import cv2
import numpy as np


def valley_map(bgr: np.ndarray, k: int = 41, blur: float = 1.6) -> np.ndarray:
    """Dark creases (muscle boundaries, folds) as a 0..1 response."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), blur)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
    bh = cv2.GaussianBlur(bh, (0, 0), 2.0)
    return cv2.normalize(bh, None, 0.0, 1.0, cv2.NORM_MINMAX)


def ridge_map(bgr: np.ndarray, k: int = 31, blur: float = 1.6) -> np.ndarray:
    """Bright ridges (bone highlights such as the clavicle)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), blur)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    wh = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker).astype(np.float32)
    wh = cv2.GaussianBlur(wh, (0, 0), 2.0)
    return cv2.normalize(wh, None, 0.0, 1.0, cv2.NORM_MINMAX)


def step_map(bgr: np.ndarray, blur: float = 2.0) -> np.ndarray:
    """Horizontal intensity steps (garment edges) as a 0..1 response."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), blur)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5))
    gy = cv2.GaussianBlur(gy.astype(np.float32), (0, 0), 2.0)
    return cv2.normalize(gy, None, 0.0, 1.0, cv2.NORM_MINMAX)


def _median(v: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    return np.array([np.median(v[max(0, i - pad): i + pad + 1])
                     for i in range(len(v))])


def _close_gaps(raw: np.ndarray, radius: int, smooth: int) -> np.ndarray:
    """Points where no response was found keep a value interpolated from the
    points that did snap, so a partially visible crease still bends the curve."""
    hit = ~np.isnan(raw)
    if hit.sum() == 0:
        return np.zeros(len(raw))
    if hit.sum() == 1:
        return np.full(len(raw), float(np.clip(raw[hit][0], -radius, radius)))
    offs = np.interp(np.arange(len(raw)), np.flatnonzero(hit),
                     np.clip(raw[hit], -radius, radius))
    n = len(offs)
    k = min(smooth, n)
    k = max(3, k | 1)
    pad = k // 2
    return np.array([np.median(offs[max(0, i - pad): i + pad + 1]) for i in range(n)])


def snap_vertical(R: np.ndarray, pts: np.ndarray, radius: int, smooth: int = 9,
                  w_min: float = 0.12, contrast: float = 1.5) -> np.ndarray:
    """Move each point up/down to the strongest response within ``radius``."""
    h, w = R.shape
    out = np.array(pts, float)
    raw = np.full(len(pts), np.nan)
    for i, (x, y) in enumerate(pts):
        xi = int(round(x))
        if not (0 <= xi < w):
            continue
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        col = R[y0:y1, xi]
        if col.size == 0:
            continue
        top, med = float(col.max()), float(np.median(col))
        if top < w_min or top < contrast * med:
            continue
        raw[i] = (y0 + int(np.argmax(col))) - y
    offs = _close_gaps(raw, radius, min(smooth, max(3, len(pts) // 3 * 2 + 1)))
    out[:, 1] = out[:, 1] + offs
    return out


def snap_horizontal(R: np.ndarray, pts: np.ndarray, radius: int, smooth: int = 9,
                    w_min: float = 0.12, contrast: float = 1.5) -> np.ndarray:
    h, w = R.shape
    out = np.array(pts, float)
    raw = np.full(len(pts), np.nan)
    for i, (x, y) in enumerate(pts):
        yi = int(round(y))
        if not (0 <= yi < h):
            continue
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        row = R[yi, x0:x1]
        if row.size == 0:
            continue
        top, med = float(row.max()), float(np.median(row))
        if top < w_min or top < contrast * med:
            continue
        raw[i] = (x0 + int(np.argmax(row))) - x
    offs = _close_gaps(raw, radius, min(smooth, max(3, len(pts) // 3 * 2 + 1)))
    out[:, 0] = out[:, 0] + offs
    return out
