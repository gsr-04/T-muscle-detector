"""Step 1 of the pipeline: cut the person out of the background.

Fuses a dedicated matting model (MediaPipe selfie segmenter) with the pose
landmarker's segmentation head, refines the boundary with GrabCut seeded by the
confident core / background, then outputs a soft alpha and an RGBA cutout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

# copy of src/extract.py: its helpers live in src/, so put that on the path
import sys as _sys
for _p in (Path(__file__).resolve().parents[1] / "src",
           Path(__file__).resolve().parents[1] / "tools"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from pose_mp import detect, to_mask
from torso_seg import (DATA, MODELS, OUT, ascii_path, clean_mask, load_bgr,
                       save_bgr)


def segmenter_mask(bgr: np.ndarray, model: Path) -> np.ndarray:
    """Person confidence map in 0..1 at the image resolution."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ascii_path(model))),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(img)
    masks = [np.asarray(m.numpy_view()) for m in res.confidence_masks]
    if len(masks) == 1:
        conf = masks[0]
    else:
        conf = np.max(masks[1:], axis=0)      # category 0 is background
    if conf.ndim == 3:
        conf = conf[..., 0]
    if conf.shape[:2] != bgr.shape[:2]:
        conf = cv2.resize(conf.astype(np.float32), (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return np.clip(conf.astype(np.float32), 0.0, 1.0)


def fuse(conf: np.ndarray, pose_fg: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Blend the matting confidence with the pose mask and a skin prior."""
    p = (pose_fg > 0).astype(np.float32)
    p = cv2.GaussianBlur(p, (0, 0), 3.0)
    # the skin rule fires on beige walls and wood too, so it only counts where
    # there is already evidence of a person nearby
    support = ((pose_fg > 0) | (conf > 0.5)).astype(np.uint8) * 255
    k = max(3, int(0.05 * min(conf.shape[:2])) | 1)
    support = cv2.dilate(support, np.ones((k, k), np.uint8))
    s = ((skin > 0) & (support > 0)).astype(np.float32)
    out = np.maximum(conf, 0.85 * p)
    out = np.maximum(out, 0.55 * s)
    return np.clip(out, 0.0, 1.0)


def refine(bgr: np.ndarray, conf: np.ndarray, pose_fg: np.ndarray | None = None,
           iters: int = 4) -> np.ndarray:
    """GrabCut with a trimap from the confidence map, then largest component."""
    h, w = bgr.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[conf > 0.75] = cv2.GC_FGD
    gc[(conf > 0.35) & (conf <= 0.75)] = cv2.GC_PR_FGD
    gc[conf < 0.03] = cv2.GC_BGD
    # a thin border counts as background only where the model is not confident
    # either way -- the subject is often cropped by the frame itself
    r = max(2, int(round(min(h, w) * 0.01)))
    body = (pose_fg > 0) if pose_fg is not None else np.zeros((h, w), bool)
    border = np.zeros((h, w), bool)
    border[:r] = border[-r:] = True
    border[:, :r] = border[:, -r:] = True
    gc[border & ~body & (conf < 0.5)] = cv2.GC_BGD
    gc[border & body] = cv2.GC_PR_FGD
    gc[body] = np.maximum(gc[body], cv2.GC_PR_FGD)
    # where the subject really is cut by the frame, pass the outermost row or
    # column straight through (the blurred pose mask is too weak there)
    for sl, outer in ((np.s_[0, :], 0), (np.s_[h - 1, :], 0),
                      (np.s_[:, 0], 1), (np.s_[:, w - 1], 1)):
        line_body = body[sl]
        if line_body.any():
            gc[sl] = np.where(line_body, cv2.GC_FGD, gc[sl]).astype(np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        m = (conf > 0.5).astype(np.uint8)
    return clean_mask(m * 255, fill_holes=False) // 255


def soften(mask01: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """Feather the binary mask into an 8-bit alpha."""
    pad = int(3 * sigma) + 1
    padded = cv2.copyMakeBorder(mask01.astype(np.float32), pad, pad, pad, pad,
                                cv2.BORDER_REPLICATE)
    a = cv2.GaussianBlur(padded, (0, 0), sigma)[pad:-pad, pad:-pad]
    a = np.clip((a - 0.35) / 0.30, 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def align_alpha(bgr: np.ndarray, alpha01: np.ndarray, radius: int,
                eps: float = 1e-4) -> np.ndarray:
    """Guided filter the soft mask with the image as guide so that the boundary
    snaps onto the real silhouette instead of staying where GrabCut left it."""
    if not hasattr(cv2, "ximgproc"):
        return alpha01
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    a = cv2.ximgproc.guidedFilter(guide, alpha01.astype(np.float32),
                                  max(2, int(radius)), eps)
    return np.clip(a, 0.0, 1.0)


def cutout(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def preview(bgr: np.ndarray, mask01: np.ndarray, square: int = 16) -> np.ndarray:
    """Checkerboard preview with the extracted silhouette outlined."""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // square) + (xx // square)) % 2 * 40 + 190).astype(np.uint8)
    vis = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    keep = mask01 > 0
    vis[keep] = bgr[keep]
    cnts, _ = cv2.findContours((mask01 * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def colour_tighten(bgr: np.ndarray, mask01: np.ndarray,
                   band: int | None = None) -> np.ndarray:
    """Drop boundary pixels that look more like the background than the subject.

    The background model comes from the four corners (almost always background in
    these photos); the foreground model from the eroded mask interior.  Only the
    boundary band is re-decided, so a dark garment is never eaten away.
    """
    h, w = mask01.shape[:2]
    band = band or max(2, int(0.02 * min(h, w)))
    m = (mask01 > 0.5).astype(np.uint8)
    if m.sum() < 100:
        return m
    c = max(8, min(h, w) // 25)
    corners = np.concatenate([bgr[:c, :c].reshape(-1, 3), bgr[:c, -c:].reshape(-1, 3),
                              bgr[-c:, :c].reshape(-1, 3), bgr[-c:, -c:].reshape(-1, 3)])
    bg_mu = corners.mean(0).astype(np.float32)
    bg_sd = np.maximum(corners.std(0).astype(np.float32), 6.0)
    inner = cv2.erode(m * 255, np.ones((2 * band + 1,) * 2, np.uint8)) > 0
    if inner.sum() < 50:
        return m
    fg_mu = bgr[inner].mean(0).astype(np.float32)
    fg_sd = np.maximum(bgr[inner].std(0).astype(np.float32), 6.0)
    d_bg = np.abs(bgr.astype(np.float32) - bg_mu) / bg_sd
    d_fg = np.abs(bgr.astype(np.float32) - fg_mu) / fg_sd
    vote = (d_fg.sum(2) < d_bg.sum(2))
    core = cv2.erode(m * 255, np.ones((2 * band + 1,) * 2, np.uint8)) > 0
    ring = (m > 0) & ~core
    out = m.copy()
    out[ring & ~vote] = 0
    keep = (out > 0) | core
    lab, n = ndi.label(keep)
    if n:
        main = 1 + int(np.argmax(ndi.sum(keep, lab, range(1, n + 1))))
        out = (lab == main).astype(np.uint8)
    return out


def colour_models(bgr: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    """Background model from border pixels outside the mask, foreground model
    from a well eroded core, so a large spill cannot contaminate the subject."""
    h, w = m.shape[:2]
    outside = m == 0
    r = max(3, int(0.03 * min(h, w)))
    ring = np.zeros((h, w), bool)
    ring[:r] = ring[-r:] = True
    ring[:, :r] = ring[:, -r:] = True
    bg_px = bgr[ring & outside]
    if bg_px.shape[0] < 50:
        c = max(6, min(h, w) // 25)
        bg_px = np.concatenate([bgr[:c, :c].reshape(-1, 3), bgr[:c, -c:].reshape(-1, 3),
                                bgr[-c:, :c].reshape(-1, 3),
                                bgr[-c:, -c:].reshape(-1, 3)])
    rr = max(2, int(0.05 * min(h, w)))
    core = cv2.erode((m * 255).astype(np.uint8), np.ones((2 * rr + 1,) * 2, np.uint8)) > 0
    while core.sum() < 50 and rr > 1:
        rr //= 2
        core = cv2.erode((m * 255).astype(np.uint8),
                         np.ones((2 * rr + 1,) * 2, np.uint8)) > 0
    fg_px = bgr[core] if core.sum() >= 50 else bgr[m > 0]
    bg_mu, bg_sd = bg_px.mean(0).astype(np.float32), np.maximum(
        bg_px.std(0).astype(np.float32), 6.0)
    fg_mu, fg_sd = fg_px.mean(0).astype(np.float32), np.maximum(
        fg_px.std(0).astype(np.float32), 6.0)
    return bg_mu, bg_sd, fg_mu, fg_sd


def background_like(bgr: np.ndarray, models) -> np.ndarray:
    bg_mu, bg_sd, fg_mu, fg_sd = models
    a = bgr.astype(np.float32)
    d_bg = (np.abs(a - bg_mu) / bg_sd).sum(2)
    d_fg = (np.abs(a - fg_mu) / fg_sd).sum(2)
    return d_bg < d_fg


def fill_holes_colour(bgr: np.ndarray, m: np.ndarray, models,
                      tiny: float = 0.004) -> np.ndarray:
    """Fill only holes that do not look like the background.

    The gap between the arm and the torso is often an enclosed hole once the arm
    touches the body above and below; blindly filling holes turns it into
    foreground, which is exactly what used to happen.
    """
    bg_like = background_like(bgr, models)
    filled = ndi.binary_fill_holes(m > 0)
    holes = filled & ~(m > 0)
    if not holes.any():
        return (m > 0).astype(np.uint8)
    lab, n = ndi.label(holes)
    out = (m > 0).copy()
    for i in range(1, n + 1):
        sel = lab == i
        size = int(sel.sum())
        if size <= 0:
            continue
        frac_bg = float(bg_like[sel].mean())
        if size < tiny * m.size and frac_bg < 0.6:
            out[sel] = True                 # tiny speck, not background colour
        elif frac_bg < 0.5:
            out[sel] = True
        # else: it looks like the background (e.g. the arm/torso gap) -> leave it
    return out.astype(np.uint8)


def fill_tiny_holes(m: np.ndarray, frac: float = 0.001) -> np.ndarray:
    """Fill only specks: a genuine gap (arm/torso, between the legs) is bigger."""
    filled = ndi.binary_fill_holes(m > 0)
    holes = filled & ~(m > 0)
    if not holes.any():
        return (m > 0).astype(np.uint8)
    lab, n = ndi.label(holes)
    out = (m > 0).copy()
    for i in range(1, n + 1):
        sel = lab == i
        if int(sel.sum()) < frac * m.size:
            out[sel] = True
    return out.astype(np.uint8)


def prune_by_row_background(bgr: np.ndarray, m: np.ndarray, pose_fg: np.ndarray,
                            tol: float = 16.0, chroma_tol: float = 0.07,
                            min_px: float = 0.0005) -> np.ndarray:
    """Delete mask areas that the pose head calls background *and* whose colour
    matches the background of that same image row.

    This is what separates the gap between arm and torso (background colour, no
    person per the pose head) from the arm itself (no person per the pose head,
    but clearly not background colour).

    A deep shadow under the arm is far from the background's absolute colour but
    keeps the background's chromaticity, so the test uses normalised rgb.
    """
    h, w = m.shape[:2]
    if not pose_fg.any():
        return m
    bg = pose_fg == 0
    out = (m > 0).copy()
    cand = (m > 0) & bg
    if not cand.any():
        return m
    for y in range(h):
        xs = np.flatnonzero(cand[y])
        if xs.size == 0:
            continue
        row_bg = np.flatnonzero(bg[y])
        if row_bg.size < 5:
            continue
        med = np.median(bgr[y, row_bg], axis=0)
        px = bgr[y, xs].astype(np.float32)
        diff_abs = np.abs(px - med).mean(axis=1)
        n_px = px / np.maximum(px.sum(axis=1, keepdims=True), 1e-6)
        n_bg = med / max(float(med.sum()), 1e-6)
        diff_chr = np.abs(n_px - n_bg).sum(axis=1)
        drop = xs[(diff_abs <= tol) | (diff_chr <= chroma_tol)]
        out[y, drop] = False
    # keep substantial components only
    lab, n = ndi.label(out)
    if n:
        sizes = ndi.sum(out, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px * m.size]
        if keep:
            out = np.isin(lab, keep)
    return out.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--segmenter", type=Path,
                    default=MODELS / "selfie_segmenter.tflite")
    ap.add_argument("--pose", type=Path, default=MODELS / "pose_landmarker_full.task")
    args = ap.parse_args()

    from torso_seg import skin_mask

    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("*.png")):
        bgr = load_bgr(path)
        h, w = bgr.shape[:2]
        conf = segmenter_mask(bgr, args.segmenter)
        pose_fg = np.zeros((h, w), np.uint8)
        try:
            res = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.pose)
            if res.pose_landmarks:
                pose_fg = to_mask(res, (h, w))
        except Exception as exc:                      # noqa: BLE001
            print(f"   pose mask unavailable: {exc}")
        sk = skin_mask(bgr)
        fused = fuse(conf, pose_fg, sk)
        m = refine(bgr, fused, pose_fg)
        # a subject cut off by the frame keeps its pixels on the outermost line
        if pose_fg.any():
            body = pose_fg > 0
            if body[0].any():
                m[0, body[0]] = 1
            if body[-1].any():
                m[-1, body[-1]] = 1
            if body[:, 0].any():
                m[body[:, 0], 0] = 1
            if body[:, -1].any():
                m[body[:, -1], -1] = 1
        soft = soften(m, sigma=max(1.0, 0.004 * min(h, w))) / 255.0
        soft = align_alpha(bgr, soft, radius=int(0.02 * min(h, w)))
        # the guided filter may add specks and creep outwards; keep it honest
        core = cv2.dilate((m * 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        m = ((soft > 0.5) & core).astype(np.uint8)
        lab2, n2 = ndi.label(m > 0)
        if n2:
            sizes2 = ndi.sum(m > 0, lab2, range(1, n2 + 1))
            keep = [i + 1 for i, s in enumerate(sizes2) if s >= 0.001 * m.size]
            m = np.isin(lab2, keep).astype(np.uint8)
        m = colour_tighten(bgr, m)
        # enclosed background between the arm and the torso looks like a hole to
        # a flood fill, so holes are decided by colour instead of blindly filled
        models = colour_models(bgr, m)
        m = fill_holes_colour(bgr, m, models)
        # and any spill left over in the pose head's background region is dropped
        # when its colour matches the background of that row
        m = prune_by_row_background(bgr, m, pose_fg)
        m = fill_tiny_holes(m)
        lab4, n4 = ndi.label(m > 0)
        if n4:
            sizes4 = ndi.sum(m > 0, lab4, range(1, n4 + 1))
            m = (lab4 == (1 + int(np.argmax(sizes4)))).astype(np.uint8)
        skirt = cv2.dilate(m * 255, np.ones((3, 3), np.uint8)) > 0
        alpha = (np.clip(soft, 0.0, 1.0) * skirt * 255).astype(np.uint8)

        save_bgr(args.out / f"alpha_{path.stem}.png", alpha)
        save_bgr(args.out / f"cutout_{path.stem}.png", cutout(bgr, alpha))
        save_bgr(args.out / f"preview_{path.stem}.png", preview(bgr, m))
        save_bgr(args.out / f"fgmask_{path.stem}.png", m * 255)

        lab, n = ndi.label(m > 0)
        sizes = ndi.sum(m > 0, lab, range(1, n + 1)) if n else np.array([0])
        touch = (m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any())
        ptouch = (pose_fg[0].any(), pose_fg[-1].any(), pose_fg[:, 0].any(),
                  pose_fg[:, -1].any()) if pose_fg.any() else (0, 0, 0, 0)
        inter = float(((m > 0) & (pose_fg > 0)).sum())
        union = float(((m > 0) | (pose_fg > 0)).sum())
        iou = inter / union if union else 0.0
        extra = float(((m > 0) & (pose_fg == 0)).sum()) / max(1, float((m > 0).sum()))
        print(f"[{path.stem}] {w}x{h} person={100.0 * m.mean():5.1f}% "
              f"components={n} largest={100.0 * float(sizes.max()) / m.size:5.1f}% "
              f"IoU(pose)={iou:.2f} extra={100.0 * extra:4.1f}% "
              f"touches={tuple(int(t) for t in touch)} pose_touch="
              f"{tuple(int(t) for t in ptouch)}")


if __name__ == "__main__":
    main()
