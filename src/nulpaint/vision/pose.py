"""OpenPose skeleton extraction (DWPose via ONNX Runtime) for ControlNet repose.

Pipeline: YOLOX person detection -> RTMPose (dw-ll_ucoco_384) keypoints -> render
the OpenPose-18 body skeleton (the colored stick figure control_v11p_sd15_openpose
expects). No torch — onnxruntime only, models in models/pose/. Lazily loaded.
"""
from __future__ import annotations
import os
import numpy as np
import cv2

from ..config import _KRITA_ROOT

POSE_DIR = os.environ.get("NULPAINT_POSE_DIR", os.path.join(_KRITA_ROOT, "models/pose"))
_DET = os.path.join(POSE_DIR, "yolox_l.onnx")
_POSE = os.path.join(POSE_DIR, "dw-ll_ucoco_384.onnx")

_sessions: dict = {}


def _sess(path: str):
    import onnxruntime as ort
    if path not in _sessions:
        _sessions[path] = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _sessions[path]


# --- YOLOX person detection -------------------------------------------------
def _detect_people(bgr, score_thr=0.3, nms_thr=0.45):
    H, W = bgr.shape[:2]
    inp = np.ones((640, 640, 3), np.float32) * 114.0
    r = min(640 / H, 640 / W)
    rh, rw = int(H * r), int(W * r)
    inp[:rh, :rw] = cv2.resize(bgr, (rw, rh)).astype(np.float32)
    blob = inp.transpose(2, 0, 1)[None]  # NCHW, BGR, raw 0-255
    s = _sess(_DET)
    out = s.run(None, {s.get_inputs()[0].name: blob})[0][0]  # (N, 85)

    # decode with grid/strides
    grids, strides = [], []
    for stride in (8, 16, 32):
        hg, wg = 640 // stride, 640 // stride
        xv, yv = np.meshgrid(np.arange(wg), np.arange(hg))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((hg * wg, 1), stride))
    grids = np.concatenate(grids, 0)
    strides = np.concatenate(strides, 0)
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides

    scores = out[:, 4:5] * out[:, 5:]          # obj * cls
    person = scores[:, 0]                        # class 0 = person
    boxes = out[:, :4].copy()
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= r

    keep = person > score_thr
    boxes_xyxy, person = boxes_xyxy[keep], person[keep]
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4))
    idx = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), person.tolist(), score_thr, nms_thr)
    idx = np.array(idx).flatten()
    return boxes_xyxy[idx]


# --- RTMPose keypoints ------------------------------------------------------
def _keypoints(bgr, box):
    s = _sess(_POSE)
    in_w, in_h = 288, 384
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * 1.25, (y1 - y0) * 1.25
    scale = max(w / in_w, h / in_h)
    src_w, src_h = in_w * scale, in_h * scale
    src = np.array([[cx, cy], [cx, cy - src_h / 2], [cx - src_w / 2, cy]], np.float32)
    dst = np.array([[in_w / 2, in_h / 2], [in_w / 2, 0], [0, in_h / 2]], np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(bgr, M, (in_w, in_h), flags=cv2.INTER_LINEAR)

    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    blob = ((crop.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    sx, sy = s.run(None, {s.get_inputs()[0].name: blob})  # simcc x/y
    sx, sy = sx[0], sy[0]                                   # (133, W*2),(133,H*2)
    px, py = sx.argmax(1), sy.argmax(1)
    score = np.minimum(sx.max(1), sy.max(1))
    kp = np.stack([px / 2.0, py / 2.0], 1)                  # model space (simcc split=2)
    Minv = cv2.invertAffineTransform(M)
    kp = kp @ Minv[:, :2].T + Minv[:, 2]
    return kp, score


# --- OpenPose-18 rendering --------------------------------------------------
# COCO-17 (RTMPose body order) -> OpenPose-18, neck = midpoint of shoulders.
_COCO2OP = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
          (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
          (0, 15), (15, 17)]
_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
           [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
           [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
           [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def _to_openpose18(kp, score, thr=0.3):
    op = np.zeros((18, 2)); ops = np.zeros(18)
    for i, c in enumerate(_COCO2OP):
        if c is None:
            continue
        op[i], ops[i] = kp[c], score[c]
    if ops[2] > thr and ops[5] > thr:            # neck = shoulder midpoint
        op[1] = (op[2] + op[5]) / 2; ops[1] = min(ops[2], ops[5])
    return op, ops


def pose_skeleton(bgr: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """BGR image -> OpenPose skeleton (BGR, black background) for ControlNet."""
    H, W = bgr.shape[:2]
    canvas = np.zeros((H, W, 3), np.uint8)
    for box in _detect_people(bgr):
        kp, score = _keypoints(bgr, box)
        op, ops = _to_openpose18(kp, score, thr)
        for ci, (a, b) in enumerate(_LIMBS):
            if ops[a] > thr and ops[b] > thr:
                pa, pb = op[a].astype(int), op[b].astype(int)
                cv2.line(canvas, tuple(pa), tuple(pb), _COLORS[ci][::-1], 4, cv2.LINE_AA)
        for i in range(18):
            if ops[i] > thr:
                cv2.circle(canvas, tuple(op[i].astype(int)), 4, _COLORS[i][::-1], -1, cv2.LINE_AA)
    return canvas
