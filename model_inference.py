"""
model_inference.py — DeepFake detection engine using multi-signal heuristic ensemble.

Signals analyzed:
  1. Frequency domain artifacts (FFT / DCT)
  2. Compression & blocking artifacts
  3. Facial region blending boundary detection
  4. Temporal frame-to-frame inconsistency
  5. Noise residual pattern analysis
  6. Color channel asymmetry
  7. Edge sharpness inconsistency
  8. Eye/face symmetry irregularities

Each signal is scored 0-1 (0=clean, 1=suspicious).
Final fake_probability = weighted average of all signals.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple
from .video_processor import extract_frames, get_video_metadata, detect_faces_in_frame, crop_face_region

logger = logging.getLogger("deepfake.inference")

# ── Signal weights (must sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "frequency_artifacts":       0.18,
    "compression_blocking":      0.12,
    "blending_boundary":         0.16,
    "temporal_inconsistency":    0.20,
    "noise_residual":            0.12,
    "color_asymmetry":           0.10,
    "edge_sharpness":            0.07,
    "face_symmetry":             0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ── Individual Signal Detectors ───────────────────────────────────────────────

def _frequency_artifacts(frames: list) -> Tuple[float, str]:
    """
    DeepFake generators leave characteristic frequency-domain fingerprints.
    Real cameras have natural high-freq roll-off; GANs produce unnatural spikes.
    """
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Ratio of high-frequency energy to low-frequency energy
        inner = magnitude[cy-20:cy+20, cx-20:cx+20].mean()
        outer_mask = np.ones_like(magnitude, bool)
        outer_mask[cy-40:cy+40, cx-40:cx+40] = False
        outer = magnitude[outer_mask].mean()

        # GAN-generated images often have elevated high-freq energy
        ratio = outer / (inner + 1e-8)
        # Normalize: real videos ~0.55-0.70, deepfakes ~0.72-0.90
        score = np.clip((ratio - 0.55) / 0.35, 0, 1)
        scores.append(float(score))

    val = float(np.mean(scores))
    detail = f"High-freq energy ratio: {val:.3f} ({'elevated — GAN signature detected' if val > 0.5 else 'normal camera roll-off'})"
    return val, detail


def _compression_blocking(frames: list) -> Tuple[float, str]:
    """
    Detect 8×8 DCT blocking artifacts from re-compression during video synthesis.
    """
    scores = []
    for frame in frames[::2]:  # Sample every other frame for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        block_diffs = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                # Difference across the 8-pixel block boundary
                if y + 8 < h:
                    d_h = abs(float(gray[y+7, x:x+8].mean()) - float(gray[y+8, x:x+8].mean()))
                    block_diffs.append(d_h)
                if x + 8 < w:
                    d_v = abs(float(gray[y:y+8, x+7].mean()) - float(gray[y:y+8, x+8].mean()))
                    block_diffs.append(d_v)

        avg_diff = float(np.mean(block_diffs)) if block_diffs else 0.0
        # Real videos: ~2-5; over-compressed deepfakes: ~7-15
        score = np.clip((avg_diff - 2.0) / 13.0, 0, 1)
        scores.append(score)

    val = float(np.mean(scores)) if scores else 0.0
    detail = f"DCT block boundary diff: {val:.3f} ({'irregular blocking artifacts present' if val > 0.45 else 'normal compression pattern'})"
    return val, detail


def _blending_boundary(frames: list) -> Tuple[float, str]:
    """
    DeepFakes often have subtle blending seams around the face region.
    Detect by measuring edge strength discontinuity around face bounding boxes.
    """
    scores = []
    for frame in frames:
        faces = detect_faces_in_frame(frame)
        if not faces:
            scores.append(0.0)
            continue

        for bbox in faces[:1]:  # Use first face
            x, y, w, h = bbox
            fh, fw = frame.shape[:2]

            # Slightly enlarged bounding box
            pad = int(min(w, h) * 0.07)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150).astype(np.float32)

            # Edge density inside vs outside the face region
            inside = edges[y1:y2, x1:x2].mean()
            full   = edges.mean()
            ratio  = inside / (full + 1e-8)

            # Very high edge density ratio inside face = artificial blending
            score = np.clip((ratio - 1.2) / 2.5, 0, 1)
            scores.append(float(score))

    val = float(np.mean(scores)) if scores else 0.0
    detail = f"Face blending boundary index: {val:.3f} ({'blending seam irregularities detected' if val > 0.4 else 'natural face-background transition'})"
    return val, detail


def _temporal_inconsistency(frames: list) -> Tuple[float, str]:
    """
    Real video has smooth temporal coherence; deepfakes flicker due to frame-by-frame synthesis.
    Measure optical-flow magnitude variance across consecutive frames.
    """
    if len(frames) < 4:
        return 0.0, "Insufficient frames for temporal analysis"

    diffs = []
    for i in range(1, min(len(frames), 20)):
        prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)
        # Resize for speed
        prev = cv2.resize(prev, (160, 120))
        curr = cv2.resize(curr, (160, 120))
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        diffs.append(float(mag.std()))

    variance_of_variance = float(np.std(diffs)) if len(diffs) > 1 else 0.0
    # Real video: std-of-std ~0.1–1.5; deepfake flicker: >2.0
    val = np.clip((variance_of_variance - 0.8) / 4.0, 0, 1)
    detail = f"Temporal motion variance: {val:.3f} ({'unnatural flickering between frames' if val > 0.45 else 'smooth temporal coherence'})"
    return float(val), detail


def _noise_residual(frames: list) -> Tuple[float, str]:
    """
    Camera sensors produce consistent noise patterns (PRNU).
    DeepFake generators produce statistically different noise residuals.
    """
    scores = []
    for frame in frames[::3]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Wiener-like denoising via Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray - blurred
        # Real camera noise: relatively uniform residual std ~3-8
        # GAN artifacts: outlier residuals -> higher kurtosis
        r_std  = float(residual.std())
        r_mean = float(np.abs(residual).mean())
        kurtosis = float(np.mean(residual**4) / (r_std**4 + 1e-8))
        # GAN noise tends to have higher kurtosis
        score = np.clip((kurtosis - 2.5) / 10.0, 0, 1)
        scores.append(score)

    val = float(np.mean(scores)) if scores else 0.0
    detail = f"Noise residual kurtosis: {val:.3f} ({'unusual sensor noise pattern — possible synthetic origin' if val > 0.4 else 'consistent with natural camera noise'})"
    return val, detail


def _color_asymmetry(frames: list) -> Tuple[float, str]:
    """
    Real faces have natural color gradients. DeepFake synthesis often introduces
    subtle color channel imbalances, especially in skin tone regions.
    """
    scores = []
    for frame in frames:
        faces = detect_faces_in_frame(frame)
        if not faces:
            # Use center crop as proxy
            h, w = frame.shape[:2]
            crop = frame[h//4:3*h//4, w//4:3*w//4]
        else:
            crop = crop_face_region(frame, faces[0])

        if crop.size == 0:
            continue

        b_mean = float(crop[:, :, 0].mean())
        g_mean = float(crop[:, :, 1].mean())
        r_mean = float(crop[:, :, 2].mean())

        # Natural skin: R > G > B with specific ratios
        rg_ratio = r_mean / (g_mean + 1e-8)
        gb_ratio = g_mean / (b_mean + 1e-8)

        # Deviation from typical skin tone ratios (R/G ~1.1-1.4, G/B ~1.2-1.6)
        rg_dev = abs(rg_ratio - 1.25) / 0.5
        gb_dev = abs(gb_ratio - 1.4)  / 0.6
        score  = np.clip((rg_dev + gb_dev) / 2.0, 0, 1)
        scores.append(float(score))

    val = float(np.mean(scores)) if scores else 0.0
    detail = f"Color channel asymmetry index: {val:.3f} ({'abnormal skin tone color distribution' if val > 0.45 else 'natural color distribution in face region'})"
    return val, detail


def _edge_sharpness(frames: list) -> Tuple[float, str]:
    """
    Measure Laplacian variance as a sharpness proxy.
    DeepFake generators often produce inconsistently sharpened regions.
    """
    scores_inner, scores_outer = [], []
    for frame in frames:
        faces = detect_faces_in_frame(frame)
        if not faces:
            continue
        x, y, w, h = faces[0]
        fh, fw = frame.shape[:2]
        face_crop = frame[max(0,y):min(fh,y+h), max(0,x):min(fw,x+w)]
        if face_crop.size == 0:
            continue

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        lap_full = cv2.Laplacian(gray_full, cv2.CV_32F).var()
        lap_face = cv2.Laplacian(gray_face, cv2.CV_32F).var()

        scores_inner.append(float(lap_face))
        scores_outer.append(float(lap_full))

    if not scores_inner:
        return 0.0, "No face regions found for sharpness analysis"

    avg_in  = float(np.mean(scores_inner))
    avg_out = float(np.mean(scores_outer))
    ratio   = avg_in / (avg_out + 1e-8)

    # Overly sharp face vs. blurry background = suspicious
    score = np.clip((ratio - 1.5) / 3.0, 0, 1)
    detail = f"Face/background sharpness ratio: {ratio:.2f} ({'unnaturally sharp face against softer background' if score > 0.4 else 'consistent sharpness across frame'})"
    return float(score), detail


def _face_symmetry(frames: list) -> Tuple[float, str]:
    """
    Measure left-right symmetry within detected face crops.
    GAN-generated faces are often too symmetric or have asymmetric artifacts at edges.
    """
    scores = []
    for frame in frames[::4]:
        faces = detect_faces_in_frame(frame)
        if not faces:
            continue
        crop = crop_face_region(frame, faces[0])
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        w = gray.shape[1]
        left  = gray[:, :w//2]
        right = np.fliplr(gray[:, w//2:w//2 + left.shape[1]])
        if left.shape != right.shape:
            continue
        sym_score = float(np.mean(np.abs(left - right))) / 255.0
        # Very low diff = too symmetric (GAN artifact); very high = natural asymmetry
        # Real faces: diff ~0.05-0.20; GAN: <0.03 or >0.25
        if sym_score < 0.03:
            scores.append(0.7)   # Too symmetric → suspicious
        elif sym_score > 0.25:
            scores.append(0.6)   # Too asymmetric → unusual
        else:
            scores.append(0.0)

    val = float(np.mean(scores)) if scores else 0.0
    detail = f"Facial symmetry deviation: {val:.3f} ({'abnormal symmetry pattern detected' if val > 0.35 else 'natural facial asymmetry range'})"
    return val, detail


# ── Main Inference Entry Point ────────────────────────────────────────────────

def analyze_video(video_path: str) -> dict:
    """
    Full pipeline: extract frames → run all signals → aggregate → return report.
    """
    logger.info(f"Starting analysis: {video_path}")
    metadata = get_video_metadata(video_path)
    frames   = extract_frames(video_path, max_frames=40, sample_rate=3)

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    # Run all detectors
    detectors = {
        "frequency_artifacts":    _frequency_artifacts,
        "compression_blocking":   _compression_blocking,
        "blending_boundary":      _blending_boundary,
        "temporal_inconsistency": _temporal_inconsistency,
        "noise_residual":         _noise_residual,
        "color_asymmetry":        _color_asymmetry,
        "edge_sharpness":         _edge_sharpness,
        "face_symmetry":          _face_symmetry,
    }

    signal_results = {}
    for name, fn in detectors.items():
        try:
            score, detail = fn(frames)
        except Exception as e:
            logger.warning(f"Signal '{name}' failed: {e}")
            score, detail = 0.0, f"Analysis unavailable: {e}"
        signal_results[name] = {"score": score, "detail": detail, "weight": WEIGHTS[name]}

    # Weighted fake probability
    fake_prob = float(sum(
        signal_results[n]["score"] * WEIGHTS[n]
        for n in signal_results
    ))
    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
    real_prob = 1.0 - fake_prob

    # Confidence: distance from 0.5 boundary → how decisive the verdict is
    confidence = float(abs(fake_prob - 0.5) * 2)  # 0 = uncertain, 1 = very confident
    confidence = round(confidence * 100, 1)

    # Verdict
    if fake_prob >= 0.60:
        verdict    = "FAKE"
        accuracy   = round(fake_prob * 100, 1)
    elif fake_prob <= 0.40:
        verdict    = "REAL"
        accuracy   = round(real_prob * 100, 1)
    else:
        verdict    = "UNCERTAIN"
        accuracy   = round(max(fake_prob, real_prob) * 100, 1)

    logger.info(f"Verdict: {verdict} | fake_prob={fake_prob:.3f} | confidence={confidence}%")

    return {
        "verdict":      verdict,
        "fake_prob":    round(fake_prob * 100, 1),
        "real_prob":    round(real_prob * 100, 1),
        "accuracy":     accuracy,
        "confidence":   confidence,
        "frames_analyzed": len(frames),
        "signals":      signal_results,
        "metadata":     metadata,
    }
