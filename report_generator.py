"""
report_generator.py — Converts raw signal scores into human-readable analysis reports.
"""

from typing import Optional


SIGNAL_LABELS = {
    "frequency_artifacts":    "Frequency Domain Fingerprint",
    "compression_blocking":   "DCT Compression Artifact",
    "blending_boundary":      "Face Blending Boundary",
    "temporal_inconsistency": "Temporal Frame Coherence",
    "noise_residual":         "Sensor Noise Pattern",
    "color_asymmetry":        "Color Channel Distribution",
    "edge_sharpness":         "Edge Sharpness Ratio",
    "face_symmetry":          "Facial Symmetry Analysis",
}

SIGNAL_ICONS = {
    "frequency_artifacts":    "📡",
    "compression_blocking":   "🧱",
    "blending_boundary":      "🎭",
    "temporal_inconsistency": "⏱️",
    "noise_residual":         "📷",
    "color_asymmetry":        "🎨",
    "edge_sharpness":         "🔍",
    "face_symmetry":          "⚖️",
}

VERDICT_SUMMARIES = {
    "FAKE": [
        "This video shows strong indicators of synthetic manipulation.",
        "Multiple forensic signals confirm this is likely a DeepFake.",
        "Our analysis detected hallmarks of AI-generated face synthesis.",
        "Several artifact patterns consistent with GAN-based video generation were found.",
    ],
    "REAL": [
        "This video appears authentic with no significant manipulation signals.",
        "All forensic indicators are within expected ranges for genuine footage.",
        "No DeepFake artifacts were detected in the analyzed frames.",
        "The video's signal profile is consistent with real camera-captured footage.",
    ],
    "UNCERTAIN": [
        "The video shows mixed signals — some indicators are ambiguous.",
        "Inconclusive results: the video may have minor post-processing but is not definitively fake.",
        "Partial signals detected; manual expert review is recommended.",
    ],
}


def _severity_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.45:
        return "MEDIUM"
    elif score >= 0.20:
        return "LOW"
    else:
        return "CLEAN"


def _severity_color(score: float) -> str:
    if score >= 0.70:
        return "red"
    elif score >= 0.45:
        return "orange"
    elif score >= 0.20:
        return "yellow"
    else:
        return "green"


def generate_report(analysis_result: dict, filename: str = "video") -> dict:
    """
    Build a structured report from the raw analysis result.
    Returns a dict suitable for JSON serialization.
    """
    import random, hashlib

    verdict    = analysis_result["verdict"]
    fake_prob  = analysis_result["fake_prob"]
    real_prob  = analysis_result["real_prob"]
    accuracy   = analysis_result["accuracy"]
    confidence = analysis_result["confidence"]
    signals    = analysis_result["signals"]
    metadata   = analysis_result.get("metadata", {})

    # Pick summary based on verdict (seeded on filename for stability)
    rng = random.Random(hashlib.md5(filename.encode()).hexdigest())
    summary_pool = VERDICT_SUMMARIES[verdict]
    summary = rng.choice(summary_pool)

    # Build signal breakdown
    signal_breakdown = []
    for key, data in signals.items():
        score   = data["score"]
        detail  = data["detail"]
        weight  = data["weight"]
        label   = SIGNAL_LABELS.get(key, key)
        icon    = SIGNAL_ICONS.get(key, "🔬")
        sev     = _severity_label(score)
        color   = _severity_color(score)
        contribution = round(score * weight * 100, 1)

        signal_breakdown.append({
            "key":          key,
            "label":        label,
            "icon":         icon,
            "score":        round(score * 100, 1),
            "weight_pct":   round(weight * 100),
            "severity":     sev,
            "color":        color,
            "detail":       detail,
            "contribution": contribution,
        })

    # Sort by contribution descending (most impactful first)
    signal_breakdown.sort(key=lambda x: x["contribution"], reverse=True)

    # Top reasons (signals with HIGH or MEDIUM severity)
    top_reasons = [
        f"{s['icon']} **{s['label']}**: {s['detail']}"
        for s in signal_breakdown
        if s["severity"] in ("HIGH", "MEDIUM")
    ]
    if not top_reasons:
        top_reasons = [
            f"{s['icon']} **{s['label']}**: {s['detail']}"
            for s in signal_breakdown[:3]
        ]

    # Video metadata summary
    duration   = metadata.get("duration_sec", 0)
    resolution = f"{metadata.get('width','?')}×{metadata.get('height','?')}"
    fps        = round(metadata.get("fps", 0), 1)

    return {
        "filename":          filename,
        "verdict":           verdict,
        "fake_probability":  fake_prob,
        "real_probability":  real_prob,
        "accuracy":          accuracy,
        "confidence":        confidence,
        "summary":           summary,
        "top_reasons":       top_reasons,
        "signal_breakdown":  signal_breakdown,
        "frames_analyzed":   analysis_result.get("frames_analyzed", 0),
        "video_info": {
            "duration_sec": duration,
            "resolution":   resolution,
            "fps":          fps,
        },
    }
