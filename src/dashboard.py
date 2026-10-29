"""
Interactive CPU Dashboard & Heritage Security Interface.
Guarding the Stones — Egocentric Heritage Protection AI.

Master UI Implementation closely following editorial heritage visual identity:
- Color System: Warm Ivory (#F7F5F0), Card White (#FFFFFF), Deep Slate (#29372F),
  Secondary Sage (#737B73), Primary Sage Green (#527653), Light Sage (#E8EFE5),
  Border (#E7E4DC), Muted Amber (#D18A18), Light Amber (#FFF2DC),
  Muted Terracotta (#B85E42), Light Terracotta (#FBE9E4), Dark Green (#284B38).
- Typography: Headings in Lora (serif), Body & UI in DM Sans (sans-serif).
- Modular 2-column workspace, functional navigation sidebar, 4 sample video loaders,
  continuous risk progress bars, video metadata inspection card, 8-frame temporal gallery
  with timestamp toggle, sensitivity sliders with default reset, field officer sign-off,
  clear history with confirmation, and CSV/JSON audit export.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import gradio as gr
import numpy as np

from src.config import CANDIDATE_ACTION_CLASSES, SUPPORTED_VIDEO_EXTENSIONS, DEFAULT_RAW_DIR
from src.video_parser import extract_video_metadata, VideoMetadata
from src.inference import (
    InferenceEngine,
    PredictionResult,
    compute_verdict,
    categorize_action_label,
    export_predictions_json,
    export_predictions_csv,
    BatchPredictionReport,
    MODEL_ENGINE_VERSION,
)
from src.models import ModelConfig


# Default session log table columns
HIST_HEADERS = [
    "Time",
    "Video File",
    "Predicted Action",
    "Confidence",
    "Risk Score",
    "Verdict",
    "Latency (ms)",
    "Officer Sign-off",
]


# =====================================================================
# CSS Design System (Warm, Natural, Editorial Heritage Tech Aesthetic)
# =====================================================================

DASHBOARD_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Lora:ital,wght@0,400..700;1,400..700&display=swap');

:root {
    --bg-main: #F7F5F0;
    --bg-card: #FFFFFF;
    --text-primary: #29372F;
    --text-secondary: #737B73;
    --accent-sage: #527653;
    --accent-sage-hover: #436043;
    --accent-sage-light: #E8EFE5;
    --border-color: #E7E4DC;
    --border-subtle: #F0EEE7;
    --status-amber: #D18A18;
    --status-amber-light: #FFF2DC;
    --status-terracotta: #B85E42;
    --status-terracotta-light: #FBE9E4;
    --dark-green: #284B38;
    --font-heading: 'Lora', Georgia, 'Times New Roman', serif;
    --font-body: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --shadow-subtle: 0 1px 3px rgba(41, 55, 47, 0.04), 0 1px 2px rgba(41, 55, 47, 0.02);
    --shadow-card: 0 4px 14px rgba(41, 55, 47, 0.05), 0 1px 3px rgba(41, 55, 47, 0.03);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* Global Page Typography & Background */
body, .gradio-container {
    background-color: var(--bg-main) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}

/* Headings */
h1, h2, h3, h4, h5, .brand-title, .section-heading, .section-title {
    font-family: var(--font-heading) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.015em;
    font-weight: 600;
}

/* Section Tags */
.section-tag {
    font-family: var(--font-body);
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent-sage);
    margin-bottom: 2px;
}

/* Cards & Containers */
.block, .panel, .gr-box {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-subtle) !important;
}

/* Primary Sage Buttons */
button.primary, button[variant="primary"], .btn-primary {
    background: var(--accent-sage) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent-sage) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 20px !important;
    box-shadow: var(--shadow-subtle) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: var(--accent-sage-hover) !important;
    border-color: var(--accent-sage-hover) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-card) !important;
}

/* Secondary Neutral Buttons */
button.secondary, button[variant="secondary"], .btn-secondary {
    background: #FFFFFF !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 9px 16px !important;
    box-shadow: var(--shadow-subtle) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    background: #FAF9F5 !important;
    border-color: #DCD9D0 !important;
}

/* Sidebar Navigation Buttons */
.sidebar-nav-btn {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin-bottom: 6px !important;
    padding: 10px 14px !important;
    font-size: 0.90rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid transparent !important;
    background: #FFFFFF !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-subtle) !important;
    transition: all 0.15s ease !important;
}
.sidebar-nav-btn:hover {
    background: #F4F2EB !important;
    color: var(--dark-green) !important;
}
.sidebar-nav-btn.active-nav {
    background: var(--accent-sage-light) !important;
    color: var(--dark-green) !important;
    font-weight: 700 !important;
    border-left: 3.5px solid var(--accent-sage) !important;
    border-color: #D5E2D2 !important;
}

/* Video Ingestion Container */
[data-testid="video"], .upload-container {
    border: 1.5px dashed var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    background: #FAF9F5 !important;
    transition: all 0.2s ease !important;
}
[data-testid="video"]:hover, .upload-container:hover {
    border-color: var(--accent-sage) !important;
    background: #F4F8F3 !important;
}

/* Form Inputs, Textboxes, Sliders */
input, textarea, select {
    font-family: var(--font-body) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    background: #FFFFFF !important;
}
input[type="range"] {
    accent-color: var(--accent-sage) !important;
}

/* Tables */
table, .dataframe {
    background: #FFFFFF !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-color) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}
table th {
    background: #F4F2EB !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding: 12px 14px !important;
}
table td {
    padding: 12px 14px !important;
    border-bottom: 1px solid #F0EEE7 !important;
    color: var(--text-primary) !important;
    font-variant-numeric: tabular-nums !important;
}
table tr:hover td {
    background: #FAF9F5 !important;
}

/* Frame Gallery Styling */
[data-testid="gallery"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="gallery"] .gallery-item {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="gallery"] .gallery-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-card) !important;
    border-color: var(--accent-sage) !important;
}
"""


# =====================================================================
# HTML Card Renderers (Light, Warm, Accessible)
# =====================================================================

def render_verdict_card(
    result: Optional[PredictionResult],
    review_thr: float,
    alert_thr: float,
    conf_thr: float,
) -> str:
    """Render an accessible, light, warm operational verdict card in HTML."""
    if result is None:
        return """
        <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:14px; padding:32px 24px; text-align:center; box-shadow:0 1px 3px rgba(41,55,47,0.04);">
            <div style="width:48px; height:48px; border-radius:24px; background:#E8EFE5; color:#527653; display:inline-flex; align-items:center; justify-content:center; margin-bottom:12px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect>
                    <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
                    <line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line>
                </svg>
            </div>
            <div style="font-family:'Lora', serif; font-size:1.15rem; font-weight:700; color:#29372F; margin-bottom:4px; letter-spacing:-0.01em;">
                READY FOR VIDEO INGESTION
            </div>
            <div style="font-size:0.88rem; color:#737B73; max-width:400px; margin:0 auto; line-height:1.45;">
                Select a sample clip or upload first-person video footage to initiate CPU-accelerated temporal action recognition and continuous heritage risk assessment.
            </div>
        </div>
        """

    if result.status == "error":
        return f"""
        <div style="background:#FFF5F5; border:1px solid #FCD4D4; border-radius:14px; padding:20px; box-shadow:0 1px 3px rgba(41,55,47,0.04);">
            <div style="display:flex; align-items:flex-start;">
                <div style="width:36px; height:36px; border-radius:18px; background:#FEE2E2; color:#B85E42; display:flex; align-items:center; justify-content:center; margin-right:12px; flex-shrink:0;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                </div>
                <div>
                    <div style="font-family:'Lora', serif; font-size:1.05rem; font-weight:700; color:#B85E42; margin-bottom:2px;">Video Processing Error</div>
                    <div style="font-size:0.85rem; color:#29372F; margin-bottom:4px;"><strong>Error Code:</strong> <code>{result.error_type}</code></div>
                    <div style="font-size:0.82rem; color:#737B73; line-height:1.4;">{result.error_message}</div>
                </div>
            </div>
        </div>
        """

    # Dynamically recompute verdict if thresholds were adjusted via sliders
    verdict, uncertain = compute_verdict(
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        risk_score=result.risk_score,
        review_threshold=review_thr,
        alert_threshold=alert_thr,
        confidence_threshold=conf_thr,
    )

    # Design tokens based on operational verdict
    if verdict == "POTENTIAL VANDALISM":
        tint_bg = "#FBE9E4"
        border_accent = "#F7D0C6"
        status_text = "#B85E42"
        badge_bg = "#B85E42"
        badge_text = "#FFFFFF"
        icon_svg = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>"""
        status_heading = "POTENTIAL VANDALISM"
        status_sub = "High-severity physical alteration pattern detected"
        risk_level_str = "High / Alert"
    elif verdict == "UNCERTAIN / HUMAN REVIEW":
        tint_bg = "#FFF2DC"
        border_accent = "#F5DEB3"
        status_text = "#D18A18"
        badge_bg = "#D18A18"
        badge_text = "#FFFFFF"
        icon_svg = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>"""
        status_heading = "UNCERTAIN / HUMAN REVIEW"
        status_sub = "Ambiguous gesture or confidence below certainty threshold"
        risk_level_str = "Elevated / Review"
    else:  # NORMAL
        tint_bg = "#E8EFE5"
        border_accent = "#C8DAC4"
        status_text = "#284B38"
        badge_bg = "#527653"
        badge_text = "#FFFFFF"
        icon_svg = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>"""
        status_heading = "NORMAL"
        status_sub = "Benign visitor interaction observed"
        risk_level_str = "Low / Benign"

    uncertainty_pill = (
        '<span style="background:#FFF2DC; color:#D18A18; border:1px solid #F5DEB3; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; margin-left:8px;">FLAGGED FOR REVIEW</span>'
        if uncertain
        else '<span style="background:#E8EFE5; color:#284B38; border:1px solid #C8DAC4; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; margin-left:8px;">CONFIDENT</span>'
    )

    risk_bar_pct = min(max(int(result.risk_score * 100), 0), 100)
    peak_evidence_str = (
        f"{result.peak_evidence_timestamp_sec:.2f}s (Frame #{result.peak_evidence_frame_idx})"
        if result.peak_evidence_timestamp_sec is not None
        else "N/A"
    )

    # Clean display name for action
    display_action = result.predicted_label.replace("_", " ").title()

    # Probability bars snippet for top predicted classes
    top_items = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
    prob_bars_html = ""
    for label, prob in top_items:
        clean_lbl = label.replace("_", " ").title()
        pct = int(prob * 100)
        prob_bars_html += f"""
        <div style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:2px;">
                <span style="font-weight:600; color:#29372F;">{clean_lbl}</span>
                <span style="font-variant-numeric:tabular-nums; color:#737B73;">{prob * 100:.1f}%</span>
            </div>
            <div style="width:100%; height:5px; background:#F0EEE7; border-radius:3px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:#527653;"></div>
            </div>
        </div>
        """

    return f"""
    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:14px; overflow:hidden; box-shadow:0 1px 4px rgba(41,55,47,0.05); font-family:'DM Sans', sans-serif;">
        <!-- Top Status Banner -->
        <div style="background:{tint_bg}; border-bottom:1px solid {border_accent}; padding:16px 20px; display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center;">
                <div style="color:{status_text}; margin-right:10px; display:flex; align-items:center;">
                    {icon_svg}
                </div>
                <div>
                    <div style="display:flex; align-items:center;">
                        <span style="font-family:'Lora', serif; font-size:1.15rem; font-weight:700; color:{status_text}; letter-spacing:-0.01em;">
                            {status_heading}
                        </span>
                        {uncertainty_pill}
                    </div>
                    <div style="font-size:0.78rem; color:{status_text}; opacity:0.90; margin-top:1px;">
                        {status_sub}
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <span style="background:#FFFFFF; color:#29372F; border:1px solid {border_accent}; padding:4px 10px; border-radius:6px; font-size:0.75rem; font-weight:600; text-transform:uppercase;">
                    {result.action_category.replace('_', ' ')}
                </span>
            </div>
        </div>

        <!-- Interior Results Body -->
        <div style="padding:20px;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:18px;">
                <!-- Predicted Action Card -->
                <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:10px; padding:14px;">
                    <div style="font-size:0.70rem; text-transform:uppercase; letter-spacing:0.04em; color:#737B73; font-weight:600; margin-bottom:4px;">
                        Predicted Action
                    </div>
                    <div style="font-family:'Lora', serif; font-size:1.20rem; font-weight:700; color:#29372F; line-height:1.2;">
                        {display_action}
                    </div>
                    <div style="font-size:0.82rem; color:#737B73; margin-top:4px; font-variant-numeric:tabular-nums;">
                        Confidence: <strong style="color:#29372F;">{result.confidence * 100:.1f}%</strong>
                    </div>
                </div>

                <!-- Continuous Risk Card -->
                <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:10px; padding:14px;">
                    <div style="font-size:0.70rem; text-transform:uppercase; letter-spacing:0.04em; color:#737B73; font-weight:600; margin-bottom:4px;">
                        Continuous Risk Score
                    </div>
                    <div style="font-family:'Lora', serif; font-size:1.20rem; font-weight:700; color:#29372F; line-height:1.2; font-variant-numeric:tabular-nums;">
                        {result.risk_score:.4f} <span style="font-size:0.80rem; font-weight:500; color:#737B73;">/ 1.0000</span>
                    </div>
                    <div style="font-size:0.82rem; color:#737B73; margin-top:4px;">
                        Risk Level: <strong style="color:{status_text};">{risk_level_str}</strong>
                    </div>
                </div>
            </div>

            <!-- Risk Progress Meter -->
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#737B73; margin-bottom:5px;">
                    <span style="font-weight:500;">Heritage Risk Scale</span>
                    <span style="font-variant-numeric:tabular-nums;">{risk_bar_pct}% (Review: {int(review_thr*100)}% | Alert: {int(alert_thr*100)}%)</span>
                </div>
                <div style="width:100%; height:8px; background:#E7E4DC; border-radius:4px; overflow:hidden;">
                    <div style="width:{risk_bar_pct}%; height:100%; background:{badge_bg}; transition:width 0.4s ease;"></div>
                </div>
            </div>

            <!-- Top Predicted Probabilities Horizontal Bars -->
            <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:10px; padding:12px; margin-bottom:14px;">
                <div style="font-size:0.70rem; text-transform:uppercase; letter-spacing:0.04em; color:#737B73; font-weight:600; margin-bottom:8px;">
                    Top Predicted Action Probabilities
                </div>
                {prob_bars_html}
            </div>

            <!-- Operational Notice Footer -->
            <div style="background:#F0EEE7; border-radius:8px; padding:10px 12px; font-size:0.78rem; color:#737B73; line-height:1.4; display:flex; align-items:flex-start;">
                <span style="margin-right:6px; color:#527653; font-size:0.9rem;">ℹ️</span>
                <span><em>Action recognition identifies temporal movement patterns. It does <strong>NOT</strong> constitute verified physical stone damage or malicious intent. Human review is mandatory.</em></span>
            </div>
        </div>
    </div>
    """


def render_video_info_card(result: Optional[PredictionResult], meta: Optional[VideoMetadata] = None) -> str:
    """Render Section C: Video Information and loaded model parameters."""
    if result is None:
        return """
        <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px 20px; font-family:'DM Sans', sans-serif; box-shadow:0 1px 3px rgba(41,55,47,0.03);">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="color:#527653; font-size:1.1rem;">📹</span>
                <span style="font-family:'Lora', serif; font-size:1.0rem; font-weight:600; color:#29372F;">Video Information & Model Parameters</span>
            </div>
            <div style="font-size:0.84rem; color:#737B73; line-height:1.45;">
                No active video loaded. Select a test sample or upload footage to inspect duration, native resolution, sampling rate, and CPU latency specifications.
            </div>
        </div>
        """

    # Extract metadata attributes safely
    filename = Path(result.video_path).name if result.video_path else "Uploaded Clip"
    duration = f"{result.video_duration_sec:.2f} s" if result.video_duration_sec > 0 else "Unavailable"
    fps = f"{result.video_fps:.2f} fps" if result.video_fps > 0 else "Unavailable"
    
    if meta and meta.width and meta.height:
        resolution = f"{meta.width} × {meta.height} px"
    else:
        resolution = "768 × 432 px (estimated)"

    num_frames = f"{result.num_frames} frames (uniform temporal)"
    latency = f"{result.total_latency_ms:.1f} ms" if result.total_latency_ms > 0 else "Unavailable"
    model_ver = result.model_version
    backbone = result.backbone
    temporal_model = f"{result.model_type.upper()} Temporal Network"

    return f"""
    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px 20px; font-family:'DM Sans', sans-serif; box-shadow:0 1px 3px rgba(41,55,47,0.03); margin-top:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #F0EEE7; padding-bottom:10px; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:#527653; font-size:1.1rem;">📹</span>
                <span style="font-family:'Lora', serif; font-size:1.02rem; font-weight:600; color:#29372F;">Section C — Video Information</span>
            </div>
            <span style="font-size:0.72rem; color:#284B38; background:#E8EFE5; border:1px solid #C8DAC4; padding:2px 8px; border-radius:12px; font-weight:600; text-transform:uppercase;">
                Ingestion Verified
            </span>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px 18px; font-size:0.84rem;">
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">File Name</div>
                <div style="color:#29372F; font-weight:600; word-break:break-all;">{filename}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Duration</div>
                <div style="color:#29372F; font-weight:600; font-variant-numeric:tabular-nums;">{duration}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Resolution</div>
                <div style="color:#29372F; font-weight:600; font-variant-numeric:tabular-nums;">{resolution}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Native FPS</div>
                <div style="color:#29372F; font-weight:600; font-variant-numeric:tabular-nums;">{fps}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Frames Sampled</div>
                <div style="color:#29372F; font-weight:600;">{num_frames}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Processing Latency</div>
                <div style="color:#527653; font-weight:700; font-variant-numeric:tabular-nums;">{latency}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Model Version</div>
                <div style="color:#29372F; font-weight:600;">v{model_ver}</div>
            </div>
            <div>
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Backbone</div>
                <div style="color:#29372F; font-weight:600;">{backbone}</div>
            </div>
            <div style="grid-column: span 2;">
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600; letter-spacing:0.04em;">Temporal Model</div>
                <div style="color:#29372F; font-weight:600;">{temporal_model}</div>
            </div>
        </div>
    </div>
    """


def render_performance_card(result: Optional[PredictionResult]) -> str:
    """Render CPU latency and execution profile metrics card in warm editorial style."""
    if result is None or result.status == "error":
        return ""

    fps_eff = (result.num_frames / (result.total_latency_ms / 1000.0)) if result.total_latency_ms > 0 else 0.0
    return f"""
    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:16px; margin-top:12px; box-shadow:0 1px 3px rgba(41,55,47,0.03); font-family:'DM Sans', sans-serif;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #F0EEE7; padding-bottom:8px; margin-bottom:10px;">
            <div style="display:flex; align-items:center;">
                <span style="color:#527653; margin-right:6px; font-size:1.0rem;">⚡</span>
                <span style="font-family:'Lora', serif; font-weight:600; font-size:0.95rem; color:#29372F;">CPU Execution Profile</span>
            </div>
            <span style="font-size:0.75rem; color:#737B73; background:#F0EEE7; padding:3px 8px; border-radius:4px;">
                {result.model_type} ({result.backbone}) v{result.model_version}
            </span>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; text-align:center;">
            <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:8px; padding:8px 4px;">
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600;">Decode + Prep</div>
                <div style="font-family:'DM Sans', sans-serif; font-weight:700; font-size:0.95rem; color:#29372F; font-variant-numeric:tabular-nums; margin-top:2px;">{result.decode_and_preprocess_ms:.1f} ms</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:8px; padding:8px 4px;">
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600;">Inference</div>
                <div style="font-family:'DM Sans', sans-serif; font-weight:700; font-size:0.95rem; color:#527653; font-variant-numeric:tabular-nums; margin-top:2px;">{result.inference_ms:.1f} ms</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:8px; padding:8px 4px;">
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600;">Total Latency</div>
                <div style="font-family:'DM Sans', sans-serif; font-weight:700; font-size:0.95rem; color:#29372F; font-variant-numeric:tabular-nums; margin-top:2px;">{result.total_latency_ms:.1f} ms</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:8px; padding:8px 4px;">
                <div style="font-size:0.70rem; text-transform:uppercase; color:#737B73; font-weight:600;">Effective Speed</div>
                <div style="font-family:'DM Sans', sans-serif; font-weight:700; font-size:0.95rem; color:#284B38; font-variant-numeric:tabular-nums; margin-top:2px;">{fps_eff:.1f} FPS</div>
            </div>
        </div>
    </div>
    """


def render_timeline_summary(result: Optional[PredictionResult]) -> str:
    """Render video duration and temporal sampling timeline card."""
    if result is None or result.status == "error" or not result.sampled_frames:
        return ""

    total_duration = result.video_duration_sec
    peak_timestamp = result.peak_evidence_timestamp_sec or 0.0
    peak_frame = result.peak_evidence_frame_idx or 0

    return f"""
    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:14px 16px; margin-top:12px; font-family:'DM Sans', sans-serif; box-shadow:0 1px 3px rgba(41,55,47,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center;">
                <span style="color:#527653; margin-right:6px; font-size:0.95rem;">⏱️</span>
                <span style="font-family:'Lora', serif; font-weight:600; font-size:0.92rem; color:#29372F;">Temporal Sequence Summary</span>
            </div>
            <div style="font-size:0.80rem; color:#737B73; font-variant-numeric:tabular-nums;">
                Duration: <strong style="color:#29372F;">{total_duration:.2f} s</strong> ({result.video_total_frames} frames @ {result.video_fps:.1f} fps)
            </div>
        </div>
        <div style="margin-top:6px; font-size:0.82rem; color:#D18A18; background:#FFF2DC; border:1px solid #F5DEB3; padding:6px 10px; border-radius:6px; display:inline-block;">
            ★ Peak Salient Motion Detected at <strong>{peak_timestamp:.2f} s</strong> (Frame #{peak_frame})
        </div>
    </div>
    """


# =====================================================================
# Dashboard Builder
# =====================================================================

def create_dashboard(
    engine: Optional[InferenceEngine] = None,
    share: bool = False,
    server_port: int = 7860,
) -> gr.Blocks:
    """Construct and launch the Gradio interactive dashboard with heritage editorial identity."""
    # Initialize CPU inference engine if not provided
    if engine is None:
        cfg = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            num_frames=8,
        )
        engine = InferenceEngine(config=cfg)

    # In-memory session history accumulator
    session_records: List[Dict[str, Any]] = []

    # Detect sample files in workspace for quick loading
    sample_carving = Path("data/raw/carving/carving 02.mp4").resolve()
    sample_photo = Path("data/raw/normal/normal (3).mp4").resolve()
    sample_walk = Path("data/raw/normal/normal (1).mp4").resolve()
    sample_touch = Path("data/raw/normal/normal (2).mp4").resolve()

    with gr.Blocks(
        title="Guarding the Stones — Egocentric Heritage Protection AI",
    ) as app:
        # Injected Custom CSS
        gr.HTML(f"<style>{DASHBOARD_CUSTOM_CSS}</style>")

        # State tracking
        current_result_state = gr.State(None)
        current_meta_state = gr.State(None)
        session_history_state = gr.State([])
        raw_gallery_state = gr.State([])

        # =================================================================
        # Header (Full-width, Light, Editorial Heritage Emblem)
        # =================================================================
        with gr.Row():
            with gr.Column(scale=12):
                gr.HTML("""
                <header style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:14px; padding:20px 24px; margin-bottom:18px; box-shadow:0 1px 4px rgba(41,55,47,0.04); font-family:'DM Sans', sans-serif;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                        <div style="display:flex; align-items:center;">
                            <!-- Classical Heritage Emblem SVG -->
                            <div style="width:48px; height:48px; border-radius:12px; background:#E8EFE5; border:1px solid #C8DAC4; display:flex; align-items:center; justify-content:center; margin-right:16px; flex-shrink:0;">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#284B38" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M4 3h16M7 3v18M17 3v18M4 21h16M2 7h20M2 17h20"></path>
                                </svg>
                            </div>
                            <div>
                                <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                                    <h1 style="margin:0; font-family:'Lora', serif; font-size:1.75rem; font-weight:700; color:#29372F; letter-spacing:-0.02em;">
                                        Guarding the Stones
                                    </h1>
                                    <span style="background:#E8EFE5; color:#284B38; border:1px solid #C8DAC4; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:12px;">
                                        CPU MODE • NO GPU REQUIRED
                                    </span>
                                </div>
                                <div style="font-size:0.92rem; color:#527653; font-weight:600; margin-top:2px;">
                                    Egocentric Heritage Protection AI • Real-Time Video Action Recognition & Risk Assessment
                                </div>
                            </div>
                        </div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div style="text-align:right; font-size:0.78rem; color:#737B73; background:#FAF9F5; border:1px solid #E7E4DC; padding:8px 14px; border-radius:8px;">
                                <div>Engine: <strong style="color:#29372F;">MobileNetV3-Small + 1D TCN</strong></div>
                                <div style="color:#284B38; font-weight:600;">v1.0.0-cpu • 8-Frame Saliency</div>
                            </div>
                        </div>
                    </div>
                </header>
                """)

        # =================================================================
        # Navigation Sidebar & Dynamic Workspace Layout
        # =================================================================
        with gr.Row():
            # Navigation Sidebar Column
            with gr.Column(scale=2, min_width=210):
                gr.HTML("""
                <div style="padding:10px 12px; font-family:'Lora', serif; font-size:1.0rem; font-weight:700; color:#284B38; border-bottom:1px solid #E7E4DC; margin-bottom:10px;">
                    Navigation
                </div>
                """)
                nav_btn_video = gr.Button("🎥 Video Analysis", elem_classes=["sidebar-nav-btn", "active-nav"], variant="secondary")
                nav_btn_batch = gr.Button("📁 Batch Analysis", elem_classes=["sidebar-nav-btn"], variant="secondary")
                nav_btn_model = gr.Button("🧠 Model Information", elem_classes=["sidebar-nav-btn"], variant="secondary")
                nav_btn_settings = gr.Button("⚙️ Settings", elem_classes=["sidebar-nav-btn"], variant="secondary")
                nav_btn_help = gr.Button("📖 Help & About", elem_classes=["sidebar-nav-btn"], variant="secondary")

                # System specs badge widget
                gr.HTML("""
                <div style="margin-top:20px; padding:14px; background:#FFFFFF; border:1px solid #E7E4DC; border-radius:10px; font-size:0.76rem; color:#737B73; box-shadow:0 1px 3px rgba(41,55,47,0.03);">
                    <div style="font-weight:700; color:#284B38; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.04em;">Operational Specs</div>
                    <div style="margin-bottom:3px;">Host: <strong style="color:#29372F;">Intel Core i5 CPU</strong></div>
                    <div style="margin-bottom:3px;">Precision: <strong style="color:#29372F;">Float32</strong></div>
                    <div style="margin-bottom:3px;">Sampling: <strong style="color:#29372F;">8 Frames Uniform</strong></div>
                    <div style="margin-top:8px; color:#527653; font-weight:600; display:flex; align-items:center; gap:4px;">
                        <span style="font-size:0.65rem;">●</span> Active Engine
                    </div>
                </div>
                """)

            # Main Workspace Area (Switched via Sidebar Navigation)
            with gr.Column(scale=10):
                # -------------------------------------------------------------
                # VIEW 1: VIDEO ANALYSIS (DEFAULT MAIN DASHBOARD)
                # -------------------------------------------------------------
                with gr.Column(visible=True) as view_video_analysis:
                    with gr.Row():
                        # Section A: Ingest Video Footage (Left Column)
                        with gr.Column(scale=6):
                            gr.HTML("""
                            <div class="section-tag">01 INGEST FOOTAGE</div>
                            <h3 class="section-title">Ingest Video Footage</h3>
                            """)
                            
                            video_input = gr.Video(
                                label="Upload Video Clip or Capture Live Stream (Supported: MP4, AVI, MOV, MKV)",
                                sources=["upload", "webcam"],
                            )

                            # Row of Sample Video Cards
                            gr.HTML("""
                            <div style="margin-top:14px; margin-bottom:8px;">
                                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; color:#737B73; font-weight:700;">
                                    Preloaded Field Samples
                                </div>
                                <div style="font-size:0.78rem; color:#737B73;">Click any card to immediately load authentic egocentric footage:</div>
                            </div>
                            """)
                            with gr.Row():
                                sample_carving_btn = gr.Button("🔨 Carving / Damage\n(carving 02.mp4)", variant="secondary", size="sm")
                                sample_photo_btn = gr.Button("📷 Photographing\n(normal 03.mp4)", variant="secondary", size="sm")
                            with gr.Row():
                                sample_walk_btn = gr.Button("🚶 Walking / Patrol\n(normal 01.mp4)", variant="secondary", size="sm")
                                sample_touch_btn = gr.Button("🖐️ Permitted Touching\n(normal 02.mp4)", variant="secondary", size="sm")

                            with gr.Row():
                                analyze_btn = gr.Button("🔍 Analyze Video Clip", variant="primary", scale=2)
                                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

                            # Section E: Sensitivity Settings
                            with gr.Accordion("⚙️ Section E — Sensitivity Settings", open=True):
                                gr.HTML("""
                                <div style="font-size:0.80rem; color:#737B73; margin-bottom:10px; line-height:1.45;">
                                    Tune detection sensitivity. Adjusting sliders immediately updates the operational verdict and risk classification without re-decoding.
                                </div>
                                """)
                                review_slider = gr.Slider(
                                    minimum=0.10,
                                    maximum=0.90,
                                    value=0.35,
                                    step=0.05,
                                    label="Review Threshold (Muted Amber — Lower = More Conservative)",
                                    elem_classes=["review-slider"],
                                )
                                alert_slider = gr.Slider(
                                    minimum=0.20,
                                    maximum=0.95,
                                    value=0.70,
                                    step=0.05,
                                    label="Alert Threshold (Muted Terracotta — Above which Potential Vandalism triggers)",
                                    elem_classes=["alert-slider"],
                                )
                                conf_slider = gr.Slider(
                                    minimum=0.10,
                                    maximum=0.90,
                                    value=0.40,
                                    step=0.05,
                                    label="Confidence Cutoff (Below which Uncertainty is flagged)",
                                )
                                reset_thresholds_btn = gr.Button("↺ Reset Thresholds to Defaults", variant="secondary", size="sm")

                        # Section B & C: Analysis Result & Video Info (Right Column)
                        with gr.Column(scale=6):
                            gr.HTML("""
                            <div class="section-tag">02 OPERATIONAL VERDICT</div>
                            <h3 class="section-title">Analysis Result & Risk Assessment</h3>
                            """)
                            verdict_html = gr.HTML(render_verdict_card(None, 0.35, 0.70, 0.40))
                            label_output = gr.Label(num_top_classes=5, label="Top Predicted Class Probabilities")
                            
                            # Section C: Video Information Card
                            video_info_html = gr.HTML(render_video_info_card(None, None))
                            perf_html = gr.HTML("")
                            timeline_html = gr.HTML("")

                    # Section D: Frames Analyzed (Temporal Evidence)
                    gr.HTML("""
                    <div style="margin-top:20px;">
                        <div class="section-tag">03 TEMPORAL EVIDENCE</div>
                        <h3 class="section-title">Section D — Sampled Frames Analyzed</h3>
                        <div style="font-size:0.82rem; color:#737B73; margin-bottom:10px;">
                            Uniform 8-frame temporal samples extracted across the video stream with model attention weights.
                        </div>
                    </div>
                    """)
                    show_timestamps_chk = gr.Checkbox(
                        value=True,
                        label="Show Timestamps & Attention Weights on Frames",
                    )
                    gallery_output = gr.Gallery(
                        label="Sampled Frames & Temporal Timestamps",
                        columns=4,
                        height="auto",
                        object_fit="contain",
                    )

                    # Section F: Field Officer Review & Active Session Audit Trail
                    gr.HTML("""
                    <div style="margin-top:24px;">
                        <div class="section-tag">05 AUDIT LOGGING</div>
                        <h3 class="section-title">Section F — Active Session Audit Trail</h3>
                    </div>
                    """)
                    with gr.Accordion("🛡️ Heritage Officer Field Review & Verification Sign-Off", open=True):
                        with gr.Row():
                            with gr.Column(scale=2):
                                officer_decision = gr.Radio(
                                    choices=[
                                        "Pending Review",
                                        "Verified: Benign / Permitted Activity",
                                        "Verified: Ambiguous Interaction (Monitor)",
                                        "Escalate: Suspected Physical Alteration",
                                        "Dismissed: False Alarm",
                                    ],
                                    value="Pending Review",
                                    label="Field Officer Determination",
                                )
                            with gr.Column(scale=3):
                                officer_notes = gr.Textbox(
                                    label="Officer Field Observations & Justification",
                                    placeholder="Enter monument sector, visitor ID, physical observation notes, or justification...",
                                    lines=2,
                                )
                                with gr.Row():
                                    submit_review_btn = gr.Button("💾 Record Officer Sign-off", variant="secondary")
                                    review_status_msg = gr.Markdown("")

                    history_table = gr.Dataframe(
                        headers=HIST_HEADERS,
                        interactive=False,
                        label="Active Evaluated Video Records",
                    )

                    with gr.Row():
                        export_csv_btn = gr.Button("📥 Export Session Log (CSV)", variant="secondary")
                        export_json_btn = gr.Button("📥 Export Session Log (JSON)", variant="secondary")
                        clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        file_download = gr.File(label="Generated Report Download", interactive=False)

                    # Clear History Confirmation Row
                    with gr.Row(visible=False) as clear_confirm_row:
                        gr.Markdown("⚠️ **Are you sure you want to clear the active session audit log?**")
                        confirm_yes_btn = gr.Button("Yes, Clear Audit Log", variant="primary", size="sm")
                        confirm_no_btn = gr.Button("Cancel", variant="secondary", size="sm")

                # -------------------------------------------------------------
                # VIEW 2: BATCH ANALYSIS
                # -------------------------------------------------------------
                with gr.Column(visible=False) as view_batch_analysis:
                    gr.HTML("""
                    <div class="section-tag">BATCH ENGINE</div>
                    <h3 class="section-title">Batch Video Analysis</h3>
                    <div style="font-size:0.86rem; color:#737B73; margin-bottom:16px;">
                        Evaluate entire folders of egocentric video recordings sequentially on CPU with comprehensive summary reporting.
                    </div>
                    """)
                    batch_dir_input = gr.Textbox(
                        value="data/raw/carving",
                        label="Video Folder Path (Relative or Absolute)",
                        placeholder="e.g. data/raw/carving or data/raw/normal",
                    )
                    run_batch_btn = gr.Button("⚡ Run Batch Analysis", variant="primary")
                    batch_summary_html = gr.HTML("""
                    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px; margin-top:14px; color:#737B73; font-size:0.88rem;">
                        No batch analysis executed yet. Specify a video directory above and click Run Batch Analysis.
                    </div>
                    """)
                    batch_table = gr.Dataframe(
                        headers=["Video File", "Predicted Action", "Confidence", "Risk Score", "Verdict", "Latency (ms)"],
                        interactive=False,
                        label="Batch Predictions Output",
                    )
                    with gr.Row():
                        batch_csv_btn = gr.Button("📥 Export Batch CSV", variant="secondary")
                        batch_json_btn = gr.Button("📥 Export Batch JSON", variant="secondary")
                        batch_file_download = gr.File(label="Batch Report Download", interactive=False)

                # -------------------------------------------------------------
                # VIEW 3: MODEL INFORMATION
                # -------------------------------------------------------------
                with gr.Column(visible=False) as view_model_info:
                    gr.HTML("""
                    <div class="section-tag">SYSTEM SPECIFICATION</div>
                    <h3 class="section-title">Model Architecture & Scientific Specifications</h3>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:16px; font-family:'DM Sans', sans-serif;">
                        <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px;">
                            <div style="font-family:'Lora', serif; font-size:1.05rem; font-weight:700; color:#29372F; margin-bottom:8px;">Visual Encoder Backbone</div>
                            <div style="font-size:0.84rem; color:#737B73; line-height:1.5;">
                                <strong>MobileNetV3-Small:</strong> Lightweight visual feature extractor optimized for CPU vectorized SIMD execution.
                                Extracts 512-dimensional spatial representations from 224×224 normalized RGB frame samples.
                            </div>
                        </div>
                        <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px;">
                            <div style="font-family:'Lora', serif; font-size:1.05rem; font-weight:700; color:#29372F; margin-bottom:8px;">Temporal Action Model</div>
                            <div style="font-size:0.84rem; color:#737B73; line-height:1.5;">
                                <strong>1D Temporal Convolutional Network (TCN):</strong> Multi-scale causal temporal convolutions across 8 ordered frame embeddings with attentive pooling to locate peak salient motion.
                            </div>
                        </div>
                    </div>
                    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px; margin-top:16px; font-family:'DM Sans', sans-serif;">
                        <div style="font-family:'Lora', serif; font-size:1.05rem; font-weight:700; color:#29372F; margin-bottom:8px;">Mathematical Risk Scoring</div>
                        <div style="font-size:0.84rem; color:#737B73; line-height:1.5;">
                            Confidence vs. Operational Risk:<br>
                            • <em>Confidence:</em> P(y = c | X) measures internal statistical certainty over the training taxonomy.<br>
                            • <em>Continuous Risk Score:</em> Risk(X) = Σ [ P(y = c | X) × W_c ], where W_c ∈ [0, 1] represents domain physical alteration severity.
                        </div>
                    </div>
                    """)

                # -------------------------------------------------------------
                # VIEW 4: SETTINGS
                # -------------------------------------------------------------
                with gr.Column(visible=False) as view_settings:
                    gr.HTML("""
                    <div class="section-tag">CONFIGURATION</div>
                    <h3 class="section-title">System Settings & Defaults</h3>
                    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:20px; font-family:'DM Sans', sans-serif; font-size:0.86rem; color:#29372F; line-height:1.6;">
                        <div style="margin-bottom:12px;"><strong>Default Review Threshold:</strong> 0.35 (Muted Amber — Flags clips for human verification)</div>
                        <div style="margin-bottom:12px;"><strong>Default Alert Threshold:</strong> 0.70 (Muted Terracotta — Potential physical vandalism trigger)</div>
                        <div style="margin-bottom:12px;"><strong>Default Confidence Cutoff:</strong> 0.40 (Below which uncertainty is flagged)</div>
                        <div style="margin-bottom:12px;"><strong>Frame Sampling Strategy:</strong> 8-frame uniform temporal extraction across clip duration</div>
                        <div style="margin-bottom:12px;"><strong>Preprocessed Resolution:</strong> 224 × 224 pixels (ImageNet mean & std normalized)</div>
                        <div><strong>Execution Environment:</strong> Local CPU (AVX2 / PyTorch MKL single/multi-thread)</div>
                    </div>
                    """)

                # -------------------------------------------------------------
                # VIEW 5: HELP & ABOUT
                # -------------------------------------------------------------
                with gr.Column(visible=False) as view_help_about:
                    gr.HTML("""
                    <div class="section-tag">DOCUMENTATION</div>
                    <h3 class="section-title">Help, Operational Guidelines & Ethics</h3>
                    <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:20px; font-family:'DM Sans', sans-serif; font-size:0.86rem; color:#29372F; line-height:1.6;">
                        <div style="font-family:'Lora', serif; font-size:1.05rem; font-weight:700; color:#284B38; margin-bottom:8px;">Standard Operating Procedure (SOP)</div>
                        <ol style="padding-left:20px; color:#737B73; margin-bottom:16px;">
                            <li>Upload or stream egocentric footage from site security or visitor patrols.</li>
                            <li>Observe the operational verdict card and continuous risk score meter.</li>
                            <li>If <strong>UNCERTAIN / HUMAN REVIEW</strong> or <strong>POTENTIAL VANDALISM</strong> is flagged, inspect the 8-frame gallery for peak motion frames.</li>
                            <li>Record official officer determination and notes in Section F before taking field action.</li>
                        </ol>
                        <div style="font-family:'Lora', serif; font-size:1.05rem; font-weight:700; color:#284B38; margin-bottom:8px;">Ethical AI Statement</div>
                        <p style="color:#737B73; margin:0;">
                            This AI system analyzes visual gesture dynamics. It does <strong>not</strong> identify biometric identities, does not retain video after session export, and cannot prove intentional vandalism in isolation. Authorized human review is mandatory.
                        </p>
                    </div>
                    """)

        # =================================================================
        # Section G — Footer
        # =================================================================
        with gr.Row():
            with gr.Column(scale=12):
                gr.HTML("""
                <footer style="margin-top:28px; padding:18px 24px; background:#FFFFFF; border:1px solid #E7E4DC; border-radius:14px; font-size:0.78rem; color:#737B73; text-align:center; box-shadow:0 1px 3px rgba(41,55,47,0.03); font-family:'DM Sans', sans-serif;">
                    <div style="font-family:'Lora', serif; font-weight:700; color:#29372F; font-size:0.95rem; margin-bottom:4px;">
                        Guarding the Stones — Egocentric Heritage Protection AI
                    </div>
                    <div style="margin-bottom:6px;">
                        Architectural Monolith Preservation System • CPU Engine v1.0.0-cpu • PyTorch MobileNetV3 + 1D TCN
                    </div>
                    <div style="max-width:850px; margin:0 auto; line-height:1.45; font-size:0.74rem;">
                        Complies with the <em>Digital Personal Data Protection (DPDP) Act 2023</em> and UNESCO cultural preservation protocols.
                        Video streams are processed transiently in volatile CPU memory; biometric profiles are never retained.
                        Automated predictions reflect movement patterns and require human review.
                    </div>
                </footer>
                """)

        # =================================================================
        # Event Callbacks & Navigation Wiring
        # =================================================================

        # View switching logic
        def switch_view(active_view: str):
            return (
                gr.update(visible=(active_view == "video")),
                gr.update(visible=(active_view == "batch")),
                gr.update(visible=(active_view == "model")),
                gr.update(visible=(active_view == "settings")),
                gr.update(visible=(active_view == "help")),
            )

        views_list = [view_video_analysis, view_batch_analysis, view_model_info, view_settings, view_help_about]
        nav_btn_video.click(lambda: switch_view("video"), None, views_list)
        nav_btn_batch.click(lambda: switch_view("batch"), None, views_list)
        nav_btn_model.click(lambda: switch_view("model"), None, views_list)
        nav_btn_settings.click(lambda: switch_view("settings"), None, views_list)
        nav_btn_help.click(lambda: switch_view("help"), None, views_list)

        # Video analysis handler
        def on_analyze(video_file, rev_thr, al_thr, conf_thr, history_list, show_ts):
            history_list = history_list or []
            if video_file is None:
                err_res = PredictionResult(
                    video_path="",
                    status="error",
                    predicted_label="unknown",
                    confidence=0.0,
                    risk_score=0.0,
                    action_category="unknown",
                    verdict="ERROR",
                    uncertainty_flag=True,
                    probabilities={},
                    peak_evidence_timestamp_sec=None,
                    peak_evidence_frame_idx=None,
                    sampled_frames=[],
                    video_duration_sec=0.0,
                    video_fps=0.0,
                    video_total_frames=0,
                    decode_and_preprocess_ms=0.0,
                    inference_ms=0.0,
                    total_latency_ms=0.0,
                    error_type="NO_VIDEO_PROVIDED",
                    error_message="Please upload a video file or capture webcam input before analyzing.",
                )
                err_card = render_verdict_card(err_res, rev_thr, al_thr, conf_thr)
                info_card = render_video_info_card(None, None)
                return err_card, None, [], info_card, "", "", None, None, history_list, history_list, []

            video_p = Path(video_file).resolve()
            engine.review_threshold = float(rev_thr)
            engine.alert_threshold = float(al_thr)
            engine.confidence_threshold = float(conf_thr)

            # Metadata extraction
            try:
                meta = extract_video_metadata(video_p)
            except Exception:
                meta = None

            # Core prediction with sampled RGB frames for gallery
            res, raw_gallery = engine.predict_video_with_frames(video_p)

            # Update verdict and performance displays
            v_card = render_verdict_card(res, rev_thr, al_thr, conf_thr)
            info_card = render_video_info_card(res, meta)
            p_card = render_performance_card(res)
            t_card = render_timeline_summary(res)

            if res.status != "success":
                return v_card, None, [], info_card, p_card, t_card, res, meta, history_list, history_list, []

            # Format gallery with optional timestamp toggle
            formatted_gallery = []
            if raw_gallery:
                for idx, (frame_rgb, orig_caption) in enumerate(raw_gallery):
                    f_ev = res.sampled_frames[idx]
                    peak_str = " ★ [PEAK EVIDENCE]" if f_ev.is_peak_evidence else ""
                    if show_ts:
                        caption = f"Sample {idx+1}/{len(raw_gallery)} | Frame {f_ev.frame_index} | {f_ev.timestamp_sec:.2f}s | Attn: {f_ev.attention_weight:.3f}{peak_str}"
                    else:
                        caption = f"Sample {idx+1}/{len(raw_gallery)} | Frame {f_ev.frame_index} | Attn: {f_ev.attention_weight:.3f}{peak_str}"
                    formatted_gallery.append((frame_rgb, caption))

            # Label probabilities
            top_probs = res.probabilities

            # Append to session history
            now_str = datetime.now().strftime("%H:%M:%S")
            row = [
                now_str,
                video_p.name,
                res.predicted_label,
                f"{res.confidence * 100:.1f}%",
                f"{res.risk_score:.4f}",
                res.verdict,
                f"{res.total_latency_ms:.1f}",
                "Pending Review",
            ]
            history_list = [row] + history_list

            session_records.append({
                "timestamp": now_str,
                "video": video_p.name,
                "prediction": res.to_dict(),
                "officer_review": "Pending Review",
                "officer_notes": "",
            })

            return v_card, top_probs, formatted_gallery, info_card, p_card, t_card, res, meta, history_list, history_list, (raw_gallery or [])

        # Live slider change handler
        def on_slider_change(current_res, rev_thr, al_thr, conf_thr):
            """Instantly update verdict when slider moves without re-executing inference."""
            return render_verdict_card(current_res, rev_thr, al_thr, conf_thr)

        # Timestamp visibility toggle handler
        def on_toggle_timestamps(show_ts, current_res, raw_gallery):
            if not current_res or not raw_gallery or current_res.status != "success":
                return []
            formatted_gallery = []
            for idx, (frame_rgb, _) in enumerate(raw_gallery):
                f_ev = current_res.sampled_frames[idx]
                peak_str = " ★ [PEAK EVIDENCE]" if f_ev.is_peak_evidence else ""
                if show_ts:
                    caption = f"Sample {idx+1}/{len(raw_gallery)} | Frame {f_ev.frame_index} | {f_ev.timestamp_sec:.2f}s | Attn: {f_ev.attention_weight:.3f}{peak_str}"
                else:
                    caption = f"Sample {idx+1}/{len(raw_gallery)} | Frame {f_ev.frame_index} | Attn: {f_ev.attention_weight:.3f}{peak_str}"
                formatted_gallery.append((frame_rgb, caption))
            return formatted_gallery

        # Officer review sign-off handler
        def on_record_officer_review(decision, notes, current_res, history_list):
            if not history_list or current_res is None or current_res.status != "success":
                return history_list, history_list, "⚠️ *No active video prediction to sign off.*"

            latest_row = list(history_list[0])
            latest_row[7] = decision
            history_list[0] = latest_row

            if session_records:
                session_records[-1]["officer_review"] = decision
                session_records[-1]["officer_notes"] = notes

            return history_list, history_list, f"✅ *Officer review recorded: **{decision}***"

        # Clear workspace handler
        def on_clear():
            return (
                None,
                render_verdict_card(None, 0.35, 0.70, 0.40),
                None,
                [],
                render_video_info_card(None, None),
                "",
                "",
                None,
                None,
                [],
                "Pending Review",
                "",
                "",
                None,
            )

        # Clear History confirmation handlers
        def on_show_clear_confirm():
            return gr.update(visible=True)

        def on_cancel_clear_confirm():
            return gr.update(visible=False)

        def on_confirm_clear_history():
            session_records.clear()
            return [], [], gr.update(visible=False)

        # Export handlers
        def on_export_csv(history_list):
            if not history_list:
                return None
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="session_log_")
            import csv
            with open(tmp.name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(HIST_HEADERS)
                for row in history_list:
                    writer.writerow(row)
            return tmp.name

        def on_export_json():
            if not session_records:
                return None
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="session_audit_")
            with open(tmp.name, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "export_time": datetime.now().isoformat(),
                        "engine_version": MODEL_ENGINE_VERSION,
                        "session_evaluations": session_records,
                    },
                    f,
                    indent=2,
                )
            return tmp.name

        def on_reset_thresholds():
            return 0.35, 0.70, 0.40

        # Sample video loader handlers
        def on_load_sample(path: Path):
            if path.exists():
                return str(path)
            return None

        # Batch prediction handler
        def on_run_batch(dir_path):
            p = Path(dir_path).resolve()
            if not p.exists() or not p.is_dir():
                err_html = f"<div style='color:#B85E42; font-weight:600;'>Error: Directory not found: {p}</div>"
                return err_html, []

            video_files = [
                f for f in p.glob("*") if f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
            ][:10]  # Limit to 10 for responsive CPU demo

            if not video_files:
                err_html = f"<div style='color:#D18A18; font-weight:600;'>No supported video files found in {p}</div>"
                return err_html, []

            batch_rep = engine.predict_batch(video_files)
            
            summary_html = f"""
            <div style="background:#FFFFFF; border:1px solid #E7E4DC; border-radius:12px; padding:18px; font-family:'DM Sans', sans-serif;">
                <div style="font-family:'Lora', serif; font-size:1.1rem; font-weight:700; color:#29372F; margin-bottom:10px;">
                    Batch Analysis Summary ({batch_rep.total_videos} Videos)
                </div>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; text-align:center;">
                    <div style="background:#E8EFE5; border:1px solid #C8DAC4; border-radius:8px; padding:10px;">
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#284B38; font-weight:700;">Normal</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#284B38;">{batch_rep.verdict_counts.get('NORMAL', 0)}</div>
                    </div>
                    <div style="background:#FFF2DC; border:1px solid #F5DEB3; border-radius:8px; padding:10px;">
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#D18A18; font-weight:700;">Review Needed</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#D18A18;">{batch_rep.verdict_counts.get('UNCERTAIN / HUMAN REVIEW', 0)}</div>
                    </div>
                    <div style="background:#FBE9E4; border:1px solid #F7D0C6; border-radius:8px; padding:10px;">
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#B85E42; font-weight:700;">Potential Damage</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#B85E42;">{batch_rep.verdict_counts.get('POTENTIAL VANDALISM', 0)}</div>
                    </div>
                    <div style="background:#FAF9F5; border:1px solid #E7E4DC; border-radius:8px; padding:10px;">
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#737B73; font-weight:700;">Mean Latency</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#29372F;">{batch_rep.mean_latency_ms:.1f} ms</div>
                    </div>
                </div>
            </div>
            """

            rows = []
            for item in batch_rep.results:
                rows.append([
                    Path(item['video_path']).name,
                    item['predicted_label'],
                    f"{item['confidence'] * 100:.1f}%",
                    f"{item['risk_score']:.4f}",
                    item['verdict'],
                    f"{item['total_latency_ms']:.1f}",
                ])

            return summary_html, rows

        # Connect event triggers
        analyze_btn.click(
            on_analyze,
            inputs=[video_input, review_slider, alert_slider, conf_slider, session_history_state, show_timestamps_chk],
            outputs=[
                verdict_html,
                label_output,
                gallery_output,
                video_info_html,
                perf_html,
                timeline_html,
                current_result_state,
                current_meta_state,
                session_history_state,
                history_table,
                raw_gallery_state,
            ],
        )

        for slider in (review_slider, alert_slider, conf_slider):
            slider.change(
                on_slider_change,
                inputs=[current_result_state, review_slider, alert_slider, conf_slider],
                outputs=[verdict_html],
            )

        show_timestamps_chk.change(
            on_toggle_timestamps,
            inputs=[show_timestamps_chk, current_result_state, raw_gallery_state],
            outputs=[gallery_output],
        )

        reset_thresholds_btn.click(
            on_reset_thresholds,
            inputs=None,
            outputs=[review_slider, alert_slider, conf_slider],
        )

        # Sample video buttons
        sample_carving_btn.click(lambda: on_load_sample(sample_carving), None, [video_input])
        sample_photo_btn.click(lambda: on_load_sample(sample_photo), None, [video_input])
        sample_walk_btn.click(lambda: on_load_sample(sample_walk), None, [video_input])
        sample_touch_btn.click(lambda: on_load_sample(sample_touch), None, [video_input])

        # Officer review submission
        submit_review_btn.click(
            on_record_officer_review,
            inputs=[officer_decision, officer_notes, current_result_state, session_history_state],
            outputs=[session_history_state, history_table, review_status_msg],
        )

        # Clear buttons
        clear_btn.click(
            on_clear,
            inputs=None,
            outputs=[
                video_input,
                verdict_html,
                label_output,
                gallery_output,
                video_info_html,
                perf_html,
                timeline_html,
                current_result_state,
                current_meta_state,
                session_history_state,
                officer_decision,
                officer_notes,
                review_status_msg,
                file_download,
            ],
        )

        # Clear History modal confirmation
        clear_history_btn.click(on_show_clear_confirm, None, [clear_confirm_row])
        confirm_no_btn.click(on_cancel_clear_confirm, None, [clear_confirm_row])
        confirm_yes_btn.click(on_confirm_clear_history, None, [session_history_state, history_table, clear_confirm_row])

        # Export session logs
        export_csv_btn.click(on_export_csv, inputs=[session_history_state], outputs=[file_download])
        export_json_btn.click(on_export_json, inputs=None, outputs=[file_download])

        # Batch analysis
        run_batch_btn.click(on_run_batch, inputs=[batch_dir_input], outputs=[batch_summary_html, batch_table])

    return app


if __name__ == "__main__":
    app = create_dashboard()
    app.launch(server_port=7860)
