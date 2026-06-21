from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import webrtc_streamer
except ImportError:
    av = None
    webrtc_streamer = None

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nestdetect.inference import predict_bgr_frame, predict_image

st.set_page_config(
    page_title="NestDetect Research Console",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_MODEL = ROOT / "models" / "nestdetect_hope_cms_only_v4.pt"
CMS_RESULTS = ROOT / "results" / "cms"
TARGET_CLASS_IDS = [0, 56, 60, 63, 73]
CLASS_ALIASES = {"dining table": "table"}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --nd-navy: #07101f;
            --nd-blue: #60a5fa;
            --nd-cyan: #22d3ee;
            --nd-green: #34d399;
            --nd-text: #f1f5f9;
            --nd-slate: #c3d0e2;
            --nd-muted: #8fa2bb;
            --nd-border: #263a55;
            --nd-surface: #101c2f;
            --nd-panel: #0d1829;
            --nd-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
        }

        .stApp {
            background:
                radial-gradient(circle at 7% 3%, rgba(34, 211, 238, 0.13), transparent 27rem),
                radial-gradient(circle at 94% 0%, rgba(59, 130, 246, 0.2), transparent 30rem),
                linear-gradient(145deg, #050b14 0%, #081221 48%, #0a1526 100%);
            color: var(--nd-text);
            min-height: 100vh;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 10% 0%, rgba(34, 211, 238, 0.16), transparent 18rem),
                linear-gradient(180deg, #07101e 0%, #0a1729 100%);
            border-right: 1px solid #1e3048;
            box-shadow: 8px 0 34px rgba(0, 0, 0, 0.32);
        }

        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border: 1px solid transparent;
            border-radius: 0.55rem;
            padding: 0.55rem 0.7rem;
            transition: all 120ms ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(96, 165, 250, 0.11);
            border-color: rgba(96, 165, 250, 0.22);
        }

        .block-container {
            background: rgba(10, 22, 38, 0.94);
            border: 1px solid #223651;
            border-radius: 1.25rem;
            box-shadow: var(--nd-shadow);
            margin-bottom: 2rem;
            margin-top: 1.15rem;
            max-width: 1320px;
            padding: 2.4rem clamp(1.4rem, 3vw, 3.2rem) 3.5rem;
        }

        .nd-brand {
            padding: 0.5rem 0 1.25rem;
        }

        .nd-brand-title {
            color: #ffffff;
            font-size: 1.4rem;
            font-weight: 760;
            letter-spacing: -0.03em;
            margin: 0;
        }

        .nd-brand-subtitle {
            color: var(--nd-muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .nd-eyebrow {
            color: var(--nd-cyan);
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .nd-hero,
        .nd-page-header {
            background:
                radial-gradient(circle at 92% 12%, rgba(34, 211, 238, 0.13), transparent 17rem),
                linear-gradient(125deg, #101f35, #0d1a2d);
            border: 1px solid #2b4565;
            border-radius: 1rem;
            overflow: hidden;
            padding: clamp(1.35rem, 3vw, 2.2rem);
            position: relative;
        }

        .nd-hero::after,
        .nd-page-header::after {
            background: linear-gradient(180deg, #60a5fa, #22d3ee);
            border-radius: 999px;
            content: "";
            height: 7rem;
            opacity: 0.13;
            position: absolute;
            right: -2rem;
            top: -2.5rem;
            transform: rotate(28deg);
            width: 13rem;
        }

        .nd-title {
            color: var(--nd-text);
            font-size: clamp(2.15rem, 5vw, 4rem);
            font-weight: 790;
            letter-spacing: -0.055em;
            line-height: 1.04;
            margin: 0;
            max-width: 900px;
            position: relative;
            z-index: 1;
        }

        .nd-page-title {
            color: var(--nd-text);
            font-size: 2.45rem;
            font-weight: 780;
            letter-spacing: -0.045em;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        .nd-lead {
            color: var(--nd-slate);
            font-size: 1.05rem;
            line-height: 1.75;
            margin: 1rem 0 0;
            max-width: 820px;
            position: relative;
            z-index: 1;
        }

        .nd-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.35rem;
            position: relative;
            z-index: 1;
        }

        .nd-badge {
            background: rgba(37, 99, 235, 0.16);
            border: 1px solid rgba(96, 165, 250, 0.35);
            border-radius: 999px;
            color: #bfdbfe;
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.35rem 0.7rem;
        }

        .nd-section {
            margin-top: 2.4rem;
        }

        .nd-section h2 {
            color: var(--nd-text);
            font-size: 1.45rem;
            font-weight: 740;
            letter-spacing: -0.025em;
            margin-bottom: 0.35rem;
        }

        .nd-section p {
            color: var(--nd-slate);
            line-height: 1.65;
        }

        .nd-card {
            background: linear-gradient(155deg, #132238 0%, #0f1c30 100%);
            border: 1px solid var(--nd-border);
            border-radius: 0.9rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            min-height: 100%;
            padding: 1.2rem 1.25rem;
            transition: border-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
        }

        .nd-card:hover {
            border-color: #3d638c;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32);
            transform: translateY(-2px);
        }

        .nd-card-label {
            color: #8ea4c0;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .nd-card-title {
            color: var(--nd-text);
            font-size: 1.02rem;
            font-weight: 720;
            margin-top: 0.4rem;
        }

        .nd-card-copy {
            color: var(--nd-slate);
            font-size: 0.9rem;
            line-height: 1.55;
            margin-top: 0.45rem;
        }

        .nd-model-strip {
            background:
                radial-gradient(circle at 90% 0%, rgba(34, 211, 238, 0.19), transparent 16rem),
                linear-gradient(120deg, #102341, #12345d);
            border: 1px solid #2d527e;
            border-radius: 0.9rem;
            box-shadow: 0 12px 28px rgba(13, 42, 78, 0.2);
            color: #ffffff;
            margin: 1.5rem 0 1.1rem;
            padding: 1.1rem 1.25rem;
        }

        .nd-model-strip strong {
            display: block;
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }

        .nd-model-strip span {
            color: #bfdbfe;
            font-size: 0.86rem;
        }

        .nd-note {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.38), rgba(15, 62, 69, 0.42));
            border: 1px solid rgba(52, 211, 153, 0.38);
            border-left: 4px solid var(--nd-green);
            border-radius: 0.65rem;
            color: #c7f9e6;
            line-height: 1.6;
            margin-top: 1rem;
            padding: 0.85rem 1rem;
        }

        .nd-footer {
            border-top: 1px solid var(--nd-border);
            color: var(--nd-muted);
            font-size: 0.78rem;
            margin-top: 3rem;
            padding-top: 1.2rem;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #14243b, #0e1b2e);
            border: 1px solid #29415f;
            border-radius: 0.8rem;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.2);
            min-height: 7.1rem;
            padding: 1rem 1.05rem;
        }

        [data-testid="stMetricLabel"] {
            color: #9bacc2;
        }

        [data-testid="stMetricValue"] {
            color: var(--nd-text);
            font-weight: 750;
        }

        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] .stMarkdown,
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {
            color: var(--nd-slate);
        }

        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4 {
            color: var(--nd-text);
        }

        .stButton > button {
            background: linear-gradient(120deg, #2563eb, #0891b2);
            border: 1px solid #3b82f6;
            border-radius: 0.55rem;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
            color: #ffffff;
            font-weight: 680;
            min-height: 2.75rem;
        }

        .stButton > button:hover {
            background: linear-gradient(120deg, #3b82f6, #06b6d4);
            border-color: #60a5fa;
            color: #ffffff;
        }

        [data-testid="stFileUploader"] {
            background: #101d31;
            border: 1px solid #2a415e;
            border-radius: 0.75rem;
            padding: 0.35rem;
        }

        [data-testid="stImage"] img {
            border: 1px solid #304b6b;
            border-radius: 0.8rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.3);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #2c4564;
            border-radius: 0.75rem;
            overflow: hidden;
        }

        [data-testid="stAlert"] {
            border-radius: 0.7rem;
        }

        [data-testid="stSlider"] > div,
        [data-testid="stRadio"] > div {
            background: rgba(15, 28, 47, 0.88);
            border: 1px solid #2a405e;
            border-radius: 0.7rem;
            padding: 0.7rem 0.8rem;
        }

        [data-testid="stMain"] input,
        [data-testid="stMain"] textarea,
        [data-baseweb="select"] > div {
            background: #101d31 !important;
            border-color: #2a415e !important;
            color: var(--nd-text) !important;
        }

        [data-testid="stMain"] code {
            background: #07111f;
            color: #c7d8ef;
        }

        [data-testid="stCodeBlock"] {
            border: 1px solid #243c59;
            border-radius: 0.75rem;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            gap: 1rem;
        }

        @media (max-width: 1024px) {
            .block-container {
                border-radius: 1rem;
                margin-left: 0.75rem;
                margin-right: 0.75rem;
                padding: 2rem 1.5rem 3rem;
                width: calc(100% - 1.5rem);
            }

            .nd-title {
                font-size: clamp(2.3rem, 7vw, 3.5rem);
            }
        }

        @media (max-width: 768px) {
            .block-container {
                border-radius: 0.8rem;
                margin: 0.5rem;
                padding: 1.35rem 1rem 2.4rem;
                width: calc(100% - 1rem);
            }

            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0.8rem !important;
            }

            [data-testid="column"],
            [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }

            .nd-title {
                font-size: 2.25rem;
                letter-spacing: -0.045em;
            }

            .nd-page-title {
                font-size: 1.9rem;
            }

            .nd-lead {
                font-size: 0.96rem;
                line-height: 1.6;
            }

            .nd-hero,
            .nd-page-header {
                padding: 1.25rem 1rem;
            }

            .nd-model-strip {
                padding: 1rem;
            }

            [data-testid="stMetric"] {
                min-height: auto;
            }

            [data-testid="stSidebar"] {
                min-width: min(84vw, 19rem);
            }
        }

        @media (max-width: 480px) {
            .block-container {
                border-left: 0;
                border-radius: 0;
                border-right: 0;
                margin: 0;
                padding: 1.1rem 0.8rem 2rem;
                width: 100%;
            }

            .nd-title {
                font-size: 1.95rem;
            }

            .nd-page-title {
                font-size: 1.7rem;
            }

            .nd-badges {
                gap: 0.4rem;
            }

            .nd-badge {
                font-size: 0.7rem;
            }

            .nd-card {
                padding: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def model_metadata() -> dict[str, Any]:
    return read_json(APP_MODEL.with_suffix(".metadata.json"))


def cms_comparison() -> pd.DataFrame:
    path = CMS_RESULTS / "comparison.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def cms_v4_metrics() -> dict[str, float]:
    frame = cms_comparison()
    if frame.empty:
        return {}
    row = frame[frame["strategy"] == "cms_v4"]
    if row.empty:
        return {}
    record = row.iloc[0]
    return {
        "old_map50_95": float(record["old_map50_95"]),
        "new_map50_95": float(record["new_map50_95"]),
        "forgetting": float(record["forgetting"]),
    }


def prediction_options() -> dict[str, Any]:
    return {
        "classes": TARGET_CLASS_IDS,
        "class_aliases": CLASS_ALIASES,
    }


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="nd-page-header">
            <div class="nd-eyebrow">{eyebrow}</div>
            <h1 class="nd-page-title">{title}</h1>
            <p class="nd-lead">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    metadata = model_metadata()
    st.sidebar.markdown(
        """
        <div class="nd-brand">
            <p class="nd-brand-title">NestDetect</p>
            <p class="nd-brand-subtitle">Continual object detection research</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Test Prediction", "Research Results", "Reproducibility"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("ACTIVE PREDICTION MODEL")
    st.sidebar.markdown("**HoPe CMS V4**")
    st.sidebar.caption("Replay-free routed detector memories")
    st.sidebar.caption(f"Checkpoint: `{APP_MODEL.name}`")
    if metadata.get("sha256"):
        st.sidebar.caption(f"SHA-256: `{metadata['sha256'][:12]}…`")
    status = "Ready" if APP_MODEL.is_file() else "Checkpoint missing"
    if APP_MODEL.is_file():
        st.sidebar.success(status)
    else:
        st.sidebar.error(status)
    return page


def overview_page() -> None:
    metrics = cms_v4_metrics()
    st.markdown(
        """
        <div class="nd-hero">
            <div class="nd-eyebrow">Research console</div>
            <h1 class="nd-title">Continual object detection without raw-data replay.</h1>
            <p class="nd-lead">
                NestDetect evaluates how YOLO11n-HoPe learns laptop and book after
                person, chair, and dining table. This console uses CMS V4, a
                replay-free model that routes old and new classes through separate
                detector memories.
            </p>
            <div class="nd-badges">
                <span class="nd-badge">YOLO11n-HoPe</span>
                <span class="nd-badge">CMS V4</span>
                <span class="nd-badge">Replay-free checkpoint</span>
                <span class="nd-badge">COCO80 head preserved</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nd-section"><h2>CMS V4 evaluation</h2></div>', unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("Old-class mAP50-95", f"{metrics.get('old_map50_95', 0.0):.4f}")
    columns[1].metric("New-class mAP50-95", f"{metrics.get('new_map50_95', 0.0):.4f}")
    columns[2].metric("Forgetting", f"{metrics.get('forgetting', 0.0):.5f}")
    columns[3].metric("Target classes", "5")

    st.markdown(
        """
        <div class="nd-note">
            CMS V4 retains the HoPe base-task score while learning the two new
            classes without storing or replaying old-task images. The result is
            specific to this controlled COCO subset and one incremental stage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nd-section"><h2>How the model works</h2></div>', unsafe_allow_html=True)
    cards = st.columns(3)
    cards[0].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">Persistent memory</div>
            <div class="nd-card-title">Old-class detector</div>
            <div class="nd-card-copy">
                An immutable base HoPe detector handles person, chair, and dining
                table.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">Plastic memory</div>
            <div class="nd-card-title">New-class specialist</div>
            <div class="nd-card-copy">
                A detector trained without old-task replay handles laptop and
                book.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">Class router</div>
            <div class="nd-card-title">Prediction composition</div>
            <div class="nd-card-copy">
                A fixed class mask selects each detector memory while preserving
                its own localization and classification pathway.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nd-section"><h2>Research scope</h2></div>', unsafe_allow_html=True)
    scope = st.columns(2)
    scope[0].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">Supported conclusion</div>
            <div class="nd-card-title">Near-zero forgetting in this protocol</div>
            <div class="nd-card-copy">
                Routed detector memories preserve old-task accuracy while
                recovering new-class capability on the curated validation set.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    scope[1].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">Important limitation</div>
            <div class="nd-card-title">Not a universal continual learner</div>
            <div class="nd-card-copy">
                Routing uses known class IDs, only one incremental transition is
                tested, and parameter storage grows with independent memories.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_summary(detections: list[dict[str, Any]]) -> None:
    counts = Counter(item["object"] for item in detections)
    columns = st.columns(3)
    columns[0].metric("Objects detected", str(len(detections)))
    columns[1].metric("Classes present", str(len(counts)))
    best = max((float(item["confidence"]) for item in detections), default=0.0)
    columns[2].metric("Highest confidence", f"{best:.1%}")

    frame = pd.DataFrame(detections)
    if frame.empty:
        return
    frame = frame.rename(
        columns={
            "no": "#",
            "object": "Class",
            "confidence": "Confidence",
            "class_id": "COCO ID",
            "x1": "X1",
            "y1": "Y1",
            "x2": "X2",
            "y2": "Y2",
        }
    )
    frame["Confidence"] = frame["Confidence"].map(lambda value: f"{float(value):.2%}")
    ordered = ["#", "Class", "Confidence", "COCO ID", "X1", "Y1", "X2", "Y2"]
    st.dataframe(frame[ordered], width="stretch", hide_index=True)


def uploaded_image_prediction(
    confidence: float,
    iou: float,
    imgsz: int,
) -> None:
    uploaded = st.file_uploader(
        "Upload a JPG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
        help="The image is processed locally by the selected checkpoint.",
    )
    if uploaded is None:
        st.info("Upload an image to begin a CMS V4 prediction test.")
        return

    image = Image.open(uploaded).convert("RGB")
    run_prediction = st.button(
        "Run CMS V4 prediction",
        type="primary",
        width="stretch",
    )
    input_column, output_column = st.columns(2, gap="medium")
    input_column.image(image, caption="Input image", width="stretch")
    if not run_prediction:
        output_column.info("The CMS V4 result will appear here.")
        return

    with st.spinner("Running CMS V4 inference..."):
        annotated, detections = predict_image(
            APP_MODEL,
            image,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            **prediction_options(),
        )

    st.success("Prediction completed with the replay-free CMS V4 checkpoint.")
    output_column.image(
        annotated,
        caption="CMS V4 detection result",
        width="stretch",
    )
    if detections:
        st.markdown("#### Detection summary")
        prediction_summary(detections)
    else:
        st.warning(
            "No target object passed the selected threshold. Try a lower confidence "
            "threshold or an image containing a person, chair, table, laptop, or book."
        )


def realtime_prediction(confidence: float, iou: float, imgsz: int) -> None:
    if av is None or webrtc_streamer is None:
        st.error(
            "Webcam dependencies are unavailable. Install the project with "
            "`python -m pip install -e .` and restart the application."
        )
        return

    st.info(
        "Select START and allow browser camera access. Remote deployments require "
        "HTTPS for camera permission."
    )

    def video_frame_callback(frame):
        image = frame.to_ndarray(format="bgr24")
        annotated, _ = predict_bgr_frame(
            APP_MODEL,
            image,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            **prediction_options(),
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key=f"nestdetect-cms-v4-{confidence:.2f}-{iou:.2f}-{imgsz}",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20},
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )


def prediction_page() -> None:
    page_header(
        "Inference workspace",
        "Test Prediction",
        "Run image or webcam inference using the replay-free HoPe CMS V4 checkpoint.",
    )
    metadata = model_metadata()
    st.markdown(
        f"""
        <div class="nd-model-strip">
            <strong>Active model: {metadata.get("name", "NestDetect HoPe CMS V4")}</strong>
            <span>
                {metadata.get("provenance", "Replay-free routed detector memories")}
                · {APP_MODEL.name}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not APP_MODEL.is_file():
        st.error(f"Required checkpoint not found: `{APP_MODEL.relative_to(ROOT)}`")
        return

    controls, workspace = st.columns([0.3, 0.7], gap="large")
    with controls:
        st.markdown("#### Inference settings")
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )
        iou = st.slider(
            "NMS IoU threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.70,
            step=0.05,
        )
        imgsz = st.select_slider(
            "Inference resolution",
            options=[320, 480, 640],
            value=640,
        )
        input_mode = st.radio(
            "Input source",
            ["Upload image", "Live camera"],
        )
        st.caption(
            "Use 640 for small books or laptops. Use 320 or 480 for faster webcam inference."
        )
    with workspace:
        if input_mode == "Upload image":
            uploaded_image_prediction(confidence, iou, imgsz)
        else:
            realtime_prediction(confidence, iou, imgsz)


def display_results_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No research comparison file is available.")
        return
    display = frame.copy()
    names = {
        "base": "HoPe base",
        "no_replay": "HoPe without replay",
        "cms_v1": "CMS V1",
        "cms_v2": "CMS V2",
        "cms_v3": "CMS V3",
        "cms_v4": "CMS V4",
        "cms_v5": "CMS V5",
        "replay": "HoPe with replay",
    }
    display["Strategy"] = display["strategy"].map(names).fillna(display["strategy"])
    display["Old mAP50-95"] = display["old_map50_95"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.4f}"
    )
    display["New mAP50-95"] = display["new_map50_95"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.4f}"
    )
    display["Forgetting"] = display["forgetting"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.5f}"
    )
    st.dataframe(
        display[["Strategy", "Old mAP50-95", "New mAP50-95", "Forgetting"]],
        width="stretch",
        hide_index=True,
    )


def research_results_page() -> None:
    page_header(
        "Experimental evidence",
        "Research Results",
        "Compare replay-free CMS variants against no-replay and replay-trained HoPe models.",
    )
    metrics = cms_v4_metrics()
    columns = st.columns(4)
    columns[0].metric("CMS V4 old mAP50-95", f"{metrics.get('old_map50_95', 0.0):.4f}")
    columns[1].metric("CMS V4 new mAP50-95", f"{metrics.get('new_map50_95', 0.0):.4f}")
    columns[2].metric("CMS V4 forgetting", f"{metrics.get('forgetting', 0.0):.5f}")
    columns[3].metric("Evaluation size", "640 px")

    st.markdown('<div class="nd-section"><h2>CMS ablation comparison</h2></div>', unsafe_allow_html=True)
    display_results_table(cms_comparison())

    plot = CMS_RESULTS / "forgetting-comparison.png"
    if plot.is_file():
        st.markdown('<div class="nd-section"><h2>Retention and acquisition</h2></div>', unsafe_allow_html=True)
        st.image(plot, caption="YOLO11n-HoPe CMS ablation results", width="stretch")

    st.markdown('<div class="nd-section"><h2>Interpretation</h2></div>', unsafe_allow_html=True)
    cards = st.columns(3)
    cards[0].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">V1 and V2</div>
            <div class="nd-card-title">Plastic but unstable</div>
            <div class="nd-card-copy">
                Both variants learn the new task but retain less old-task
                performance than plain no-replay HoPe.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">V3</div>
            <div class="nd-card-title">Stable but not plastic</div>
            <div class="nd-card-copy">
                Strict parameter isolation preserves the old task but removes
                almost all new-class capability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        """
        <div class="nd-card">
            <div class="nd-card-label">V4</div>
            <div class="nd-card-title">Routed detector memories</div>
            <div class="nd-card-copy">
                Separate complete representations retain the base score while
                restoring new-class detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "These values come from a small curated COCO validation set and one seed. "
        "They should not be interpreted as production performance or a universal "
        "zero-forgetting guarantee."
    )


def reproducibility_page() -> None:
    page_header(
        "Experiment workflow",
        "Reproducibility",
        "Rebuild datasets, train the HoPe stages, construct CMS V4, and regenerate the reported metrics.",
    )
    st.markdown('<div class="nd-section"><h2>1. Prepare the environment</h2></div>', unsafe_allow_html=True)
    st.code(
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        'python -m pip install -e ".[dev]"\n'
        "pytest",
        language="bash",
    )
    st.markdown('<div class="nd-section"><h2>2. Build datasets</h2></div>', unsafe_allow_html=True)
    st.code(
        "python scripts/prepare_coco_subset.py\n"
        "python scripts/build_datasets.py",
        language="bash",
    )
    st.markdown('<div class="nd-section"><h2>3. Train HoPe</h2></div>', unsafe_allow_html=True)
    st.code(
        "nestdetect train configs/hope/base.yaml\n"
        "nestdetect train configs/hope/incremental-no-replay.yaml\n"
        "nestdetect train configs/hope/incremental-replay.yaml",
        language="bash",
    )
    st.markdown('<div class="nd-section"><h2>4. Build and evaluate CMS V4</h2></div>', unsafe_allow_html=True)
    st.code(
        "python scripts/consolidate_cms.py\n"
        "nestdetect train configs/ablations/cms-v1.yaml\n"
        "python scripts/build_cms_v4.py\n"
        "python scripts/evaluate_cms.py\n"
        "python scripts/plot_results.py --results-dir results/cms",
        language="bash",
    )
    st.info(
        "Full configuration details, expected outputs, and limitations are documented "
        "in `docs/reproducibility.md` and `docs/research-report.md`."
    )


def footer() -> None:
    st.markdown(
        """
        <div class="nd-footer">
            NestDetect · Research and demonstration software · Microsoft COCO 2017 subset
        </div>
        """,
        unsafe_allow_html=True,
    )


apply_styles()
active_page = sidebar()
pages = {
    "Overview": overview_page,
    "Test Prediction": prediction_page,
    "Research Results": research_results_page,
    "Reproducibility": reproducibility_page,
}
pages[active_page]()
footer()
