from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from road_damage.config import CLASS_NAMES, DEFAULT_MODEL_PATH
from road_damage.motion import motion_strip


st.set_page_config(page_title="Road Damage / Field Monitor", page_icon="◈", layout="wide")


@st.cache_resource(show_spinner="Loading the trained road-damage model…")
def load_model(checkpoint: str, modified_ns: int, file_size: int):
    from road_damage.model import load_road_damage_model
    return load_road_damage_model(checkpoint)


@st.cache_resource(show_spinner="Downloading generic YOLOv8n model (one-time)…")
def load_generic_model():
    from road_damage.model import load_fallback_model
    return load_fallback_model()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#11120f; --paper:#efeee6; --line:#35372e; --acid:#d5ff58; }
        html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; }
        .stApp { background:var(--ink); color:var(--paper); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#171814; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] * { color:var(--paper); }
        [data-testid="stSidebar"] code { color:var(--acid); }
        .block-container { max-width:1440px; padding-top:1.3rem; padding-bottom:4rem; }
        h1,h2,h3 { letter-spacing:-.04em; color:var(--paper); }
        h2 { font-size:1.55rem !important; }
        .hero { min-height:300px; position:relative; overflow:hidden; isolation:isolate; padding:1.5rem 2rem 1.2rem;
          margin-bottom:1rem; border:1px solid #44463b; border-radius:7px; color:var(--paper);
          background:radial-gradient(ellipse at 82% 45%,#39402a 0,transparent 36%),linear-gradient(125deg,#1c1e19,#11120f 70%);
          animation:hero-in .7s cubic-bezier(.2,.8,.2,1) both; }
        .hero:before { content:""; position:absolute; inset:0; z-index:-1; opacity:.13; pointer-events:none;
          background-image:linear-gradient(rgba(240,240,220,.2) 1px,transparent 1px),linear-gradient(90deg,rgba(240,240,220,.2) 1px,transparent 1px);
          background-size:42px 42px; mask-image:linear-gradient(90deg,transparent,#000 55%,#000); }
        .hero:after { content:""; position:absolute; z-index:-1; top:0; bottom:0; left:-25%; width:22%;
          background:linear-gradient(90deg,transparent,rgba(213,255,88,.05),transparent); transform:skewX(-22deg); animation:hero-scan 7s linear infinite; }
        .hero-top,.hero-bottom { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
        .micro,.live,.hero-bottom,.section-kicker { font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; }
        .micro { color:#a9ac9e; }
        .live { display:flex; align-items:center; gap:.5rem; color:var(--acid); }
        .live i { width:7px;height:7px;border-radius:50%;background:var(--acid);display:inline-block;box-shadow:0 0 14px var(--acid);animation:blink 1.6s ease-in-out infinite; }
        .hero-layout { display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; margin:2rem 0; }
        .hero h1 { font-size:clamp(3.3rem,8vw,7.2rem); line-height:.78; letter-spacing:-.095em; font-weight:700; margin:0; text-transform:uppercase; }
        .hero h1 em { color:var(--acid); font-style:normal; }
        .hero-copy { max-width:340px; color:#c5c7b9; font-size:1rem; line-height:1.5; margin:0 1rem .15rem 0; }
        .hero-bottom { padding-top:1rem; border-top:1px solid #44463b; color:#a9ac9e; }
        .hero-bottom b { color:var(--paper); font-weight:500; }
        .section-kicker { color:var(--acid); margin-bottom:.4rem; }
        @keyframes blink { 50% { opacity:.35; } }
        @keyframes hero-in { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        @keyframes hero-scan { from { left:-25%; } to { left:125%; } }
        .note { border:1px solid #666039; background:#272519; color:#e3dfc6; padding:.9rem 1rem; border-radius:4px; line-height:1.5; }
        .fallback-note { border:1px solid #665339; background:#271f19; color:#e3d5c6; padding:.9rem 1rem; border-radius:4px; line-height:1.5; }
        .stButton > button,.stDownloadButton > button { border-radius:3px; border:1px solid var(--acid); background:var(--acid); color:#11120f;
          font-family:'DM Mono',monospace; font-size:.76rem; letter-spacing:.04em; font-weight:500; text-transform:uppercase;
          min-height:2.7rem; transition:transform .18s ease,filter .18s ease; }
        .stButton > button:hover,.stDownloadButton > button:hover { color:#11120f; filter:brightness(1.1); transform:translateY(-2px); }
        .stButton > button:disabled { background:#45473d; color:#999c8d; border-color:#45473d; }
        [data-testid="stMetric"] { background:#191a16; border:1px solid #35372e; padding:.85rem 1rem; border-radius:5px; transition:transform .18s,border-color .18s; }
        [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:#79804e; }
        [data-testid="stMetricLabel"] { color:#aaad9e; } [data-testid="stMetricValue"] { color:var(--acid); }
        [data-testid="stFileUploader"] { background:#191a16; border:1px dashed #696d58; border-radius:6px; padding:.3rem; }
        [data-testid="stFileUploaderDropzone"] { background:#191a16; border:0; }
        [data-testid="stFileUploaderDropzone"] * { color:#dedfd3; }
        [data-testid="stDataFrame"] { border:1px solid #35372e; border-radius:5px; }
        hr { border-color:#35372e; }
        @media(max-width:720px) { .hero { min-height:0; padding:1.1rem; } .hero-layout { display:block; margin:2rem 0; }
          .hero h1 { font-size:4rem; } .hero-copy { margin:1.2rem 0 0; } .hero-bottom { align-items:flex-start; flex-direction:column; gap:.4rem; } }
        @media(prefers-reduced-motion:reduce) { *,*:before,*:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; } }
        </style>
        <section class="hero">
          <div class="hero-top"><span class="micro">RDD2022 &nbsp; / &nbsp; FIELD MONITOR 01</span><span class="live"><i></i> LOCAL INFERENCE</span></div>
          <div class="hero-layout"><h1>READ THE<br><em>ROAD.</em></h1>
            <p class="hero-copy">Road-surface damage analysis from your uploaded field imagery. One frame or a full drive, processed locally with your trained YOLO model.</p>
          </div>
          <div class="hero-bottom"><span><b>INPUT</b> &nbsp; IMAGE / VIDEO</span><span><b>MODEL</b> &nbsp; YOLOv8n · RDD2022</span><span>NO LIVE CAMERA · FILES STAY LOCAL</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def dataframe_for(rows: list[dict]):
    import pandas as pd
    columns = [
        "timestamp_seconds", "frame_number", "class_code", "class_name", "confidence",
        "severity", "severity_score", "area_fraction", "x1", "y1", "x2", "y2",
    ]
    return pd.DataFrame(rows, columns=columns)


def show_results(result: dict, filename: str, mode: str, is_rdd: bool = True) -> None:
    import pandas as pd
    import plotly.express as px

    df = dataframe_for(result["rows"])
    count = len(df)
    st.markdown('<div class="section-kicker">01 / Analysis output</div>', unsafe_allow_html=True)
    st.header("Inspection results")

    if not is_rdd:
        st.markdown(
            '<div class="fallback-note"><strong>⚡ General detection mode.</strong> '
            'The trained RDD2022 checkpoint was not found, so analysis is running with the '
            'generic YOLOv8n (COCO-pretrained) model. Detections show general objects, not '
            'road-damage-specific classes. Train a model for accurate road damage detection.</div>',
            unsafe_allow_html=True,
        )

    if mode == "Image":
        source, annotated = st.columns(2)
        with source:
            st.caption("SOURCE IMAGE")
            st.image(result["source"], use_container_width=True)
        with annotated:
            st.caption("ANNOTATED IMAGE")
            st.image(result["image"], use_container_width=True)
        st.caption(f'{result["width"]} × {result["height"]} pixels · single image inference')
        st.download_button("Download annotated image", result["image"],
                           file_name=f"{Path(filename).stem}_annotated.png", mime="image/png",
                           key="download_annotated_image")
    else:
        st.video(result["video"], format="video/mp4")
        st.download_button("Download annotated video", result["video"],
                           file_name=f"{Path(filename).stem}_annotated.mp4", mime="video/mp4",
                           key="download_annotated_video")
        st.caption(f'{result["width"]} × {result["height"]} · {result["fps"]:.2f} FPS · '
                   f'{result["duration"]:.1f} seconds · analyzed {result["sampled_frames"]:,} of {result["frame_count"]:,} frames')

    st.markdown(
        '<div class="note"><strong>Heuristic severity estimate.</strong> This transparent class and '
        'image-area rule is not an engineering assessment. Video counts are detections in sampled frames; '
        'the same road feature may be counted again later.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-kicker" style="margin-top:1.5rem">02 / Damage summary</div>', unsafe_allow_html=True)
    st.subheader("Detection observations")

    if is_rdd:
        # RDD2022 mode: show the four specific damage classes
        counts = df["class_name"].value_counts() if count else pd.Series(dtype=int)
        metrics = st.columns(5)
        metrics[0].metric("Total detections", f"{count:,}")
        for i, name in enumerate(CLASS_NAMES.values(), 1):
            metrics[i].metric(name, f'{int(counts.get(name, 0)):,}')
    else:
        # Generic mode: show top detected classes
        counts = df["class_name"].value_counts() if count else pd.Series(dtype=int)
        top_classes = counts.head(4)
        metrics = st.columns(min(5, 1 + len(top_classes)))
        metrics[0].metric("Total detections", f"{count:,}")
        for i, (name, cnt) in enumerate(top_classes.items(), 1):
            if i < len(metrics):
                metrics[i].metric(name.title(), f"{int(cnt):,}")

    if count:
        class_counts = df.groupby("class_name", as_index=False).size().rename(columns={"size": "count"})
        severity_counts = (df["severity"].value_counts().reindex(["Minor", "Moderate", "Severe"], fill_value=0)
                           .rename_axis("severity").reset_index(name="count"))
        layout = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Mono, monospace", color="#dedfd3", size=11),
                      margin=dict(l=12, r=12, t=55, b=12))
        left, right = st.columns(2)
        with left:
            fig = px.bar(class_counts, x="class_name", y="count", color="class_name",
                         title="DETECTIONS BY CLASS",
                         color_discrete_sequence=["#d5ff58", "#fb8958", "#a8a9fc", "#82d5d0"])
            fig.update_layout(**layout, showlegend=False)
            fig.update_xaxes(title=None, gridcolor="#303129")
            fig.update_yaxes(title="Observations", gridcolor="#303129")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(severity_counts, x="severity", y="count", color="severity",
                         title="HEURISTIC SEVERITY ESTIMATES",
                         color_discrete_map={"Minor": "#d5ff58", "Moderate": "#fbad58", "Severe": "#fb6958"})
            fig.update_layout(**layout, showlegend=False)
            fig.update_xaxes(title=None, gridcolor="#303129")
            fig.update_yaxes(title="Observations", gridcolor="#303129")
            st.plotly_chart(fig, use_container_width=True)
        if mode == "Video":
            timeline = (df.assign(time_bucket=(df["timestamp_seconds"] // 5 * 5).astype(int))
                        .groupby(["time_bucket", "class_name"], as_index=False).size()
                        .rename(columns={"size": "count", "time_bucket": "Time (s)"}))
            fig = px.line(timeline, x="Time (s)", y="count", color="class_name", markers=True,
                          title="DETECTION OBSERVATIONS OVER VIDEO TIME",
                          color_discrete_sequence=["#d5ff58", "#fb8958", "#a8a9fc", "#82d5d0"])
            fig.update_layout(**layout)
            fig.update_xaxes(gridcolor="#303129")
            fig.update_yaxes(title="Observations", gridcolor="#303129")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections passed the selected confidence threshold.")

    st.markdown('<div class="section-kicker">03 / Detection log</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Export detection log as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{Path(filename).stem}_detections.csv", mime="text/csv",
                       key="download_detection_csv")


inject_styles()
st.iframe(motion_strip(), height=175)

# ── Determine which model mode to use ──────────────────────────────────────────
model_path = Path(DEFAULT_MODEL_PATH)
has_trained_model = model_path.is_file()

with st.sidebar:
    st.markdown('<div class="section-kicker">Control panel</div>', unsafe_allow_html=True)
    st.header("Analysis settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    sample_every = st.slider("Analyze every Nth video frame", 1, 30, 3, 1)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768], value=640)
    device_choice = st.selectbox("Inference device", ["Auto", "CPU", "CUDA: 0"])
    inference_device = {"Auto": None, "CPU": "cpu", "CUDA: 0": "0"}[device_choice]
    st.caption("A wider frame interval is faster but records fewer video observations.")
    st.divider()
    if has_trained_model:
        st.caption(f"Checkpoint path: {model_path.as_posix()}")
        st.caption("Inference requires a trained four-class RDD2022 checkpoint.")
    else:
        st.info("🔄 **Fallback mode**: using generic YOLOv8n (COCO).")
        st.caption("For road-damage-specific detection, train a model with `scripts/train.py`.")

st.markdown('<div class="section-kicker">Input / Field upload</div>', unsafe_allow_html=True)
mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True, label_visibility="collapsed")
extensions = ["jpg", "jpeg", "png"] if mode == "Image" else ["mp4", "avi", "mov"]
media_label = "image" if mode == "Image" else "video"
uploaded = st.file_uploader(
    f"Drop a road {media_label} here or browse · {', '.join(x.upper() for x in extensions)}",
    type=extensions, key=f"road_damage_{mode.lower()}_upload",
    help="Files are processed locally. No webcam or live stream is available.",
)

if not has_trained_model:
    st.markdown(
        '<div class="fallback-note">⚡ <strong>Running in general detection mode.</strong> '
        'The trained RDD2022 weights were not found. The app will use the generic YOLOv8n model '
        '(COCO-pretrained, 80 classes) for analysis. Upload an image or video to begin.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("How to install the trained checkpoint for road-damage detection"):
        st.markdown(
            "Prepare the RDD2022 dataset and run the documented training command. The training script "
            "validates best.pt and copies it to the expected path automatically. For an existing compatible "
            "checkpoint, follow the manual install step in README.md."
        )
        st.code(
            r'.\.venv312\Scripts\python.exe scripts\train.py --data "data\rdd2022_yolo\data.yaml"',
            language="powershell",
        )
        st.caption(
            "Generic yolov8n.pt weights are training initialization only; they do not detect the RDD2022 road-damage classes."
        )

if uploaded is None:
    st.info(f"Choose an uploaded road {media_label} to begin.")
else:
    source_bytes = uploaded.getvalue()
    if not source_bytes:
        st.error("The uploaded file is empty.")
    else:
        st.caption(f"Selected: **{uploaded.name}** · {len(source_bytes) / (1024 * 1024):.1f} MiB")
        if mode == "Image":
            st.image(source_bytes, caption="Uploaded source image", use_container_width=True)
        else:
            with st.expander("Preview uploaded video"):
                st.video(source_bytes)

        digest = hashlib.sha256(source_bytes).hexdigest()
        model_key = str(model_path.resolve()) if has_trained_model else "generic_yolov8n"
        key = (mode, digest, confidence, sample_every if mode == "Video" else None,
               image_size, inference_device, model_key)
        button = "Analyze image" if mode == "Image" else "Analyze video"

        # Analysis is always enabled now — fallback model is used when trained weights are absent
        if st.button(button, type="primary", key=f"analyze_{mode.lower()}"):
            progress = st.progress(0, text="Preparing local inference…") if mode == "Video" else None

            def report_progress(fraction: float, message: str) -> None:
                if progress is not None:
                    progress.progress(min(100, int(fraction * 100)), text=message)

            try:
                if has_trained_model:
                    model_stat = model_path.stat()
                    model = load_model(str(model_path), model_stat.st_mtime_ns, model_stat.st_size)
                else:
                    model = load_generic_model()

                from road_damage.model import is_road_damage_model
                is_rdd = is_road_damage_model(model)

                with st.spinner("Running YOLO inference…"):
                    if mode == "Image":
                        from road_damage.image import process_image
                        output = process_image(source_bytes, uploaded.name, model, confidence, image_size, device=inference_device)
                        stored = {"key": key, "image": output.image_bytes, "source": source_bytes,
                                  "rows": output.detections, "width": output.width, "height": output.height,
                                  "is_rdd": is_rdd}
                    else:
                        from road_damage.video import process_video
                        output = process_video(source_bytes, uploaded.name, model, confidence, sample_every,
                                               image_size, report_progress, device=inference_device)
                        stored = {"key": key, "video": output.video_bytes, "rows": output.detections,
                                  "frame_count": output.frame_count, "sampled_frames": output.sampled_frames,
                                  "fps": output.fps, "width": output.width, "height": output.height,
                                  "duration": output.duration, "is_rdd": is_rdd}
                st.session_state["road_damage_result"] = stored
                if progress is not None:
                    progress.progress(100, text="Video processing complete")
            except Exception as exc:
                st.session_state.pop("road_damage_result", None)
                st.error(f"Could not process this {media_label}: {exc}")

        saved = st.session_state.get("road_damage_result")
        if saved and saved.get("key") == key:
            st.divider()
            show_results(saved, uploaded.name, mode, is_rdd=saved.get("is_rdd", True))

with st.expander("Supported RDD2022 damage classes"):
    st.markdown("\n".join(f"- **{code} — {name}**" for code, name in CLASS_NAMES.items()))
