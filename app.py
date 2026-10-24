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
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#11120f; --paper:#efeee6; --line:#35372e; --acid:#d5ff58; --danger:#ef4444; --warning:#f97316; }
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
        .section-kicker { color:var(--acid); margin-bottom:.4rem; font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; }
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

        /* ADAS HUD Warning Alert Banners */
        .hud-banner {
          border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
          position: relative; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.5);
          animation: hero-in .5s ease-out both;
        }
        .hud-critical {
          background: radial-gradient(ellipse at 85% 50%, rgba(239,68,68,0.25) 0%, transparent 65%), linear-gradient(135deg, #2b1111 0%, #150a0a 100%);
          border: 1px solid #ef4444; animation: pulse-glow-red 2.4s infinite ease-in-out;
        }
        .hud-high {
          background: radial-gradient(ellipse at 85% 50%, rgba(249,115,22,0.22) 0%, transparent 65%), linear-gradient(135deg, #2c1a0e 0%, #17100b 100%);
          border: 1px solid #f97316;
        }
        .hud-caution {
          background: radial-gradient(ellipse at 85% 50%, rgba(234,179,8,0.18) 0%, transparent 65%), linear-gradient(135deg, #27230e 0%, #15140a 100%);
          border: 1px solid #eab308;
        }
        .hud-safe {
          background: radial-gradient(ellipse at 85% 50%, rgba(34,197,94,0.18) 0%, transparent 65%), linear-gradient(135deg, #0e2716 0%, #0a170f 100%);
          border: 1px solid #22c55e;
        }
        @keyframes pulse-glow-red {
          0%, 100% { box-shadow: 0 0 15px rgba(239,68,68,0.3); }
          50% { box-shadow: 0 0 32px rgba(239,68,68,0.65); }
        }
        .hud-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: .6rem; }
        .hud-badge {
          display: inline-flex; align-items: center; gap: .4rem;
          font-family: 'DM Mono', monospace; font-size: .74rem; font-weight: 600;
          letter-spacing: .12em; padding: .3rem .75rem; border-radius: 4px; text-transform: uppercase;
        }
        .hud-badge-critical { background: #ef4444; color: #fff; box-shadow: 0 0 10px rgba(239,68,68,0.5); }
        .hud-badge-high { background: #f97316; color: #fff; }
        .hud-badge-caution { background: #eab308; color: #111; }
        .hud-badge-safe { background: #22c55e; color: #111; }
        .hud-title { font-size: 1.35rem; font-weight: 700; color: #ffffff; margin: .3rem 0 .4rem; letter-spacing: -.02em; }
        .hud-desc { color: #d6d8c8; font-size: .95rem; line-height: 1.5; margin-bottom: .75rem; }
        .hud-action {
          background: rgba(0,0,0,0.55); border-left: 3px solid var(--acid);
          padding: .75rem 1rem; border-radius: 0 5px 5px 0;
          font-family: 'DM Mono', monospace; font-size: .88rem; color: #ffffff;
        }
        .hud-action strong { color: var(--acid); }

        /* Vehicle Vulnerability Card & Meters */
        .vuln-card {
          background: #181914; border: 1px solid #35372e; border-radius: 6px;
          padding: 1rem 1.15rem; margin-bottom: .85rem; transition: transform .2s, border-color .2s;
        }
        .vuln-card:hover { transform: translateY(-2px); border-color: #79804e; }
        .vuln-card-title { font-family: 'DM Mono', monospace; font-size: .75rem; text-transform: uppercase; color: #aaad9e; letter-spacing: .08em; }
        .vuln-card-val { font-size: 1.35rem; font-weight: 700; color: var(--paper); margin: .3rem 0; }
        .vuln-bar-bg { background: #252720; border-radius: 999px; height: 8px; overflow: hidden; margin: .55rem 0 .35rem; }
        .vuln-bar-fill { height: 100%; border-radius: 999px; transition: width .5s ease; }
        .vuln-card-desc { font-size: .78rem; color: #a9ac9e; line-height: 1.4; }

        /* Vehicle Type Box */
        .vtype-card {
          background: #191b15; border: 1px solid #373a30; border-radius: 6px;
          padding: .9rem 1rem; margin-bottom: .6rem;
        }
        .vtype-title { font-family: 'DM Mono', monospace; font-size: .78rem; font-weight: 600; color: var(--acid); letter-spacing: .05em; }
        .vtype-text { font-size: .85rem; color: #c9cbc0; margin-top: .3rem; line-height: 1.4; }

        @media(max-width:720px) { .hero { min-height:0; padding:1.1rem; } .hero-layout { display:block; margin:2rem 0; }
          .hero h1 { font-size:4rem; } .hero-copy { margin:1.2rem 0 0; } .hero-bottom { align-items:flex-start; flex-direction:column; gap:.4rem; } }
        @media(prefers-reduced-motion:reduce) { *,*:before,*:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; } }
        </style>
        <section class="hero">
          <div class="hero-top"><span class="micro">RDD2022 &nbsp; / &nbsp; ADAS FIELD MONITOR</span><span class="live"><i></i> ACTIVE INFERENCE</span></div>
          <div class="hero-layout"><h1>READ THE<br><em>ROAD.</em></h1>
            <p class="hero-copy">AI road hazard perception, vehicle impact assessment, and driver warning advisory. Processed locally with real-time lane trajectory tracking.</p>
          </div>
          <div class="hero-bottom"><span><b>INPUT</b> &nbsp; IMAGE / VIDEO</span><span><b>AI MODULE</b> &nbsp; YOLOv8 + VEHICLE IMPACT ADAS</span><span>NO LIVE CAMERA · FILES STAY LOCAL</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def dataframe_for(rows: list[dict]):
    import pandas as pd
    if not rows:
        return pd.DataFrame()
    columns_order = [
        "timestamp_seconds", "frame_number", "class_code", "class_name", "confidence",
        "warning_level", "impact_level", "impact_score", "lane_zone", "in_vehicle_path",
        "proximity_zone", "estimated_distance_m", "recommended_action",
        "vibration_shock", "suspension_risk", "tire_risk", "two_wheeler_risk",
        "severity", "severity_score", "area_fraction", "x1", "y1", "x2", "y2",
    ]
    all_keys = list(rows[0].keys())
    ordered_cols = [c for c in columns_order if c in all_keys]
    for k in all_keys:
        if k not in ordered_cols:
            ordered_cols.append(k)
    return pd.DataFrame(rows)[ordered_cols]


def show_results(result: dict, filename: str, mode: str, is_rdd: bool = True) -> None:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from road_damage.vehicle_impact import summarize_overall_assessment

    df = dataframe_for(result["rows"])
    count = len(df)
    summary = result.get("summary") or {}
    if not summary and count > 0:
        # Fallback if summary wasn't directly cached
        from road_damage.vehicle_impact import analyze_damaged_spot
        assessments = []
        for r in result["rows"]:
            a = analyze_damaged_spot(
                x1=r.get("x1", 0), y1=r.get("y1", 0), x2=r.get("x2", 1), y2=r.get("y2", 1),
                img_width=result.get("width", 640), img_height=result.get("height", 640),
                class_code=r.get("class_code", "D40"), class_name=r.get("class_name", "Pothole"),
                confidence=r.get("confidence", 0.5),
            )
            assessments.append(a)
        summary = summarize_overall_assessment(assessments)
    elif not summary:
        summary = summarize_overall_assessment([])

    # ── 00 / ACTIVE ADAS WARNING HUD ──────────────────────────────────────────
    warning_level = summary.get("overall_warning_level", "CLEAR")
    if not count:
        warning_level = "CLEAR"

    hud_class = {
        "CRITICAL": "hud-critical",
        "HIGH": "hud-high",
        "CAUTION": "hud-caution",
        "SAFE": "hud-safe",
        "CLEAR": "hud-safe",
    }.get(warning_level, "hud-caution")

    badge_class = {
        "CRITICAL": "hud-badge-critical",
        "HIGH": "hud-badge-high",
        "CAUTION": "hud-badge-caution",
        "SAFE": "hud-badge-safe",
        "CLEAR": "hud-badge-safe",
    }.get(warning_level, "hud-badge-caution")

    action_text = summary.get("recommended_action") or (
        "Road surface is clear. Safe to proceed at regular speed." if not count else "Proceed with road awareness."
    )
    dominant_hazard = summary.get("dominant_hazard", "None")
    max_impact = summary.get("max_impact_score", 0)
    in_path_count = summary.get("in_path_count", 0)

    icon = "🚨" if warning_level == "CRITICAL" else ("⚠️" if warning_level == "HIGH" else ("⚡" if warning_level == "CAUTION" else "✓"))
    warning_title = f"{icon} {warning_level} ALERT: {dominant_hazard.upper()}" if count else "✓ ROAD SURFACE CLEAR: NO HAZARDS DETECTED"

    st.markdown('<div class="section-kicker">01 / ADAS Real-Time Driver Warning</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hud-banner {hud_class}">
          <div class="hud-header">
            <span class="hud-badge {badge_class}">{icon} WARNING LEVEL: {warning_level}</span>
            <span class="micro" style="color:#d6d8c8">COLLISION IMPACT INDEX: <b style="color:var(--acid);font-size:1.15rem">{max_impact}%</b></span>
          </div>
          <div class="hud-title">{warning_title}</div>
          <div class="hud-desc">
            AI detection localized <b>{count:,}</b> road defect(s). 
            <b>{in_path_count}</b> hazard spot(s) detected directly in the vehicle travel path.
          </div>
          <div class="hud-action">
            <strong>RECOMMENDED DRIVER ACTION:</strong> {action_text}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if count > 0 and warning_level in {"CRITICAL", "HIGH", "CAUTION"}:
        components.html(
            """
            <div style="display:flex;align-items:center;gap:14px;margin:2px 0 8px;">
              <button id="alertBtn" onclick="playAdasChime()" style="
                background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
                color:#ffffff; border:1px solid #f87171; padding:7px 16px; border-radius:4px;
                font-family:'DM Mono',monospace; font-size:11px; font-weight:600;
                letter-spacing:0.08em; cursor:pointer; display:inline-flex; align-items:center; gap:7px;
                box-shadow: 0 0 12px rgba(239,68,68,0.45); transition: transform 0.15s ease;
              " onmouseover="this.style.transform='scale(1.03)'" onmouseout="this.style.transform='scale(1)'">
                <span>🔊</span> PLAY ADAS AUDIBLE ALERT CHIME
              </button>
              <span style="color:#a8aa9c;font-family:'DM Mono',monospace;font-size:11px;letter-spacing:0.05em;">
                In-cabin audio advisory simulation
              </span>
            </div>
            <script>
            function playAdasChime() {
              try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const now = ctx.currentTime;
                // High tone pulse
                const osc1 = ctx.createOscillator();
                const gain1 = ctx.createGain();
                osc1.type = 'sawtooth';
                osc1.frequency.setValueAtTime(920, now);
                gain1.gain.setValueAtTime(0.3, now);
                gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
                osc1.connect(gain1);
                gain1.connect(ctx.destination);
                osc1.start(now);
                osc1.stop(now + 0.18);

                // Second alert pulse
                const osc2 = ctx.createOscillator();
                const gain2 = ctx.createGain();
                osc2.type = 'sawtooth';
                osc2.frequency.setValueAtTime(1240, now + 0.22);
                gain2.gain.setValueAtTime(0.35, now + 0.22);
                gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.50);
                osc2.connect(gain2);
                gain2.connect(ctx.destination);
                osc2.start(now + 0.22);
                osc2.stop(now + 0.50);
              } catch(e) {}
            }
            </script>
            """,
            height=48,
        )

    # ── 01 / Visual Inspection ────────────────────────────────────────────────
    st.markdown('<div class="section-kicker" style="margin-top:1.2rem">02 / Visual ADAS Perception</div>', unsafe_allow_html=True)
    st.header("Inspection imagery & trajectory overlay")

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
            st.caption("ANNOTATED ADAS IMAGE (WITH TRAJECTORY & TELEMETRY)")
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

    # ── 02 / Vehicle Impact & Mechanical Stress Analysis ──────────────────────
    st.markdown('<div class="section-kicker" style="margin-top:1.6rem">03 / Vehicle Impact & Risk Analysis</div>', unsafe_allow_html=True)
    st.subheader("Vehicle mechanical stress assessment")

    # High-level telemetry metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Impact Index", f"{max_impact}%", delta=f"{summary.get('overall_impact_level', 'NONE')}")
    m2.metric("Average Impact", f"{summary.get('avg_impact_score', 0)}%")
    m3.metric("Hazards in Wheel Path", f"{in_path_count:,}")
    worst_shock = "Mild Bump"
    if count and "vibration_shock" in df.columns:
        worst_shock = df.iloc[0]["vibration_shock"] if len(df) else "Mild Bump"
    m4.metric("Dynamic Vibration", worst_shock.split("(")[0].strip())

    # Component Vulnerability Gauges
    vuln = summary.get("vulnerability", {})
    st.markdown('<div class="section-kicker" style="margin-top:1.2rem">Subsystem Vulnerability Index</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    def color_for_pct(val: int) -> str:
        if val >= 75:
            return "#ef4444"
        if val >= 50:
            return "#f97316"
        if val >= 25:
            return "#eab308"
        return "#22c55e"

    with c1:
        tire_val = vuln.get("tires", 0)
        c_tire = color_for_pct(tire_val)
        st.markdown(
            f"""
            <div class="vuln-card">
              <div class="vuln-card-title">🛞 Tires & Rims</div>
              <div class="vuln-card-val" style="color:{c_tire}">{tire_val}%</div>
              <div class="vuln-bar-bg"><div class="vuln-bar-fill" style="width:{tire_val}%;background:{c_tire}"></div></div>
              <div class="vuln-card-desc">Rim denting, bead unseating, and sidewall blowout stress.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        susp_val = vuln.get("suspension", 0)
        c_susp = color_for_pct(susp_val)
        st.markdown(
            f"""
            <div class="vuln-card">
              <div class="vuln-card-title">🏎️ Suspension & Shocks</div>
              <div class="vuln-card-val" style="color:{c_susp}">{susp_val}%</div>
              <div class="vuln-bar-bg"><div class="vuln-bar-fill" style="width:{susp_val}%;background:{c_susp}"></div></div>
              <div class="vuln-card-desc">Strut damper bottoming, coil fatigue, control-arm bushing strain.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        steer_val = vuln.get("steering", 0)
        c_steer = color_for_pct(steer_val)
        st.markdown(
            f"""
            <div class="vuln-card">
              <div class="vuln-card-title">⚙️ Steering & Alignment</div>
              <div class="vuln-card-val" style="color:{c_steer}">{steer_val}%</div>
              <div class="vuln-bar-bg"><div class="vuln-bar-fill" style="width:{steer_val}%;background:{c_steer}"></div></div>
              <div class="vuln-card-desc">Toe/camber drift, sudden steering kickback, and groove tramlining.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        chas_val = vuln.get("chassis", 0)
        c_chas = color_for_pct(chas_val)
        st.markdown(
            f"""
            <div class="vuln-card">
              <div class="vuln-card-title">🛡️ Underbody & Clearance</div>
              <div class="vuln-card-val" style="color:{c_chas}">{chas_val}%</div>
              <div class="vuln-bar-bg"><div class="vuln-bar-fill" style="width:{chas_val}%;background:{c_chas}"></div></div>
              <div class="vuln-card-desc">Front splitter scrape, oil pan hazard, undercarriage impact.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Multi-Vehicle Type Vulnerability Breakdown
    st.markdown('<div class="section-kicker" style="margin-top:1rem">Impact by Vehicle Category</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        two_wheel_val = vuln.get("two_wheeler", 0)
        tw_color = "#ef4444" if two_wheel_val >= 60 else "#f97316"
        st.markdown(
            f"""
            <div class="vtype-card" style="border-color:{tw_color}">
              <div class="vtype-title" style="color:{tw_color}">🏍️ Two-Wheelers ({two_wheel_val}%)</div>
              <div class="vtype-text">
                {"<b>EXTREME FALL HAZARD</b>: High wheel-trap risk in potholes/cracks. Sudden handlebar deflection likely." if two_wheel_val >= 60 else "Elevated instability. Defect causes rapid front tire deflection."}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            """
            <div class="vtype-card">
              <div class="vtype-title">🚗 Sedans & Hatchbacks</div>
              <div class="vtype-text">
                Low ground clearance creates high risk of bumper scrapes and rim pinch flats on deep potholes.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v3:
        st.markdown(
            """
            <div class="vtype-card">
              <div class="vtype-title">🚙 SUVs & Crossovers</div>
              <div class="vtype-text">
                Generous clearance avoids undercarriage contact, but high vertical axle forces accelerate shock wear.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v4:
        st.markdown(
            """
            <div class="vtype-card">
              <div class="vtype-title">🚚 Commercial Trucks</div>
              <div class="vtype-text">
                Heavy payload transmits high kinetic impact, accelerating pavement failure and tire wear.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 03 / Damaged Spot Spatial Localization ────────────────────────────────
    st.markdown('<div class="section-kicker" style="margin-top:1.5rem">04 / Damaged Spot Localization</div>', unsafe_allow_html=True)
    st.subheader("Spatial road position & proximity")

    if count and "lane_zone" in df.columns:
        pos_col1, pos_col2 = st.columns(2)
        with pos_col1:
            lane_counts = df["lane_zone"].value_counts().reset_index()
            lane_counts.columns = ["Lane Position", "Spot Count"]
            fig_lane = px.pie(
                lane_counts, names="Lane Position", values="Spot Count",
                title="DEFECTS BY LATERAL LANE POSITION",
                color="Lane Position",
                color_discrete_map={
                    "Direct Vehicle Travel Path": "#ef4444",
                    "Left Shoulder / Lane": "#f97316",
                    "Right Shoulder / Lane": "#d5ff58",
                },
            )
            fig_lane.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono, monospace", color="#dedfd3", size=11),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_lane, use_container_width=True)

        with pos_col2:
            prox_counts = df["proximity_zone"].value_counts().reset_index()
            prox_counts.columns = ["Proximity Zone", "Spot Count"]
            fig_prox = px.bar(
                prox_counts, x="Proximity Zone", y="Spot Count",
                title="DEFECTS BY LONGITUDINAL PROXIMITY",
                color="Proximity Zone",
                color_discrete_sequence=["#ef4444", "#f97316", "#d5ff58"],
            )
            fig_prox.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono, monospace", color="#dedfd3", size=11),
                margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
            )
            fig_prox.update_xaxes(gridcolor="#303129")
            fig_prox.update_yaxes(gridcolor="#303129")
            st.plotly_chart(fig_prox, use_container_width=True)

    # ── 04 / Visual Analytics ─────────────────────────────────────────────────
    st.markdown('<div class="section-kicker" style="margin-top:1.5rem">05 / Visual Analytics</div>', unsafe_allow_html=True)
    st.subheader("Damage distribution & trends")

    if is_rdd:
        counts = df["class_name"].value_counts() if count else pd.Series(dtype=int)
        metrics = st.columns(5)
        metrics[0].metric("Total detections", f"{count:,}")
        for i, name in enumerate(CLASS_NAMES.values(), 1):
            metrics[i].metric(name, f'{int(counts.get(name, 0)):,}')
    else:
        counts = df["class_name"].value_counts() if count else pd.Series(dtype=int)
        top_classes = counts.head(4)
        metrics = st.columns(min(5, 1 + len(top_classes)))
        metrics[0].metric("Total detections", f"{count:,}")
        for i, (name, cnt) in enumerate(top_classes.items(), 1):
            if i < len(metrics):
                metrics[i].metric(name.title(), f"{int(cnt):,}")

    if count:
        class_counts = df.groupby("class_name", as_index=False).size().rename(columns={"size": "count"})
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
            if "impact_level" in df.columns:
                impact_counts = (df["impact_level"].value_counts().reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
                                 .rename_axis("impact_level").reset_index(name="count"))
                fig = px.bar(impact_counts, x="impact_level", y="count", color="impact_level",
                             title="VEHICLE IMPACT LEVEL DISTRIBUTION",
                             color_discrete_map={"LOW": "#22c55e", "MODERATE": "#eab308", "HIGH": "#f97316", "CRITICAL": "#ef4444"})
                fig.update_layout(**layout, showlegend=False)
                fig.update_xaxes(title=None, gridcolor="#303129")
                fig.update_yaxes(title="Observations", gridcolor="#303129")
                st.plotly_chart(fig, use_container_width=True)
            else:
                severity_counts = (df["severity"].value_counts().reindex(["Minor", "Moderate", "Severe"], fill_value=0)
                                   .rename_axis("severity").reset_index(name="count"))
                fig = px.bar(severity_counts, x="severity", y="count", color="severity",
                             title="HEURISTIC SEVERITY ESTIMATES",
                             color_discrete_map={"Minor": "#d5ff58", "Moderate": "#fbad58", "Severe": "#fb6958"})
                fig.update_layout(**layout, showlegend=False)
                fig.update_xaxes(title=None, gridcolor="#303129")
                fig.update_yaxes(title="Observations", gridcolor="#303129")
                st.plotly_chart(fig, use_container_width=True)

        if mode == "Video":
            # Video Impact timeline
            vleft, vright = st.columns(2)
            with vleft:
                timeline = (df.assign(time_bucket=(df["timestamp_seconds"] // 5 * 5).astype(int))
                            .groupby(["time_bucket", "class_name"], as_index=False).size()
                            .rename(columns={"size": "count", "time_bucket": "Time (s)"}))
                fig = px.line(timeline, x="Time (s)", y="count", color="class_name", markers=True,
                              title="OBSERVATIONS OVER VIDEO TIME",
                              color_discrete_sequence=["#d5ff58", "#fb8958", "#a8a9fc", "#82d5d0"])
                fig.update_layout(**layout)
                fig.update_xaxes(gridcolor="#303129")
                fig.update_yaxes(title="Observations", gridcolor="#303129")
                st.plotly_chart(fig, use_container_width=True)

            with vright:
                if "impact_score" in df.columns:
                    impact_time = (df.groupby("timestamp_seconds", as_index=False)["impact_score"].max()
                                   .rename(columns={"timestamp_seconds": "Time (s)", "impact_score": "Max Impact %"}))
                    fig_imp = px.area(impact_time, x="Time (s)", y="Max Impact %",
                                      title="VEHICLE IMPACT SEVERITY OVER VIDEO DRIVE",
                                      color_discrete_sequence=["#ef4444"])
                    fig_imp.update_layout(**layout)
                    fig_imp.update_xaxes(gridcolor="#303129")
                    fig_imp.update_yaxes(range=[0, 100], gridcolor="#303129")
                    st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No detections passed the selected confidence threshold.")

    # ── 05 / ADAS Telemetry & Detection Log ───────────────────────────────────
    st.markdown('<div class="section-kicker">06 / ADAS Telemetry & Detection Log</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Export detection log as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{Path(filename).stem}_adas_detections.csv", mime="text/csv",
                       key="download_detection_csv")



inject_styles()
st.iframe(motion_strip(), height=175)

# ── Determine which model mode to use ──────────────────────────────────────────
model_path = Path(DEFAULT_MODEL_PATH)
has_trained_model = model_path.is_file()

with st.sidebar:
    st.markdown('<div class="section-kicker">Control panel</div>', unsafe_allow_html=True)
    st.header("Analysis settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.15, 0.05,
                           help="Lower confidence (e.g. 0.10–0.20) catches subtle, water-filled, or shaded damage spots.")
    sample_every = st.slider("Analyze every Nth video frame", 1, 30, 3, 1)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768], value=640)
    device_choice = st.selectbox("Inference device", ["Auto", "CPU", "CUDA: 0"])
    inference_device = {"Auto": None, "CPU": "cpu", "CUDA: 0": "0"}[device_choice]
    st.caption("A wider frame interval is faster but records fewer video observations.")
    st.divider()
    if has_trained_model:
        st.success("✅ **RDD2022 AI Model Active**")
        st.caption(f"Loaded: `{model_path.name}`")
        st.caption("Trained on: Potholes, Alligator Cracks, Transverse Cracks, Longitudinal Cracks.")
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
                        stored = {
                            "key": key,
                            "image": output.image_bytes,
                            "source": source_bytes,
                            "rows": output.detections,
                            "width": output.width,
                            "height": output.height,
                            "is_rdd": is_rdd,
                            "summary": getattr(output, "summary", {}),
                        }
                    else:
                        from road_damage.video import process_video
                        output = process_video(source_bytes, uploaded.name, model, confidence, sample_every,
                                               image_size, report_progress, device=inference_device)
                        stored = {
                            "key": key,
                            "video": output.video_bytes,
                            "rows": output.detections,
                            "frame_count": output.frame_count,
                            "sampled_frames": output.sampled_frames,
                            "fps": output.fps,
                            "width": output.width,
                            "height": output.height,
                            "duration": output.duration,
                            "is_rdd": is_rdd,
                            "summary": getattr(output, "summary", {}),
                        }
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
