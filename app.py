import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Industrial Safety Monitor", page_icon="🦺", layout="wide")
st.title("🦺 AI-Powered Industrial Safety Monitor")
st.caption("YOLO PPE detection • ByteTrack tracking • Reviewable potential-violation reports")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.40, 0.05)
    model_source = st.text_input(
        "Model weights path",
        value="best.pt",
        help="Path to your PPE-trained YOLO weights. The weights file is not included in this repository."
    )
    st.info("Use a PPE-trained model. Generic YOLO weights do not reliably provide PPE classes.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
run = st.button("Analyze video", type="primary", disabled=uploaded is None)

if run and uploaded:
    model_path = Path(model_source)
    if not model_path.exists():
        st.error(
            f"Model weights not found: {model_path}. Download/place your authorized PPE model weights "
            "at this path and check its license before use."
        )
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / uploaded.name
        input_path.write_bytes(uploaded.getvalue())
        output_dir = temp / "output"
        output_dir.mkdir()

        with st.spinner("Running PPE detection and tracking…"):
            model = YOLO(str(model_path))
            rows = []
            frame_number = 0
            video_results = model.track(
                source=str(input_path),
                tracker="bytetrack.yaml",
                conf=conf,
                persist=True,
                stream=True,
                save=True,
                project=str(output_dir),
                name="annotated",
                exist_ok=True,
                verbose=False,
            )
            # Ultralytics saves the annotated video. For CSV timestamps, use frame index / FPS.
            import cv2
            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()

            for result in video_results:
                timestamp = frame_number / fps if fps > 0 else None
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = str(model.names[class_id])
                        track_id = int(box.id[0]) if box.id is not None else None
                        rows.append({
                            "frame_number": frame_number,
                            "timestamp_seconds": round(timestamp, 3) if timestamp is not None else None,
                            "class_name": class_name,
                            "confidence": round(float(box.conf[0]), 4),
                            "track_id": track_id,
                            "potential_violation": class_name.lower() == "no-helmet",
                        })
                frame_number += 1

            df = pd.DataFrame(rows, columns=[
                "frame_number", "timestamp_seconds", "class_name", "confidence",
                "track_id", "potential_violation"
            ])
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.session_state["report_csv"] = csv_bytes
            st.session_state["report_df"] = df

            video_files = [
                p for p in output_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ]
            if video_files:
                st.session_state["annotated_video"] = video_files[0].read_bytes()
                st.session_state["annotated_name"] = video_files[0].name
            else:
                st.session_state["annotated_video"] = None

if "report_df" in st.session_state:
    df = st.session_state["report_df"]
    lower = df["class_name"].astype(str).str.lower() if not df.empty else pd.Series(dtype=str)
    potential = df[lower.eq("no-helmet")] if not df.empty else df

    c1, c2, c3 = st.columns(3)
    c1.metric("Total detections", len(df))
    c2.metric("Human detections", int(lower.eq("human").sum()) if not df.empty else 0)
    c3.metric("Potential no-helmet detections", len(potential))

    st.warning(
        "These are model detections for human review, not confirmed safety incidents. "
        "Repeated frames can contain the same person, and false positives/negatives are possible."
    )
    st.subheader("Detection counts")
    if not df.empty:
        st.bar_chart(df["class_name"].value_counts())
        st.subheader("Detection log")
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not potential.empty and potential["timestamp_seconds"].notna().any():
            st.subheader("Potential no-helmet detections over time")
            timeline = potential.dropna(subset=["timestamp_seconds"]).groupby("timestamp_seconds").size()
            st.line_chart(timeline)
    st.download_button(
        "Download detection report (CSV)",
        data=st.session_state["report_csv"],
        file_name="ppe_detection_report.csv",
        mime="text/csv",
    )
    video_data = st.session_state.get("annotated_video")
    if video_data:
        st.subheader("Annotated output video")
        st.video(video_data)
        st.download_button(
            "Download annotated video",
            data=video_data,
            file_name=st.session_state.get("annotated_name", "annotated_video.mp4"),
            mime="video/mp4",
        )
else:
    st.markdown("""
### How to use
1. Install the requirements.
2. Place your authorized PPE-trained YOLO weights at `best.pt` (or specify another path).
3. Upload a short video and click **Analyze video**.
4. Review the annotated output and download the CSV.

**Important:** validate the model on representative industrial footage before using it for any real safety decisions.
""")
