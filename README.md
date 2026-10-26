# AI-Powered Industrial Safety Monitoring

A computer-vision portfolio project exploring PPE detection in video using a YOLO model, ByteTrack tracking, annotated video output, and a downloadable CSV detection log.

## Features

- Upload a video in the Streamlit interface
- Run a PPE-trained YOLO detector
- Track detections with ByteTrack
- Save annotated video output
- Export frame-level detections with timestamps, confidence scores, class names, and tracking IDs
- Summarize potential `no-helmet` detections for human review

## Important validation note

This is a prototype, **not a certified safety system**. Model predictions may include false positives and false negatives. A detection of `no-helmet` is a *potential violation* and must be reviewed; it is not a confirmed incident. Repeated frame detections are not unique people or unique incidents.

The early pipeline test used a short generic demo video (~2.07 seconds). It demonstrated that inference/tracking and report generation can run, but it does **not** establish real-world PPE accuracy. The model also needs validation on representative industrial footage, including helmets, no-helmet cases, vests, occlusion, lighting, camera angles, and varied worker distances.

## Setup

Python 3.10–3.12 is recommended. Newer Python versions may not yet be supported by every ML dependency.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Model weights

The repository intentionally does not include model weights. Place your authorized PPE-trained YOLO weights in the project directory as `best.pt`, or enter another path in the app sidebar.

If using a third-party model, review its model card, license, and usage restrictions before redistributing weights or using it commercially. Keep a copy of the model's license/attribution information in your own records.

## Run the app

```bash
streamlit run app.py
```

Upload an MP4, AVI, MOV, or MKV file and select **Analyze video**. The app generates an annotated video and a CSV report.

## CSV fields

- `frame_number`
- `timestamp_seconds`
- `class_name`
- `confidence`
- `track_id` (may be blank)
- `potential_violation` (true when the predicted class is `no-helmet`)

Timestamps are estimated as frame index divided by the video's reported FPS. Variable-frame-rate footage may require a more precise timestamping method.

## Tech stack

- Python
- Ultralytics YOLO
- OpenCV
- ByteTrack
- Streamlit
- Pandas

## Suggested next improvements

- Validate on a labeled PPE dataset and report precision, recall, and per-class metrics
- Add person-to-PPE association rather than treating object classes independently
- Group consecutive detections into candidate events using track ID and time gaps
- Add unit tests, configurable model loading, and a deployment guide
- Test with variable-frame-rate and longer videos

## License

Choose and add a project license before publishing. Check the licenses of dependencies and any third-party model weights separately.
