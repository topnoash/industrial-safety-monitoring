# Development notes

## Verified during initial prototype testing
- PPE YOLO weights loaded and exposed classes: `helmet`, `human`, `no-helmet`, `vest`.
- YOLO tracking with `bytetrack.yaml` completed on a short demo video.
- CSV report and a basic class-count chart were generated in Colab.
- Confidence-threshold comparison showed that counts changed substantially as the threshold changed.

## Not yet validated
- Real-world helmet/vest accuracy.
- Person-to-PPE association.
- Unique-person or unique-incident counts.
- Long-video memory/runtime behavior in the Streamlit app.
- Commercial suitability of third-party model weights.

Do not describe unverified items as completed features or performance results.
