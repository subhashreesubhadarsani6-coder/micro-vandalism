# Troubleshooting Guide: Guarding the Stones AI

Common diagnostic scenarios, remediation procedures, and architectural recovery steps for the CPU-only Egocentric Heritage Action Recognition pipeline.

---

## 1. Video Decoding & Container Issues

### Issue 1.1: `DECODING_ERROR: moov atom not found` (MP4/MOV)
- **Symptom**: OpenCV returns `False` on `cap.isOpened()`, or stderr reports `[mov,mp4,m4a] moov atom not found`.
- **Root Cause**: The MP4 file was incompletely copied, interrupted during recording, or truncated before the index atom (`moov`) was finalized.
- **Resolution**:
  1. Inspect the file size: `python dataset_tool.py inspect --file "path/to/clip.mp4"`.
  2. If the file is 0 bytes or corrupted, replace with a valid video container.
  3. The `InferenceEngine` automatically catches this error, marks the verdict as `ERROR` with `error_type="DECODING_ERROR"`, and does not crash the batch pipeline.

### Issue 1.2: Corrupted Frame Dimensions or Unsupported Codecs
- **Symptom**: `cv2.VideoCapture.read()` returns empty frames after opening.
- **Resolution**:
  - The pipeline uses uniform temporal seeking with cyclic boundary fallback. If all frames fail to decode, `VideoDecodingError` is raised and caught cleanly.
  - Convert troublesome files using standard H.264 MP4:
    ```bash
    ffmpeg -i input.mkv -c:v libx264 -pix_fmt yuv420p output.mp4
    ```

---

## 2. Pretrained Backbone Weights & Offline Operation

### Issue 2.1: `PretrainedWeightsUnavailableError`
- **Symptom**: An exception is raised when `--weights pretrained` is selected without local weight checkpoints.
- **Root Cause**: The system strictly enforces offline CPU execution and forbids automatic multi-megabyte weight downloads without explicit `--allow-download` approval.
- **Resolution**:
  1. Use offline weights mode (default):
     ```bash
     python dataset_tool.py extract-features --backbone mobilenet_v3_small --weights none
     ```
  2. To permit downloading official PyTorch weights once:
     ```bash
     python dataset_tool.py extract-features --backbone mobilenet_v3_small --weights pretrained --allow-download
     ```
  3. Or copy pretrained `.pth` files into `~/.cache/torch/hub/checkpoints/`.

---

## 3. Pre-Training Gate & Training Halts

### Issue 3.1: Pre-Training Verification Gate Refusal
- **Symptom**: Running `python dataset_tool.py train` halts with:
  `PretrainingPrerequisiteError: Pre-training checks failed. Cannot proceed with model training.`
- **Root Cause**: The dataset contains zero authentic first-person egocentric recordings, and 100% of labels are unverified. The gate prevents training biased, uncalibrated shortcut models on third-person stock clips.
- **Resolution**:
  1. Collect and annotate genuine egocentric video data following [`DATASET_GUIDELINES.md`](DATASET_GUIDELINES.md).
  2. For synthetic software development tests only, pass `--allow-unverified`.

---

## 4. Gradio Dashboard & Web Interface

### Issue 4.1: Port Already in Use (`OSError: Port 7860 is in use`)
- **Symptom**: Launching `dataset_tool.py dashboard` fails with address bind error.
- **Resolution**:
  - Specify an alternate port via the `--port` flag:
    ```bash
    python dataset_tool.py dashboard --port 7865
    ```

### Issue 4.2: Webcam Not Found in Browser
- **Symptom**: The Gradio video component shows a black screen or permissions prompt when clicking the webcam button.
- **Resolution**:
  - Modern web browsers require an `https://` origin or `http://localhost` / `http://127.0.0.1` for camera permissions.
  - If accessing from a remote machine, launch with `--share` to generate an HTTPS endpoint.
  - If no physical camera is connected, use file upload mode.

---

## 5. CPU Performance & Thread Allocation

### Issue 5.1: High CPU Latency During Decoding
- **Symptom**: Per-clip latency exceeds 500 ms on long video files.
- **Root Cause**: Software video decoding via OpenCV scales with video duration and resolution.
- **Resolution**:
  1. Use 8-frame sampling (`--num-frames 8`) instead of 16-frame.
  2. Utilize the incremental feature cache (`data/processed/features_mobilenet_v3_small_8f_224.npz`) to eliminate repeated decoding of static files.
  3. Ensure CPU thread count matches physical cores:
     ```bash
     python -c "import torch; print('Threads:', torch.get_num_threads())"
     ```
