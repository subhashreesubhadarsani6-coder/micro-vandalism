# Guarding the Stones — Egocentric Heritage Protection AI

An end-to-end, CPU-optimized computer vision and temporal action recognition system designed to detect subtle, calm micro-vandalism (carving, scratching, chipping, marking) on protected cultural heritage monument surfaces from first-person (egocentric) body-worn or smart-glass video streams.

---

## 1. System Architecture

The pipeline is engineered for low-power edge execution on standard multi-core CPUs without requiring dedicated CUDA GPUs:

```
[Raw Video (.mp4/.avi/.mov/.mkv)]
        │
        ▼ (OpenCV Decoders)
[Uniform Temporal Frame Sampling (8 or 16 frames @ 224x224 RGB)]
        │
        ▼ (Frozen Pretrained Backbone)
[Visual Feature Extractor: MobileNetV3-Small (576-dim) / ResNet18 (512-dim)]
        │
        ▼ (Atomic Disk Cache: .npz)
[Temporal Action Recognizer: TemporalConv1DNet (1D TCN + Attentive Pooling)]
        │
        ├─► [Action Class Probabilities (11 Candidate Classes)]
        ├─► [Continuous Heritage Risk Score: R = sum P(c) * W_c in [0.0, 1.0]]
        ├─► [Temporal Frame Saliency & Timestamp Attribution]
        └─► [Operational Verdict: NORMAL / UNCERTAIN / POTENTIAL VANDALISM]
```

---

## 2. Hardware Profile & Target Environment

- **Target System**: Windows 11 (tested on Intel Core i5-12450H, 8 Cores / 12 Threads, 16 GB RAM).
- **Compute Target**: 100% CPU-only execution; zero CUDA runtime dependency.
- **Python Compatibility**: Python 3.10, 3.11, 3.12, 3.13.

---

## 3. Quickstart & Clean Installation

### 3.1 Environment Setup
```powershell
# Clone or navigate to the workspace
cd "c:\Project\micro vandalism"

# Create a dedicated virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install CPU dependencies
pip install -r requirements.txt
```

### 3.2 Verify Installation (Run 62 Unit & Integration Tests)
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

---

## 4. Unified CLI Reference (`dataset_tool.py`)

All pipeline operations are consolidated in the modular CLI:

| Command | Purpose | Example Usage |
| :--- | :--- | :--- |
| `extract` | Safely unpacks dataset zip without overwriting | `python dataset_tool.py extract` |
| `inspect` | Probes container metadata, FPS, codecs, SHA-256 | `python dataset_tool.py inspect --file "data/raw/carving/carving 02.mp4"` |
| `manifest` | Builds JSONL/CSV manifest with duplicate detection | `python dataset_tool.py manifest` |
| `summary` | Prints dataset health, problem counts, distributions | `python dataset_tool.py summary` |
| `schema` | Exports formal JSON Schema specification | `python dataset_tool.py schema --out data/annotations/schema.json` |
| `template` | Generates annotation workfile for human labeling | `python dataset_tool.py template --out data/annotations/workfile.json` |
| `validate` | Validates annotations against schema rules | `python dataset_tool.py validate --annotations data/annotations/workfile.json` |
| `benchmark` | Measures decode, preprocessing & inference latency | `python dataset_tool.py benchmark --backbone mobilenet_v3_small --num-frames 8` |
| `extract-features` | Incremental atomic feature caching to `.npz` | `python dataset_tool.py extract-features --backbone mobilenet_v3_small` |
| `verify-model` | Verifies CPU forward pass and tensor shapes | `python dataset_tool.py verify-model --model-type conv1d` |
| `train` | Pre-training gate check and model training | `python dataset_tool.py train --model-type conv1d --epochs 30` |
| `evaluate` | Evaluates accuracy, ECE, ROC-AUC, confusion matrix | `python dataset_tool.py evaluate --samples 100` |
| `predict` | Runs CPU inference & risk scoring on videos | `python dataset_tool.py predict --video "data/raw/carving/carving 02.mp4"` |
| `dashboard` | Launches interactive Gradio 6 security interface | `python dataset_tool.py dashboard --port 7860` |

---

## 5. Confidence vs. Real-World Risk

- **Classification Confidence**:
  $$\text{Confidence}(X) = \max_{c} P(y=c \mid X)$$
  Statistical certainty over the candidate taxonomy. A tourist pointing at an inscription can have 99% confidence on `pointing`, yet this action entails **zero physical risk**.

- **Domain-Weighted Heritage Risk Score**:
  $$R(X) = \sum_{c=1}^{C} P(y=c \mid X) \cdot W_c \quad (W_c \in [0.0, 1.0])$$
  Weights actions by physical alteration severity (`potential_chipping`: 1.0, `potential_scratching`: 0.95, `potential_scraping`: 0.90, `potential_marking`: 0.85, `ambiguous_interaction`: 0.50, `inspection`: 0.15, benign tourist gestures: 0.00).

- **Operational Verdict Matrix**:
  - `NORMAL`: Low risk ($R < 0.35$) and benign action category.
  - `UNCERTAIN / HUMAN REVIEW`: Risk in $[0.35, 0.70)$, ambiguous gesture, or confidence $< 0.40$.
  - `POTENTIAL VANDALISM`: Risk $\ge 0.70$ and action is in the potential damage class.

---

## 6. CPU Performance Benchmarks (Intel Core i5-12450H)

| Component / Architecture | Input Config | CPU Latency | Throughput | Peak RAM | ECE (Calibration) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TemporalConv1DNet` (Main)** | 8 frames | **1.07 ms** | **932 clips/s** | 0.21 MB | **0.078** |
| **`TemporalPoolingBaseline`** | 8 frames | **0.48 ms** | **2,079 clips/s** | 0.12 MB | **0.053** |
| **`LogisticRegression` (Legacy)** | 8 frames | 0.48 ms | 2,060 clips/s | 0.08 MB | 0.489 |
| **End-to-End Clip Processing** | 8 frames | **~254 ms** | **3.9 clips/s** | ~180 MB | N/A (Full Decode + Prep + Infer) |

*Full evaluation and latency details are documented in `EVALUATION_REPORT.md` and `INFERENCE_REPORT.md`.*

---

## 7. Interactive Gradio 6 Dashboard

Launch the local web dashboard for real-time video upload, live webcam streaming, timeline inspection, frame-by-frame evidence gallery, and officer review:

```powershell
python dataset_tool.py dashboard --port 7860
```
Open your browser at `http://localhost:7860`.

---

## 8. Repository Structure

```
micro vandalism/
├── src/
│   ├── config.py             # Configurable paths, taxonomy (11 classes), viewpoints
│   ├── video_parser.py       # Pure-Python container inspection & SHA-256 calculation
│   ├── annotation_schema.py  # JSON Schema specification & VideoAnnotation validation
│   ├── dataset_manager.py    # Non-destructive zip extraction & manifest builder
│   ├── preprocessing.py      # OpenCV uniform sampling (8/16f) & ImageNet normalization
│   ├── feature_extractor.py  # Frozen MobileNetV3-Small/ResNet18 backbones
│   ├── feature_cache.py      # Configuration-isolated atomic NPZ cache manager
│   ├── benchmark.py          # CPU latency & throughput benchmarking engine
│   ├── models.py             # TemporalConv1DNet, TemporalPoolingBaseline, GRU, FocalLoss
│   ├── train.py              # Pre-training verification gate & group-aware splitting
│   ├── evaluate.py           # Calibration (ECE), confusion matrix, latency percentiles
│   ├── inference.py          # Production CPU inference engine & temporal frame saliency
│   ├── dashboard.py          # Gradio 6 interactive security dashboard
│   └── cli.py                # Comprehensive unified CLI dispatcher
├── tests/
│   ├── test_video_parser.py
│   ├── test_annotation_schema.py
│   ├── test_manifest.py
│   ├── test_feature_pipeline.py
│   ├── test_models.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_inference.py
│   ├── test_dashboard.py
│   └── test_integration.py   # Full lifecycle integration tests (10 steps)
├── data/
│   ├── annotations/          # schema.json, manifest.jsonl, manifest.csv
│   ├── evaluations/          # evaluation_metrics.json, summary.csv, batch_inference.json
│   ├── processed/            # features_mobilenet_v3_small_8f_224.npz
│   └── raw/                  # Extracted video clips (carving/ and normal/)
├── vandalism_cpu.py          # Legacy prototype baseline (preserved untouched)
├── dataset_tool.py           # CLI entrypoint script
├── requirements.txt          # Python dependencies
├── DATASET_GUIDELINES.md     # Legal, privacy (DPDP/GDPR), ASI protocol & annotation instructions
├── TROUBLESHOOTING.md        # Common issues, recovery procedures, and FAQ
└── README.md                 # Complete system documentation
```

---

## 9. Legal & Ethical Notice

Predictions output by this AI model represent statistical temporal motion patterns. They do **not** constitute legal evidence of physical stone vandalism or malicious intent. In accordance with Archaeological Survey of India (ASI) standards and the India DPDP Act 2023, automated alerts require on-site human verification by authorized heritage conservation officers before enforcement actions are taken.
