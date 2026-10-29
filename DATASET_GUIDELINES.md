# Dataset Licensing, Ethics & Privacy Guidelines
## Project: Guarding the Stones — Egocentric Heritage Protection AI

---

## 1. Executive Summary & Purpose

The **Guarding the Stones** project aims to detect fine-grained, calm physical defacement (micro-vandalism such as carving, scratching, and marking) on protected cultural heritage monument surfaces (such as ASI-protected sites: Ajanta Caves, Purana Qila, Qutub Minar, Hampi).

Action recognition algorithms targeting body-worn smart glasses or guard-worn body cameras require high-quality, authentic **first-person (egocentric)** video data. This document outlines the legal, ethical, licensing, and privacy compliance standards for dataset curation, annotation, and field collection.

---

## 2. Licensing Audit of Existing Video Assets

### 2.1 Commercial Stock Footage (`carving.zip`)
1. **Provenance & Watermarks**:
   - The initial archive (`carving.zip`, 223 clips) contains commercial stock video clips bearing embedded *"iStock by Getty Images"* watermarks.
   - **Legal Restrictions**: iStock Standard and Extended Content Licenses expressly prohibit redistribution of raw video files or incorporating them into publicly distributable machine learning benchmarks without written enterprise licensing.
   - **Machine Learning Risk (Shortcut Bias / Clever Hans Effect)**: Neural network backbones easily learn watermarks, studio color grading, or cinematic camera pan patterns as spurious correlations for "vandalism", causing catastrophic out-of-distribution failure when deployed in field cameras.
   - **Recommendation**: Existing stock clips should serve strictly as an internal development pipeline mock. They must be phased out in favor of an ethically collected, egocentric benchmark.

### 2.2 Recommended Licensing for Future Dataset Releases
- **Permissive Open Science**: Datasets curated under project auspices should target **Creative Commons Attribution 4.0 International (CC-BY 4.0)** or **Open Data Commons Open Database License (ODbL)**.
- **Code & Tooling**: Released under Apache 2.0 or MIT License.

---

## 3. Privacy, Data Protection & Regulatory Compliance

Field collection and processing of video footage must comply with applicable data privacy laws, notably:
- **India**: Digital Personal Data Protection Act, 2023 (DPDP Act 2023)
- **European Union**: General Data Protection Regulation (GDPR - Regulation 2016/679)
- **Archaeological Survey of India (ASI) Regulations**: Photography and filming bylaws at protected heritage monuments.

### 3.1 Personally Identifiable Information (PII) Protocols
Egocentric cameras capture the wearer's surroundings, including faces, clothing, and voices of innocent tourists, children, and site employees.

1. **Automated Facial Anonymization**:
   - All non-actor faces visible in video frames must be blurred or pixelated prior to dataset ingestion or external sharing.
   - License plates, badges, and personal ID cards must be masked.
2. **Audio Stripping**:
   - Unless acoustic tool-contact sound is a designated research variable, raw audio streams must be stripped to prevent eavesdropping or ambient speech logging.
3. **Right to Erasure (DPDP Section 12 / GDPR Article 17)**:
   - If an identifiable individual requests exclusion, the corresponding video segment must be excised from raw storage and manifest registries.

### 3.2 Participant & Actor Consent
When conducting staged micro-vandalism simulations:
- **Informed Consent Forms**: All participating actors must sign written consent acknowledging that video recordings will be used for AI research and public safety system development.
- **Vulnerable Populations**: Minors must never be recorded performing vandalism simulations.

---

## 4. Egocentric Heritage Data Collection Protocol

To resolve the complete absence of authentic first-person footage identified in the Part 1 audit, subsequent data collection must follow this standardized protocol:

### 4.1 Camera & Mounting Hardware
- **Hardware Profile**: Smart glasses (e.g. Ray-Ban Meta, Rokid, or custom 3D-printed head-band mounted action cameras).
- **Camera Mount**: Head or temple mount is strictly preferred over chest mount to ensure the camera optical axis aligns with the wearer's gaze and hand manipulations.
- **Specifications**:
  - Resolution: $1920 \times 1080$ (1080p) minimum
  - Frame Rate: 30 FPS or 60 FPS (higher frame rate preserves key/coin stroke dynamics)
  - Field of View (FOV): $100^\circ - 120^\circ$ (captures natural hand reach space)

### 4.2 Non-Destructive Simulation Guidelines at Heritage Monuments
> [!CAUTION]
> **Strict Heritage Preservation Rule**: Under no circumstances should actual monument surfaces, historic stones, or cultural artifacts be scratched, chipped, or altered during data collection. Doing so violates the Ancient Monuments and Archaeological Sites and Remains Act (AMASR Act, 1958) and carries criminal penalties.

#### Allowed Simulation Methods:
1. **Portable Mock Substrates**:
   - Bring uncarved, modern quarry sandstone, granite, or brick tiles to the site vicinity.
   - Mount mock stones on portable easels and simulate carving actions on the mock surface against monument ambient lighting.
2. **Zero-Contact Gestures (Air Carving)**:
   - Actors hold keys, coins, or styluses and perform authentic micro-scratching finger motions $1 - 2\text{ cm}$ away from modern perimeter walls without making contact.
3. **Dry Removable Chalk / Water Stylus**:
   - On authorized modern mock stones, use dry slate chalk or water-wetted brushes that evaporate immediately without leaving residue.

---

## 5. Candidate Action Class Taxonomy

| Class ID | Action Name | Category | Description | Primary Visual Cues |
| :--- | :--- | :--- | :--- | :--- |
| `01` | `normal_looking` | Benign | Passive viewing; head panning across wall; hands at sides or in pockets. | Empty hand zone; steady head motion. |
| `02` | `walking` | Benign | Locomotion along pathways or site corridors; camera oscillation. | Cyclic vertical camera shake; changing background. |
| `03` | `pointing` | Benign | Index finger extended to highlight architectural detail; hand retreats. | Transient hand extension; no surface contact. |
| `04` | `photographing` | Benign | Holding mobile device or camera in two hands to frame monument shot. | Both hands centered holding rectangular phone/camera. |
| `05` | `permitted_touching` | Benign | Flat palm or open finger touch on designated interactive / tactile areas. | Open palm, broad contact, gentle flat palm motion. |
| `06` | `inspection` | Benign | Close visual inspection by guide, scholar, or guard; leaning in. | Proximity decrease, slow head tilting, hands non-destructive. |
| `07` | `potential_scratching`| Vandalism | Repeated fine motor dragging of hard object (key, coin, nail) on stone. | Precision pinch grip; high-frequency linear/circular stroke. |
| `08` | `potential_scraping` | Vandalism | Flat or chisel-edge dragging; planar abrasive friction motion. | Firm lateral arm motion; tool contact angle $< 45^\circ$. |
| `09` | `potential_chipping` | Vandalism | Percussive tap or impact motion aimed at dislodging surface stone. | Back-and-forth wrist flick / hammer gesture toward stone. |
| `10` | `potential_marking`  | Vandalism | Application of marker, pen, chalk, lipstick, or spray can. | Continuous handwriting or pressurized nozzle spraying. |
| `11` | `ambiguous_interaction`| Edge Case | Hand obscured by body or bag near wall; suspicious prolonged loitering. | Partial occlusion, boundary condition requiring human review. |

---

## 6. Verification and Audit Sign-Off

All dataset manifests (`manifest.jsonl`, `manifest.csv`) must be generated via `dataset_tool.py` and accompanied by a corresponding hash checksum. No data file should enter training or evaluation pipelines without an explicit `verified_by_human: true` annotation record.

---

## 7. Step-by-Step Dataset Annotation Instructions

To prepare authentic ground-truth labels for model training and evaluation:

### Step 7.1: Generate Annotation Template
Generate a pre-filled JSON template with all discovered video IDs and filenames:
```bash
python dataset_tool.py template --out data/annotations/annotations_workfile.json
```

### Step 7.2: Apply Ground-Truth Annotations
Open the JSON workfile and fill each entry following these mandatory rules:
1. **`action_label`**: Select exactly one of the 11 candidate classes defined in Section 5. Never invent custom names.
2. **`camera_viewpoint`**: Must be `"egocentric"` (first-person body-worn/smart glasses). Third-person clips (`"stock_footage"`, `"third_person"`, `"surveillance_cctv"`) must be identified as such and excluded from egocentric training splits.
3. **`verified_by_human`**: Set to `true` once human review of the video stream is complete. The training gate strictly rejects files where `verified_by_human` is `false`.
4. **`start_sec` and `end_sec`**: Define the temporal interval containing the primary action ($0 \le start\_sec < end\_sec \le video\_duration$).
5. **`annotator_id`**: Enter the unique ID of the annotator (e.g., `"ASI_CONSERVATOR_04"`).
6. **`confidence`**: Enter confidence rating in $[0.0, 1.0]$.
7. **`tool_used`**: Record object in hand (`"key"`, `"coin"`, `"chalk"`, `"phone"`, `"none"`).

### Step 7.3: Validate Against JSON Schema
Verify that the annotation JSON strictly adheres to the schema:
```bash
python dataset_tool.py validate --annotations data/annotations/annotations_workfile.json
```

### Step 7.4: Compile the Verified Manifest
Re-generate the production manifest integrating your verified annotations:
```bash
python dataset_tool.py manifest --annotations data/annotations/annotations_workfile.json
```
This updates `data/annotations/manifest.jsonl` and `data/annotations/manifest.csv` for training and evaluation.
