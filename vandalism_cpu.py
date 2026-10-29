"""
Vandalism (carving) vs normal video classifier - CPU only, no NVIDIA needed.

Method: ResNet18 (pretrained) reads 8 frames per video -> feature numbers ->
        Logistic Regression classifier.

Usage (in the VS Code terminal):
    python vandalism_cpu.py            # extract features + train + test report
    python vandalism_cpu.py demo       # launch the interactive dashboard
    python vandalism_cpu.py predict "C:\\path\\to\\video.mp4"

Feature cache is incremental: when you add videos, only NEW videos are processed.
"""
import os
import sys
import time
import warnings
from datetime import datetime

import cv2
import joblib
import numpy as np
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ===================== SETTINGS =====================
DATA_DIR = r"C:\Users\kprit\Downloads\vandalism_final"   # must contain carving\ and normal\
CLASSES = ["normal", "carving"]        # index 0 = normal, index 1 = carving (vandalism)
NUM_FRAMES = 8                         # frames per video
IMG_SIZE = 224
SEED = 42
FEATURE_CACHE = "features_cache.npz"
MODEL_FILE = "vandalism_model.joblib"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv")

# Demo verdict bands (design choice, state this in your report)
VANDALISM_THRESHOLD = 0.80   # default: carving probability >= this -> VANDALISM detected
UNCERTAIN_THRESHOLD = 0.50   # between 0.50 and the alert threshold -> UNCERTAIN, human review

# Shown in the dashboard only if the saved model has no stored metrics
# (retraining with this script stores the real numbers automatically).
METRICS_FALLBACK = dict(accuracy=0.837, precision=0.643, recall=0.818, f1=0.720,
                        n_test=43, n_carving_test=11)
# ====================================================

torch.set_num_threads(os.cpu_count() or 4)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_extractor = None
_model_cache = None


def get_extractor():
    """Load ResNet18 once (downloads ~45 MB the first time, needs internet)."""
    global _extractor
    if _extractor is None:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()   # output 512 numbers per frame
        model.eval()
        _extractor = model
    return _extractor


def read_frames(path, n=NUM_FRAMES):
    """Return n evenly spaced RGB frames (resized), or None if unreadable."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, n).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    if len(frames) < n:  # fallback: read everything, then sample
        cap.release()
        cap = cv2.VideoCapture(path)
        allf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            allf.append(frame)
        if not allf:
            cap.release()
            return None
        idxs = np.linspace(0, len(allf) - 1, n).astype(int)
        frames = [allf[i] for i in idxs]
    cap.release()
    out = []
    for f in frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (IMG_SIZE, IMG_SIZE))
        out.append(f)
    return out


def frames_to_features(frames):
    """One feature vector (1024 numbers) from a list of RGB frames."""
    x = np.stack(frames).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2)  # (n, 3, H, W)
    with torch.no_grad():
        feats = get_extractor()(x).numpy()       # (n, 512)
    return np.concatenate([feats.mean(axis=0), feats.max(axis=0)])


def video_features(path):
    """One feature vector for a whole video, or None if unreadable."""
    frames = read_frames(path)
    if frames is None:
        return None
    return frames_to_features(frames)


def list_videos():
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            sys.exit(f"ERROR: folder not found: {folder}")
        for root, _, files in os.walk(folder):
            for fn in sorted(files):
                if fn.lower().endswith(VIDEO_EXTS):
                    paths.append(os.path.normpath(os.path.join(root, fn)))
                    labels.append(label)
    return paths, np.array(labels)


def build_features():
    """Incremental: reuse cached features, only process videos not seen before."""
    paths, labels = list_videos()
    print(f"Found {int((labels == 1).sum())} carving and {int((labels == 0).sum())} normal videos in folders.")

    cache = {}
    if os.path.exists(FEATURE_CACHE):
        d = np.load(FEATURE_CACHE, allow_pickle=True)
        for p, f, l in zip(d["paths"], d["X"], d["y"]):
            cache[os.path.normpath(str(p))] = (f, int(l))
        print(f"Loaded {len(cache)} videos from cache.")

    todo = [(p, l) for p, l in zip(paths, labels) if p not in cache or cache[p][1] != int(l)]
    print(f"New or changed videos to process: {len(todo)}")

    t0 = time.time()
    for i, (p, lab) in enumerate(todo, 1):
        f = video_features(p)
        if f is None:
            print(f"  skipped unreadable video: {p}")
            continue
        cache[p] = (f, int(lab))
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)} done | elapsed {el/60:.1f} min | ~{eta/60:.1f} min left")

    X, y, kept = [], [], []
    for p, l in zip(paths, labels):
        if p in cache:
            X.append(cache[p][0])
            y.append(cache[p][1])
            kept.append(p)
    X, y, kept = np.array(X), np.array(y), np.array(kept)
    np.savez(FEATURE_CACHE, X=X, y=y, paths=kept)
    print(f"Saved features to {FEATURE_CACHE}")
    return X, y, kept


def train():
    X, y, paths = build_features()
    print(f"\nTotal usable videos: {len(y)} (carving={int((y==1).sum())}, normal={int((y==0).sum())})")

    # Split by VIDEO (each row = one video), stratified: 70% train / 15% val / 15% test
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    print(f"Split -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"Carving in test set: {int((y[te]==1).sum())}")

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])

    best_c, best_score, best_clf = None, -1, None
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        clf.fit(Xtr, y[tr])
        pv = clf.predict(Xva)
        score = f1_score(y[va], pv, pos_label=1)
        print(f"  C={c:<6} val carving recall={recall_score(y[va], pv, pos_label=1):.2f}  F1={score:.2f}")
        if score > best_score:
            best_c, best_score, best_clf = c, score, clf
    print(f"Best C = {best_c}")

    pred = best_clf.predict(Xte)
    print("\n================ TEST REPORT ================")
    print(classification_report(y[te], pred, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows = true, columns = predicted)")
    print(f"labels order: {CLASSES}")
    print(confusion_matrix(y[te], pred))

    t0 = time.time()
    video_features(str(paths[0]))
    print(f"\nTime to process one clip (frames + features): {time.time()-t0:.2f} s")

    metrics = dict(
        accuracy=float(accuracy_score(y[te], pred)),
        precision=float(precision_score(y[te], pred, pos_label=1, zero_division=0)),
        recall=float(recall_score(y[te], pred, pos_label=1)),
        f1=float(f1_score(y[te], pred, pos_label=1)),
        n_test=int(len(te)),
        n_carving_test=int((y[te] == 1).sum()),
    )
    joblib.dump({"scaler": scaler, "clf": best_clf, "classes": CLASSES, "metrics": metrics}, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}  (no need to train again)")


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_FILE):
            sys.exit(f"ERROR: {MODEL_FILE} not found. Run: python vandalism_cpu.py")
        _model_cache = joblib.load(MODEL_FILE)
    return _model_cache


def prob_from_frames(frames):
    m = load_model()
    f = frames_to_features(frames)
    return float(m["clf"].predict_proba(m["scaler"].transform([f]))[0][1])


def predict_video(path):
    frames = read_frames(path)
    if frames is None:
        return None
    p = prob_from_frames(frames)
    return {"normal": 1 - p, "carving (vandalism)": p}


def verdict_from_prob(p, alert_thr=VANDALISM_THRESHOLD):
    if p >= alert_thr:
        return "VANDALISM (carving) detected"
    if p >= UNCERTAIN_THRESHOLD:
        return "UNCERTAIN - suspicious behaviour, needs human review"
    return "Normal"


# ===================== DASHBOARD =====================
def verdict_card(p, alert_thr):
    """Big colour-coded result card (solid colours so it looks fine in light and dark mode)."""
    if p is None:
        return ('<div style="padding:28px;border-radius:14px;background:#475569;color:white;text-align:center;">'
                '<div style="font-size:42px;">&#127909;</div>'
                '<div style="font-size:22px;font-weight:700;">Waiting for a video</div>'
                '<div style="opacity:.85;margin-top:6px;">Upload a clip and press Analyze</div></div>')
    if p >= alert_thr:
        bg, icon, title, sub = "#dc2626", "&#128680;", "VANDALISM DETECTED", "Carving behaviour found. Alert the guard and save the evidence clip."
    elif p >= UNCERTAIN_THRESHOLD:
        bg, icon, title, sub = "#d97706", "&#9888;&#65039;", "UNCERTAIN", "Suspicious behaviour. Needs human review."
    else:
        bg, icon, title, sub = "#16a34a", "&#9989;", "NORMAL", "No wall-carving behaviour detected."
    pct = p * 100
    return (f'<div style="padding:24px;border-radius:14px;background:{bg};color:white;text-align:center;">'
            f'<div style="font-size:42px;">{icon}</div>'
            f'<div style="font-size:26px;font-weight:800;letter-spacing:1px;">{title}</div>'
            f'<div style="font-size:15px;opacity:.95;margin:6px 0 14px;">{sub}</div>'
            f'<div style="background:rgba(255,255,255,.3);border-radius:10px;height:16px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:16px;background:white;"></div></div>'
            f'<div style="margin-top:8px;font-size:15px;">Carving probability: <b>{pct:.0f}%</b> '
            f'&nbsp;|&nbsp; alert at {alert_thr*100:.0f}%</div></div>')


def header_html():
    return ('<div style="padding:22px;border-radius:14px;background:#1e293b;color:white;text-align:center;">'
            '<div style="font-size:30px;font-weight:800;">&#127963;&#65039; Guarding the Stones</div>'
            '<div style="opacity:.85;margin-top:4px;">Vandalism (carving) detector for heritage walls &nbsp;|&nbsp; '
            'runs fully on CPU, no cloud</div></div>')


def about_markdown():
    try:
        mt = load_model().get("metrics") or METRICS_FALLBACK
    except SystemExit:
        mt = METRICS_FALLBACK
    return f"""
**Pipeline:** upload video -> OpenCV samples 8 frames -> ResNet18 (pretrained, frozen) turns each frame into 512 numbers
-> mean + max pooling gives 1024 numbers per video -> Logistic Regression gives the carving probability -> verdict.

**Verdict bands:** under 50% = Normal, 50% up to the alert threshold = Uncertain (human review),
at or above the alert threshold = Vandalism. Move the slider to see how the verdict changes.

**Test results on {mt['n_test']} unseen videos ({mt['n_carving_test']} carving):**
accuracy {mt['accuracy']*100:.1f}% | carving recall {mt['recall']*100:.1f}% |
carving precision {mt['precision']*100:.1f}% | carving F1 {mt['f1']:.2f}

**Limits:** frame features only (no motion), small test set, some false alarms on suspicious-but-harmless behaviour
(for example a person peeking into a car).
"""


def demo():
    import gradio as gr

    HIST_HEADERS = ["Time", "Video", "Carving %", "Verdict", "Seconds"]

    def analyze(video, alert_thr, history):
        history = history or []
        if video is None:
            return verdict_card(None, alert_thr), None, [], None, history, history
        t0 = time.time()
        frames = read_frames(video)
        if frames is None:
            err = ('<div style="padding:24px;border-radius:14px;background:#7f1d1d;color:white;text-align:center;">'
                   'Could not read this video. Try an .mp4 file.</div>')
            return err, None, [], None, history, history
        p = prob_from_frames(frames)
        secs = time.time() - t0
        row = [datetime.now().strftime("%H:%M:%S"), os.path.basename(video),
               f"{p*100:.0f}%", verdict_from_prob(p, alert_thr), f"{secs:.2f}"]
        history = [row] + history
        gallery = [(f, f"Frame {i+1}") for i, f in enumerate(frames)]
        label = {"carving (vandalism)": p, "normal": 1 - p}
        return verdict_card(p, alert_thr), label, gallery, p, history, history

    def rerender(p, alert_thr):
        # Slider moved: update the verdict instantly without re-running the model
        return verdict_card(p, alert_thr)

    def clear_all():
        return None, verdict_card(None, VANDALISM_THRESHOLD), None, [], None, [], []

    with gr.Blocks(title="Guarding the Stones") as app:
        gr.HTML(header_html())
        last_p = gr.State(None)
        hist = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="1. Upload a CCTV clip")
                thr = gr.Slider(0.50, 0.95, value=VANDALISM_THRESHOLD, step=0.05,
                                label="Alert threshold (lower = more sensitive, more false alarms)")
                with gr.Row():
                    btn = gr.Button("Analyze", variant="primary")
                    clr = gr.Button("Clear")
            with gr.Column(scale=1):
                card = gr.HTML(verdict_card(None, VANDALISM_THRESHOLD))
                label = gr.Label(num_top_classes=2, label="Model probabilities")

        gr.Markdown("### What the model sees (8 frames sampled from the clip)")
        gallery = gr.Gallery(label="Sampled frames", columns=4, height=300)

        gr.Markdown("### Session history")
        table = gr.Dataframe(headers=HIST_HEADERS, interactive=False)

        with gr.Accordion("How it works and test results", open=False):
            gr.Markdown(about_markdown())

        btn.click(analyze, [video, thr, hist], [card, label, gallery, last_p, hist, table])
        thr.change(rerender, [last_p, thr], card)
        clr.click(clear_all, None, [video, card, label, gallery, last_p, hist, table])

    app.launch()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    elif mode == "demo":
        demo()
    elif mode == "predict" and len(sys.argv) > 2:
        res = predict_video(sys.argv[2])
        print(res)
        if res:
            print(verdict_from_prob(res["carving (vandalism)"]))
    else:
        print(__doc__)
