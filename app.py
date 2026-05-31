from unittest import result

import streamlit as st
import numpy as np
import cv2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import joblib
import os
import time
import math
from PIL import Image
from io import BytesIO
from scipy.signal import wiener
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import ImageDraw, ImageFont
import tempfile
from datetime import datetime

# ─────────────────────────────────────────────
# MEDIAPIPE — mapping dari dlib 68-point ke Face Mesh
# ─────────────────────────────────────────────
#
# dlib idx → MediaPipe Face Mesh idx (approx equivalent)
#   0  (jaw left)      → 234
#   1  (jaw left+1)    → 227
#   8  (chin)          → 152
#  15  (jaw right-1)   → 447
#  16  (jaw right)     → 454
#  17  (brow left L)   → 70
#  19  (brow left mid) → 66
#  26  (brow right R)  → 296
#  27  (nose bridge)   → 168
#  28-35 (nose ridge)  → 168,6,197,195,5,4,1,2   (range 27-36)
#  17-26 (both brows)  → mapped below
#

def download_face_landmarker():
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "face_landmarker.task")

    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    return model_path
    
MP_LANDMARK_MAP = {
    0:  234,   # jaw far left
    1:  227,   # jaw left
    8:  152,   # chin bottom
    15: 447,   # jaw right
    16: 454,   # jaw far right
    17: 70,    # left brow outer
    18: 63,
    19: 66,    # left brow mid
    20: 65,
    21: 55,
    22: 285,
    23: 295,
    24: 282,
    25: 283,
    26: 296,   # right brow outer
    27: 168,   # nose bridge top
    28: 6,
    29: 197,
    30: 195,
    31: 5,
    32: 4,
    33: 1,
    34: 19,
    35: 94,    # nose tip
}

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR      = os.path.join(BASE_DIR, "models")
FOUNDATION_CSV = os.path.join(BASE_DIR, "foundation_mst.csv")

FEATURE_COLS = [
    'cheek_L_mean', 'cheek_L_std', 'cheek_a_mean', 'cheek_a_std',
    'cheek_b_mean', 'cheek_b_std', 'cheek_ITA',
    'forehead_L_mean', 'forehead_L_std', 'forehead_a_mean', 'forehead_a_std',
    'forehead_b_mean', 'forehead_b_std', 'forehead_ITA',
    'nose_L_mean', 'nose_L_std', 'nose_a_mean', 'nose_a_std',
    'nose_b_mean', 'nose_b_std', 'nose_ITA',
    'global_L_mean', 'global_L_std', 'global_a_mean', 'global_a_std',
    'global_b_mean', 'global_b_std', 'global_ITA',
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # MediaPipe Face Mesh
    model_path = download_face_landmarker()
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(options)

    ensemble = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler   = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    kmeans_path = None
    for f in os.listdir(MODEL_DIR):
        if f.startswith("kmeans_k") and f.endswith(".pkl"):
            kmeans_path = os.path.join(MODEL_DIR, f)
            break
    if kmeans_path is None:
        raise FileNotFoundError("kmeans_k*.pkl tidak ditemukan di MODEL_DIR")
    kmeans = joblib.load(kmeans_path)

    df_found  = pd.read_csv(FOUNDATION_CSV)
    centroids = (
        df_found.groupby("mst_id")[["lab_L", "lab_a", "lab_b"]]
        .median()
        .rename(columns={"lab_L": "L_ref", "lab_a": "a_ref", "lab_b": "b_ref"})
        .reset_index()
    )
    mst_hex_lookup = (
        df_found.drop_duplicates("mst_id")
        .set_index("mst_id")["mst_hex"]
        .to_dict()
    )
    return face_mesh, ensemble, scaler, kmeans, df_found, centroids, mst_hex_lookup


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img):
    lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_blur = cv2.GaussianBlur(img_norm, (5, 5), 1.0)
    result = np.zeros_like(img_blur, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = wiener(img_blur[:, :, c].astype(np.float32), mysize=5)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# DETEKSI LANDMARK (MediaPipe → dlib-style list)
# ─────────────────────────────────────────────
def detect_landmarks(img_rgb, face_mesh):
    import mediapipe as mp_lib
    h, w = img_rgb.shape[:2]
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=img_rgb)
    results = face_mesh.detect(mp_image)
    if not results.face_landmarks:
        return None, None

    mp_lms = results.face_landmarks[0]

    lms = {}
    for dlib_idx, mp_idx in MP_LANDMARK_MAP.items():
        pt = mp_lms[mp_idx]
        lms[dlib_idx] = (int(pt.x * w), int(pt.y * h))

    xs = [p[0] for p in lms.values()]
    ys = [p[1] for p in lms.values()]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    return lms, bbox


# ─────────────────────────────────────────────
# MASK HELPERS — identik dengan notebook
# ─────────────────────────────────────────────
def make_cheek_ellipse_mask(img_shape, landmarks):
    h, w   = img_shape[:2]
    mid_y  = (landmarks[27][1] + landmarks[8][1]) // 2
    face_w = landmarks[16][0] - landmarks[0][0]
    ew, eh = int(face_w * 0.22), int(face_w * 0.15)
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (landmarks[1][0] + ew, mid_y),  (ew, eh), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (landmarks[15][0] - ew, mid_y), (ew, eh), 0, 0, 360, 1, -1)
    return mask.astype(bool)

def make_forehead_mask(img_shape, landmarks):
    h, w    = img_shape[:2]
    brow_y  = int(np.mean([landmarks[i][1] for i in range(17, 27)]))
    brow_lx = landmarks[17][0]
    brow_rx = landmarks[26][0]
    face_h  = landmarks[8][1] - landmarks[19][1]
    top_y   = max(0, brow_y - int(face_h * 0.35))
    pts  = np.array([[brow_lx, top_y], [brow_rx, top_y],
                     [brow_rx, brow_y], [brow_lx, brow_y]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def make_nose_mask(img_shape, landmarks):
    h, w     = img_shape[:2]
    nose_pts = np.array([landmarks[i] for i in range(27, 36)], dtype=np.int32)
    hull     = cv2.convexHull(nose_pts)
    mask     = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    return mask.astype(bool)

def filter_skin_pixels(lab_pixels):
    mask = (
        (lab_pixels[:, 0] >= 20) & (lab_pixels[:, 0] <= 97) &
        (lab_pixels[:, 1] >= 3)  & (lab_pixels[:, 1] <= 30)
    )
    return lab_pixels[mask]


# ─────────────────────────────────────────────
# EKSTRAKSI FITUR
# ─────────────────────────────────────────────
def get_skin_features(img_rgb, lms):
    from skimage.color import rgb2lab as skimage_rgb2lab
    
    # Pastikan hanya 3 channel RGB, buang alpha jika ada
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]
    
    lab        = skimage_rgb2lab(img_rgb.astype(np.float32) / 255.0)
    all_pixels = []
    feats      = {}

    zones = {
        'cheek'   : make_cheek_ellipse_mask(img_rgb.shape, lms),
        'forehead': make_forehead_mask(img_rgb.shape, lms),
        'nose'    : make_nose_mask(img_rgb.shape, lms),
    }

    for zone_name, mask in zones.items():
        if mask.sum() < 10:
            for s in ['L_mean','L_std','a_mean','a_std','b_mean','b_std','ITA']:
                feats[f"{zone_name}_{s}"] = 0.0
            continue
        px = lab[mask]
        px = filter_skin_pixels(px)
        if len(px) < 5:
            for s in ['L_mean','L_std','a_mean','a_std','b_mean','b_std','ITA']:
                feats[f"{zone_name}_{s}"] = 0.0
            continue
        all_pixels.append(px)
        for ci, ch in enumerate(['L', 'a', 'b']):
            feats[f'{zone_name}_{ch}_mean'] = float(px[:, ci].mean())
            feats[f'{zone_name}_{ch}_std']  = float(px[:, ci].std())
        # FIX Bug 1: Formula ITA yang benar adalah atan2(L - 50, b)
        # L dikurangi 50 sesuai standar ilmiah ITA dan konsisten dengan predict_mst_hybrid()
        feats[f'{zone_name}_ITA'] = math.degrees(
            math.atan2(px[:, 0].mean() - 50, px[:, 2].mean())
        )

    if not all_pixels:
        return None

    combined = np.vstack(all_pixels)
    for ci, ch in enumerate(['L', 'a', 'b']):
        feats[f'global_{ch}_mean'] = float(combined[:, ci].mean())
        feats[f'global_{ch}_std']  = float(combined[:, ci].std())
    # FIX Bug 1: Formula ITA global juga harus L - 50
    feats['global_ITA'] = math.degrees(
        math.atan2(combined[:, 0].mean() - 50, combined[:, 2].mean())
    )
    return feats


# ─────────────────────────────────────────────
# PREDIKSI HYBRID
# ─────────────────────────────────────────────
def predict_mst_hybrid(feats, ensemble, scaler, kmeans, centroids, feature_cols,
                        alpha=0.40, temperature=0.6, sigma_eucl=2.0, sigma_ita=4.0):
    x    = np.array([[feats.get(c, 0.0) for c in feature_cols]])
    x_sc = scaler.transform(x)
    dist = kmeans.transform(x_sc)
    x_aug = np.hstack([x_sc, dist])

    model_proba   = ensemble.predict_proba(x_aug)[0]
    model_classes = ensemble.classes_

    log_p = np.log(model_proba + 1e-10) / temperature
    model_proba = np.exp(log_p - log_p.max())
    model_proba = model_proba / model_proba.sum()

    L_inp   = feats.get('global_L_mean', 50)
    a_inp   = feats.get('global_a_mean', 8)
    b_inp   = feats.get('global_b_mean', 12)
    ita_inp = math.degrees(math.atan2(L_inp - 50, b_inp))

    mst_keys = centroids['mst_id'].values

    dist_arr     = np.sqrt(
        (centroids['L_ref'].values - L_inp)**2 +
        (centroids['a_ref'].values - a_inp)**2 +
        (centroids['b_ref'].values - b_inp)**2
    )
    inv_dist     = np.exp(-dist_arr / sigma_eucl)
    db_proba_lab = inv_dist / inv_dist.sum()

    ita_centroids = np.degrees(np.arctan2(
        centroids['L_ref'].values - 50,
        centroids['b_ref'].values
    ))
    ita_dist     = np.abs(ita_centroids - ita_inp)
    inv_ita      = np.exp(-ita_dist / sigma_ita)
    db_proba_ita = inv_ita / inv_ita.sum()

    db_proba = 0.60 * db_proba_lab + 0.40 * db_proba_ita

    combined = {}
    for i, mst in enumerate(mst_keys):
        idx     = np.where(model_classes == mst)[0]
        model_p = float(model_proba[idx[0]]) if len(idx) > 0 else 0.0
        combined[mst] = (1 - alpha) * model_p + alpha * float(db_proba[i])

    best_mst = max(combined, key=combined.get)
    total    = sum(combined.values())

    top3_candidates = sorted(combined.items(), key=lambda x: -x[1])
    top3 = [item for item in top3_candidates if abs(item[0] - best_mst) <= 2][:3]
    if len(top3) < 3:
        remaining = [item for item in top3_candidates if item not in top3]
        top3 += sorted(remaining, key=lambda x: abs(x[0] - best_mst))[:3 - len(top3)]

    return (
        int(best_mst),
        round(combined[best_mst] / total * 100, 1),
        [{'mst': int(m), 'conf': round(p / total * 100, 1)} for m, p in top3]
    )


# ─────────────────────────────────────────────
# REKOMENDASI
# ─────────────────────────────────────────────
def recommend_foundation(mst_pred, L, a, b, df_found, top_n=6):
    df = df_found.copy()
    df['delta_e'] = np.sqrt(
        (df['lab_L'] - L)**2 +
        (df['lab_a'] - a)**2 +
        (df['lab_b'] - b)**2
    )
    mst_range  = [mst_pred - 1, mst_pred, mst_pred + 1]
    df_primary = df[df['mst_id'].isin(mst_range)].sort_values('delta_e')
    df_fallback= df[~df['mst_id'].isin(mst_range)].sort_values('delta_e')
    return pd.concat([df_primary, df_fallback]).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# HELPER: CIELAB → HEX
# ─────────────────────────────────────────────
def cielab_to_hex(L, a, b):
    from skimage.color import lab2rgb
    rgb = lab2rgb([[[ L, a, b ]]])[0][0]
    rgb = np.clip(rgb, 0, 1)
    r, g, b_ = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    return f"#{r:02x}{g:02x}{b_:02x}"

def format_rupiah(value):
    try:
        value = float(value)
        return f"Rp{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return value

def estimate_user_undertone(a, b):
    """
    Estimasi undertone pengguna dari nilai CIELAB.
    b* tinggi cenderung warm/yellowish,
    b* rendah cenderung cool,
    area tengah dianggap neutral.
    """
    if b >= 14:
        return "Warm"
    elif b <= 10:
        return "Cool"
    else:
        return "Neutral"


def estimate_user_skintone(mst):
    """
    Estimasi skintone berdasarkan prediksi MST.
    MST 1-3  : Light/Fair
    MST 4-6  : Medium
    MST 7-10 : Deep
    """
    if mst <= 3:
        return "Light/Fair"
    elif mst <= 6:
        return "Medium"
    else:
        return "Deep"
    
def classify_user_skintone_from_mst(mst):
    """
    Klasifikasi skintone pengguna berdasarkan hasil MST.
    MST 1-3  : Light/Fair
    MST 4-6  : Medium
    MST 7-10 : Deep
    """
    try:
        mst = int(mst)
    except:
        return "-"

    if mst <= 3:
        return "Light/Fair"
    elif mst <= 6:
        return "Medium"
    else:
        return "Deep"


def classify_user_undertone_from_lab(a, b):
    """
    Estimasi undertone pengguna dari nilai CIELAB.
    Heuristik sederhana:
    - b* jauh lebih tinggi dari a* -> Warm
    - a* lebih dominan dibanding b* -> Cool
    - selain itu -> Neutral
    """
    try:
        score = float(b) - float(a)
    except:
        return "-"

    if score >= 3:
        return "Warm"
    elif score <= -2:
        return "Cool"
    else:
        return "Neutral"

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(img_rgb, face_mesh, ensemble, scaler,
                 kmeans, centroids, df_found, mst_hex_lookup, feature_cols):
    t0 = time.time()

    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    h, w = img_rgb.shape[:2]
    if max(h, w) > 512:
        scale   = 512 / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

    lms, bbox = detect_landmarks(img_rgb, face_mesh)
    if lms is None:
        img_pre = preprocess_image(img_rgb)
        lms, bbox = detect_landmarks(img_pre, face_mesh)
    else:
        img_pre = preprocess_image(img_rgb)

    if lms is None:
        return None, "❌ Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah menghadap kamera."

    feats = get_skin_features(img_pre, lms)
    if feats is None:
        return None, "❌ Ekstraksi fitur gagal. Wajah terlalu kecil atau terhalang."

    mst, conf, top3 = predict_mst_hybrid(
        feats, ensemble, scaler, kmeans, centroids, feature_cols
    )

    top3_hex = [{"mst": t["mst"], "conf": t["conf"],
                 "hex": mst_hex_lookup.get(t["mst"], "#888888")} for t in top3]

    recs    = recommend_foundation(
        mst, feats["global_L_mean"], feats["global_a_mean"], feats["global_b_mean"],
        df_found, top_n=5
    )
    top_rec = recs.iloc[0]
    latency = round((time.time() - t0) * 1000, 1)

    vis = img_rgb.copy()
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 100), 2)
    for (px, py) in lms.values():
        cv2.circle(vis, (int(px), int(py)), 1, (255, 100, 0), -1)

    skin_hex = cielab_to_hex(
    feats["global_L_mean"],
    feats["global_a_mean"],
    feats["global_b_mean"]
    )
    
    user_undertone = estimate_user_undertone(
    feats["global_a_mean"],
    feats["global_b_mean"]
    )

    user_skintone = estimate_user_skintone(mst)
    
    return {
    "mst_pred"  : mst,
    "confidence": conf,
    "top3"      : top3_hex,
    "shade_name": top_rec["Shade"],
    "brand"     : top_rec["Brand"],
    "product"   : top_rec["Product"],
    "hex_color" : cielab_to_hex(top_rec["lab_L"], top_rec["lab_a"], top_rec["lab_b"]),
    "skin_hex"  : skin_hex,
    "user_undertone": user_undertone,
    "user_skintone" : user_skintone,
    "undertone" : top_rec["Undertone"],
    "price"     : format_rupiah(top_rec["Price"]),
    "top5_recs" : recs.to_dict(orient="records"),
    "cielab"    : {
        "L": round(feats["global_L_mean"], 2),
        "a": round(feats["global_a_mean"], 2),
        "b": round(feats["global_b_mean"], 2),
    },
    "latency_ms": latency,

    "vis_frame" : vis,

    "report_face_image": vis,
}, None
def load_font(size, bold=False):
    paths = [
        "arialbd.ttf" if bold else "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except:
            continue

    return ImageFont.load_default()

from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
from datetime import datetime
import numpy as np

# =========================================================
# HELPER
# =========================================================
def _load_font(size=24, bold=False):
    candidates = []
    if bold:
        candidates = [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except:
            continue
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        test = word if current == "" else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    return lines


def _draw_card(draw, xy, fill="#FFFFFF", outline="#E8CAD8", radius=24, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_text_block(draw, x, y, text, font, fill, max_width, line_gap=8):
    lines = _wrap_text(draw, text, font, max_width)
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=fill)
        bbox = draw.textbbox((x, yy), line, font=font)
        yy += (bbox[3] - bbox[1]) + line_gap
    return yy


import re
from pathlib import Path
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np


def _format_price(value):
    """Format harga menjadi Rp.239.000,00."""
    if value is None or value == "":
        return "-"
    try:
        if isinstance(value, (int, float)):
            integer = int(round(float(value)))
            return f"Rp.{integer:,}".replace(",", ".") + ",00"

        s = str(value).strip()
        if s in ["-", "nan", "None"]:
            return "-"

        s = s.replace("Rp", "").replace("rp", "").strip()
        s = re.sub(r",00$", "", s)
        digits = re.sub(r"[^\d]", "", s)
        if digits:
            integer = int(digits)
            return f"Rp.{integer:,}".replace(",", ".") + ",00"
        return str(value)
    except Exception:
        return str(value)


def _load_font(size=24, bold=False):
    candidates = []
    if bold:
        candidates = [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_text(draw, text, font, max_width):
    text = str(text)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    ell = "…"
    while text and draw.textbbox((0, 0), text + ell, font=font)[2] > max_width:
        text = text[:-1]
    return text + ell if text else ell


def _draw_card(draw, xy, fill="#FFFFFF", outline="#E8CAD8", radius=24, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_text_block(draw, x, y, text, font, fill, max_width, line_gap=8, max_lines=None):
    lines = _wrap_text(draw, text, font, max_width)
    if max_lines is not None:
        lines = lines[:max_lines]
        if len(_wrap_text(draw, text, font, max_width)) > max_lines and lines:
            lines[-1] = _fit_text(draw, lines[-1], font, max_width)
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=fill)
        bbox = draw.textbbox((x, yy), line, font=font)
        yy += (bbox[3] - bbox[1]) + line_gap
    return yy


def _extract_report_image(result):
    possible_keys = [
        "report_face_image",
        "vis_frame",
        "annotated_image",
        "face_preview",
        "frame_landmark_image",
        "result_image",
        "preview_image",
    ]
    for key in possible_keys:
        obj = result.get(key)
        if obj is None:
            continue
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, np.ndarray) and obj.ndim == 3:
            try:
                return Image.fromarray(obj.astype(np.uint8)).convert("RGB")
            except Exception:
                pass
    return None


def _draw_progress_bar(draw, x, y, w, h, percent, fill="#F48ABD", bg="#F2E8ED"):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=bg)
    try:
        p = max(0, min(100, float(percent)))
    except Exception:
        p = 0
    fw = int(w * (p / 100.0))
    if fw > 0:
        draw.rounded_rectangle((x, y, x + fw, y + h), radius=h // 2, fill=fill)


def _load_logo_image(size=(54, 54)):
    app_dir = Path(__file__).parent
    candidate_paths = [
        app_dir / "assets" / "logo.png",
        app_dir / "assets" / "logo.jpg",
        app_dir / "assets" / "logo.jpeg",
        app_dir / "assets" / "logo.webp",
        app_dir / "assets" / "logo.svg",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".svg":
                try:
                    import cairosvg
                    png_bytes = cairosvg.svg2png(url=str(path))
                    logo = Image.open(BytesIO(png_bytes)).convert("RGBA")
                except Exception:
                    continue
            else:
                logo = Image.open(path).convert("RGBA")
            logo.thumbnail(size, Image.LANCZOS)
            return logo
        except Exception:
            continue
    return None


def _paste_logo_or_fallback(base_img, draw, x, y, size=54):
    logo = _load_logo_image((size, size))
    if logo is not None:
        lx = x + (size - logo.width) // 2
        ly = y + (size - logo.height) // 2
        base_img.paste(logo, (lx, ly), logo)
    else:
        draw.rounded_rectangle((x, y, x + size, y + size), radius=16, fill="#F6D8E6", outline="#EBCFDB", width=2)
        draw.text((x + 15, y + 9), "✿", font=_load_font(24, bold=True), fill="#2F2330")


def _safe_hex_from_rec(rec, fallback="#D8C3BC"):
    try:
        hx = str(rec.get("hex_color", "")).strip()
        if hx.startswith("#") and len(hx) in [4, 7]:
            return hx
        if all(k in rec for k in ["lab_L", "lab_a", "lab_b"]):
            return cielab_to_hex(float(rec["lab_L"]), float(rec["lab_a"]), float(rec["lab_b"]))
    except Exception:
        pass
    return fallback


def _get_alt_mst(result, confidence):
    top3 = result.get("top3") or result.get("top_mst_alternatives") or []
    alts = []
    for item in top3:
        if isinstance(item, dict):
            if "mst" in item:
                alts.append({"label": f"MST-{item.get('mst')}", "score": float(item.get("conf", item.get("score", 0)) or 0), "hex": item.get("hex")})
            else:
                alts.append({"label": str(item.get("label", "-")), "score": float(item.get("score", 0) or 0), "hex": item.get("hex")})
    if not alts:
        alts = [{"label": f"MST-{result.get('mst_pred', '-')}", "score": confidence, "hex": None}]
    return alts


def create_analysis_report(result, dark_mode=True):
    mst = result.get("mst_pred", "-")
    confidence = float(result.get("confidence", result.get("mst_confidence", 0) or 0))
    skin_hex = result.get("skin_hex", "#B2806F")
    user_undertone = result.get("user_undertone", "-")
    user_skintone = result.get("user_skintone", "-")

    lab = result.get("cielab", {})
    L_val = lab.get("L", "-")
    a_val = lab.get("a", "-")
    b_val = lab.get("b", "-")

    brand = result.get("brand", "-")
    product = result.get("product", "-")
    shade = result.get("shade_name", result.get("shade", "-"))
    rec_undertone = result.get("undertone", "-")
    price = _format_price(result.get("price", "-"))
    top5 = result.get("top5_recs", [])
    alt_mst = _get_alt_mst(result, confidence)
    report_img = _extract_report_image(result)

    W = 1600
    if dark_mode:
        bg = "#1E171D"
        card_fill = "#2D222B"
        border = "#A65782"
        text_dark = "#F8EEF4"
        text_muted = "#D8C5D0"
        pink = "#FF9ED0"
        pink_soft = "#543247"
        green = "#C8DDA5"
        green_soft = "#33402B"
        blue = "#8DCCF5"
        orange = "#FFB07A"
        line = "#5E4454"
        frame_fill = "#372935"
        soft_outline = "#7A4A62"
        mst_fill = "#3A2935"
        row_fill = "#3A2935"
        badge_fill = "#5A3048"
        lab_blue_fill = "#263645"
        lab_pink_fill = "#4A2B3C"
        lab_orange_fill = "#4A3428"
        product_muted = "#CDB7C5"
        footer_muted = "#B99AAA"
    else:
        bg = "#FCF7FA"
        card_fill = "#FFFFFF"
        border = "#EBCFDB"
        text_dark = "#2F2330"
        text_muted = "#7A6674"
        pink = "#F48ABD"
        pink_soft = "#FFF0F6"
        green = "#758952"
        green_soft = "#EEF3E5"
        blue = "#66B4E8"
        orange = "#FF9A57"
        line = "#E9D7E0"
        frame_fill = "#FAF6F8"
        soft_outline = "#E9D7E0"
        mst_fill = "#FFF9FC"
        row_fill = "#FFF9FC"
        badge_fill = "#FCE8F1"
        lab_blue_fill = "#EEF6FF"
        lab_pink_fill = "#FFF1F1"
        lab_orange_fill = "#FFF6EB"
        product_muted = "#8A7682"
        footer_muted = "#A18897"

    font_title = _load_font(42, bold=True)
    font_sub = _load_font(20)
    font_h2 = _load_font(28, bold=True)
    font_label = _load_font(18, bold=True)
    font_text = _load_font(20)
    font_small = _load_font(16)
    font_tiny = _load_font(14)
    font_mst = _load_font(17, bold=True)
    font_conf = _load_font(14, bold=True)
    font_lab_title = _load_font(18, bold=True)
    font_lab_label = _load_font(15, bold=True)
    font_lab_value = _load_font(15, bold=True)
    font_hex = _load_font(20, bold=True)
    font_rec_label = _load_font(20, bold=True)
    font_rec_value = _load_font(22)
    font_top5_title = _load_font(19, bold=True)
    font_top5_meta = _load_font(16)

    margin = 50
    header_y1, header_y2 = 35, 155
    top_y = 190

    left_x1, left_y1, left_x2 = margin, top_y, 760
    rx1, ry1, rx2 = 800, top_y, W - margin

    preview = None
    frame_w = frame_h = 0
    if report_img is not None:
        preview = report_img.copy()
        preview.thumbnail((600, 500), Image.LANCZOS)
        frame_pad = 10
        frame_w = preview.width + frame_pad * 2
        frame_h = preview.height + frame_pad * 2
        left_y2 = left_y1 + 100 + frame_h + 36
    else:
        left_y2 = left_y1 + 540

    ry2 = ry1 + 540
    by = max(left_y2, ry2) + 40
    bottom_y2 = by + 620
    H = bottom_y2 + 90

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    _draw_card(draw, (margin, header_y1, W - margin, header_y2), fill=card_fill, outline=border, radius=28)
    _paste_logo_or_fallback(img, draw, 76, 68, size=54)
    draw.text((145, 62), "ShadeMate", font=font_title, fill=text_dark)
    draw.text((145, 112), "Foundation Shade Detector • Analysis Report", font=font_sub, fill=text_muted)
    timestamp = datetime.now().strftime("%d %B %Y • %H:%M")
    tw = draw.textbbox((0, 0), timestamp, font=font_small)[2]
    draw.text((W - margin - 40 - tw, 70), timestamp, font=font_small, fill=text_muted)

    _draw_card(draw, (left_x1, left_y1, left_x2, left_y2), fill=card_fill, outline=border, radius=26)
    draw.text((left_x1 + 28, left_y1 + 24), "Frame + Landmark", font=font_h2, fill=text_dark)

    if preview is not None:
        fx1 = left_x1 + ((left_x2 - left_x1) - frame_w) // 2
        fy1 = left_y1 + 86
        fx2, fy2 = fx1 + frame_w, fy1 + frame_h
        draw.rounded_rectangle((fx1, fy1, fx2, fy2), radius=16, fill=frame_fill, outline=soft_outline, width=2)
        img.paste(preview, (fx1 + 10, fy1 + 10))
    else:
        box_x1, box_y1, box_x2, box_y2 = left_x1 + 40, left_y1 + 100, left_x2 - 40, left_y2 - 40
        draw.rounded_rectangle((box_x1, box_y1, box_x2, box_y2), radius=18, fill=frame_fill, outline=soft_outline, width=2)
        _draw_text_block(draw, box_x1 + 28, box_y1 + 50, "No preview image stored in result.", font_text, text_muted, box_x2 - box_x1 - 56)

    _draw_card(draw, (rx1, ry1, rx2, ry2), fill=card_fill, outline=border, radius=26)
    draw.text((rx1 + 28, ry1 + 24), "Skin Tone Summary", font=font_h2, fill=text_dark)
    draw.text((rx1 + 28, ry1 + 80), "Detected Skin Color", font=font_label, fill=text_dark)

    # more compact swatch and equal-length chips
    sw_x1, sw_y1, sw_x2, sw_y2 = rx1 + 28, ry1 + 112, rx1 + 350, ry1 + 166
    draw.rounded_rectangle((sw_x1, sw_y1, sw_x2, sw_y2), radius=17, fill=skin_hex, outline=None)
    hex_bbox = draw.textbbox((0, 0), skin_hex, font=font_hex)
    hex_w = hex_bbox[2] - hex_bbox[0]
    hex_h = hex_bbox[3] - hex_bbox[1]
    light_hexes = ["#f6ede4", "#f3e7db", "#f7ead0", "#f8ead8", "#ffffff"]
    draw.text((sw_x1 + ((sw_x2 - sw_x1) - hex_w) // 2, sw_y1 + ((sw_y2 - sw_y1) - hex_h) // 2 - 1), skin_hex, font=font_hex, fill="#FFFFFF" if str(skin_hex).lower() not in light_hexes else text_dark)

    chip_y = ry1 + 119
    chip_w = 150
    chip_h = 38
    chip_gap = 14
    chip1_x1 = sw_x2 + 16
    chip2_x1 = chip1_x1 + chip_w + chip_gap
    draw.rounded_rectangle((chip1_x1, chip_y, chip1_x1 + chip_w, chip_y + chip_h), radius=19, fill=pink_soft, outline=soft_outline)
    draw.rounded_rectangle((chip2_x1, chip_y, chip2_x1 + chip_w, chip_y + chip_h), radius=19, fill=green_soft, outline=soft_outline)
    draw.text((chip1_x1 + 18, chip_y + 8), f"{user_undertone}", font=font_small, fill=pink)
    draw.text((chip2_x1 + 18, chip_y + 8), f"{user_skintone}", font=font_small, fill=green)

    # compact MST box with confidence on same line as MST value
    mst_box_x1, mst_box_y1, mst_box_x2, mst_box_y2 = rx1 + 28, ry1 + 194, rx2 - 28, ry1 + 276
    draw.rounded_rectangle((mst_box_x1, mst_box_y1, mst_box_x2, mst_box_y2), radius=20, fill=row_fill, outline=soft_outline, width=2)
    draw.text((mst_box_x1 + 22, mst_box_y1 + 11), "Predicted MST", font=font_label, fill=text_dark)
    line_y = mst_box_y1 + 39
    draw.text((mst_box_x1 + 22, line_y), f"MST-{mst}", font=font_mst, fill=text_dark)
    conf_x = mst_box_x1 + 182
    draw.text((conf_x, line_y + 1), "Confidence", font=font_small, fill=text_muted)
    conf_label_w = draw.textbbox((0, 0), "Confidence", font=font_small)[2]
    draw.text((conf_x + conf_label_w + 10, line_y), f"{confidence:.1f}%", font=font_conf, fill=pink)
    _draw_progress_bar(draw, mst_box_x1 + 22, mst_box_y1 + 61, (mst_box_x2 - mst_box_x1) - 44, 10, confidence, fill=pink, bg=soft_outline)

    # compact CIELAB title and boxes
    cielab_y = ry1 + 304
    draw.text((rx1 + 28, cielab_y), "CIELAB Values", font=font_lab_title, fill=text_dark)
    lab_cards = [("L*", str(L_val), lab_blue_fill, blue), ("a*", str(a_val), lab_pink_fill, pink), ("b*", str(b_val), lab_orange_fill, orange)]
    lab_y = ry1 + 342
    lab_box_w, lab_box_h, gap = 104, 44, 12
    start_x = rx1 + 28
    for i, (label, value, fill_col, accent) in enumerate(lab_cards):
        bx1 = start_x + i * (lab_box_w + gap)
        by1 = lab_y
        bx2, by2 = bx1 + lab_box_w, by1 + lab_box_h
        draw.rounded_rectangle((bx1, by1, bx2, by2), radius=12, fill=fill_col, outline=accent, width=2)
        draw.text((bx1 + 12, by1 + 12), label, font=font_lab_label, fill=accent)
        draw.text((bx1 + 40, by1 + 12), str(value), font=font_lab_value, fill=text_dark)

    draw.text((rx1 + 28, ry1 + 406), "Top Alternative MST", font=font_label, fill=text_dark)
    alt_y = ry1 + 440
    alt_color_list = ["#D9C19D", "#B8885E", "#8F6548", "#6C4E3B"]
    for i, alt in enumerate(alt_mst[:3]):
        label = alt.get("label", "-")
        score = float(alt.get("score", 0) or 0)
        cy = alt_y + i * 28
        swatch = alt.get("hex") or alt_color_list[i % len(alt_color_list)]
        draw.rounded_rectangle((rx1 + 28, cy, rx1 + 56, cy + 20), radius=8, fill=swatch, outline=soft_outline, width=1)
        draw.text((rx1 + 68, cy), f"{label} • {score:.1f}%", font=font_small, fill=text_dark)

    def _load_product_thumb(brand_name, shade_name, target_w, target_h):
        img_path = product_image_path(brand_name, shade_name)
        if not img_path:
            return None
        try:
            pimg = Image.open(img_path).convert("RGBA")
            scale = min(target_w / max(1, pimg.width), target_h / max(1, pimg.height))
            new_w = max(1, int(pimg.width * scale))
            new_h = max(1, int(pimg.height * scale))
            pimg = pimg.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            off_x = (target_w - new_w) // 2
            off_y = (target_h - new_h) // 2
            canvas.paste(pimg, (off_x, off_y), pimg)
            return canvas
        except Exception:
            return None

    left_bottom = (margin, by, 760, bottom_y2)
    right_bottom = (800, by, W - margin, bottom_y2)
    _draw_card(draw, left_bottom, fill=card_fill, outline=border, radius=26)
    _draw_card(draw, right_bottom, fill=card_fill, outline=border, radius=26)

    # Main recommendation with larger text and product image on the right
    lx1, ly1, lx2, ly2 = left_bottom
    draw.text((lx1 + 28, ly1 + 24), "Main Recommendation", font=font_h2, fill=text_dark)
    draw.line((lx1 + 28, ly1 + 72, lx2 - 28, ly1 + 72), fill=line, width=2)

    image_box_w, image_box_h = 190, 240
    image_box_x1 = lx2 - 28 - image_box_w
    image_box_y1 = ly1 + 120
    image_box_x2 = image_box_x1 + image_box_w
    image_box_y2 = image_box_y1 + image_box_h
    draw.rounded_rectangle((image_box_x1, image_box_y1, image_box_x2, image_box_y2), radius=18, fill=row_fill, outline=soft_outline, width=1)

    main_thumb = _load_product_thumb(brand, shade, image_box_w - 22, image_box_h - 22)
    if main_thumb is not None:
        img.paste(main_thumb, (image_box_x1 + 11, image_box_y1 + 11), main_thumb)
    else:
        draw.text((image_box_x1 + 34, image_box_y1 + image_box_h // 2 - 10), "No image", font=font_small, fill=text_muted)

    info_x, info_y = lx1 + 28, ly1 + 104
    value_x = info_x + 150
    value_w = image_box_x1 - value_x - 20
    main_items = [("Brand", brand), ("Product", product), ("Shade", shade), ("Undertone", rec_undertone), ("Price", price)]
    for label, value in main_items:
        draw.text((info_x, info_y), str(label), font=font_rec_label, fill=text_muted)
        info_y = _draw_text_block(draw, value_x, info_y, value, font_rec_value, text_dark, value_w, line_gap=8, max_lines=2)
        info_y += 24

    # Top 5 recommendations with index, product image, text, and swatch
    rx1b, ry1b, rx2b, ry2b = right_bottom
    draw.text((rx1b + 28, ry1b + 24), "Top 5 Recommendations", font=font_h2, fill=text_dark)
    draw.line((rx1b + 28, ry1b + 72, rx2b - 28, ry1b + 72), fill=line, width=2)
    row_x1, row_x2 = rx1b + 28, rx2b - 28
    row_y, row_h, row_gap = ry1b + 92, 88, 10
    for idx, rec in enumerate(top5[:5], start=1):
        brand_i = str(rec.get("Brand", "-"))
        product_i = str(rec.get("Product", "-"))
        shade_i = str(rec.get("Shade", rec.get("shade_name", "-")))
        undertone_i = str(rec.get("Undertone", "-"))
        price_i = _format_price(rec.get("Price", "-"))
        swatch_hex = _safe_hex_from_rec(rec)
        y1 = row_y + (idx - 1) * (row_h + row_gap)
        y2 = y1 + row_h
        draw.rounded_rectangle((row_x1, y1, row_x2, y2), radius=16, fill=row_fill, outline=soft_outline, width=1)

        badge_size = 32
        badge_x1, badge_y1 = row_x1 + 12, y1 + (row_h - badge_size) // 2
        draw.rounded_rectangle((badge_x1, badge_y1, badge_x1 + badge_size, badge_y1 + badge_size), radius=10, fill=badge_fill, outline=soft_outline, width=1)
        num = str(idx)
        nb = draw.textbbox((0, 0), num, font=font_small)
        draw.text((badge_x1 + (badge_size - (nb[2] - nb[0])) // 2, badge_y1 + (badge_size - (nb[3] - nb[1])) // 2 - 1), num, font=font_small, fill="#D94E91")

        thumb_size = 52
        thumb_x1 = badge_x1 + badge_size + 12
        thumb_y1 = y1 + (row_h - thumb_size) // 2
        draw.rounded_rectangle((thumb_x1, thumb_y1, thumb_x1 + thumb_size, thumb_y1 + thumb_size), radius=12, fill=frame_fill, outline=soft_outline, width=1)
        thumb = _load_product_thumb(brand_i, shade_i, thumb_size - 8, thumb_size - 8)
        if thumb is not None:
            img.paste(thumb, (thumb_x1 + 4, thumb_y1 + 4), thumb)

        sw_size = 38
        sw_x1, sw_y1 = row_x2 - 52, y1 + (row_h - sw_size) // 2
        draw.rounded_rectangle((sw_x1, sw_y1, sw_x1 + sw_size, sw_y1 + sw_size), radius=12, fill=swatch_hex, outline=soft_outline, width=1)

        text_x = thumb_x1 + thumb_size + 14
        text_max_w = sw_x1 - 16 - text_x
        title = _fit_text(draw, f"{brand_i} • {shade_i}", font_top5_title, text_max_w)
        product_line = _fit_text(draw, product_i, font_top5_meta, text_max_w)
        meta = _fit_text(draw, f"{undertone_i} • {price_i}", font_top5_meta, text_max_w)
        draw.text((text_x, y1 + 10), title, font=font_top5_title, fill=text_dark)
        draw.text((text_x, y1 + 35), product_line, font=font_top5_meta, fill=product_muted)
        draw.text((text_x, y1 + 57), meta, font=font_top5_meta, fill=text_muted)

    draw.text((margin + 4, H - 42), "Generated by ShadeMate", font=font_small, fill=footer_muted)
    draw.text((W - 320, H - 42), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=font_small, fill=footer_muted)
    return img


# shade matching
from pathlib import Path
import base64
import html
import re

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
PRODUCT_DIR = ASSETS_DIR / "products"

MST_COLORS = {
    1:"#f6ede4", 2:"#f3e7db", 3:"#f7ead0", 4:"#eadaba", 5:"#d7bd96",
    6:"#a07850", 7:"#825c43", 8:"#604134", 9:"#3a312a", 10:"#292420"
}

BRANDS = [
    "Wardah", "Luxcrime", "Omg", "Mop", "Jacquelle", "Dazzle Me",
    "Make Over", "Maybelline", "Fenty Beauty", "L'Oreal Paris"
]


def load_css():
    css_path = APP_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)



def inject_ui_hotfix_css():
    st.markdown("""
    <style>
    header, [data-testid="stHeader"] { visibility: visible !important; display:block !important; background:transparent !important; height:3.2rem !important; }
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], button[kind="header"], button[data-testid="baseButton-header"] {
        visibility: visible !important; display:flex !important; opacity:1 !important; color:#758952 !important; background:rgba(255,240,245,.95) !important; border:1px solid rgba(255,168,214,.55) !important; border-radius:999px !important; box-shadow:0 8px 18px rgba(200,107,133,.16) !important;
    }
    [data-testid="collapsedControl"] svg, [data-testid="stSidebarCollapsedControl"] svg, button[kind="header"] svg, button[data-testid="baseButton-header"] svg { color:#758952 !important; fill:#758952 !important; stroke:#758952 !important; }
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { color:#2F2330 !important; }
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: inherit; }
    .stButton > button:not([kind="primary"]) { background:rgba(255,255,255,.86) !important; color:#758952 !important; border:1px solid rgba(117,137,82,.65) !important; box-shadow:none !important; }
    .stButton > button:not([kind="primary"]) * { color:#758952 !important; }
    .stButton > button[kind="primary"] { background:linear-gradient(135deg,#F48ABD,#E7569F) !important; color:white !important; border:0 !important; }
    .stButton > button[kind="primary"] * { color:white !important; }

    .upload-real-card [data-testid="stFileUploader"] { background:rgba(255,255,255,.80) !important; border:2px dashed rgba(255,168,214,.72) !important; border-radius:1.35rem !important; min-height:355px !important; display:flex !important; align-items:center !important; justify-content:center !important; padding:1.4rem !important; box-shadow:0 16px 35px rgba(244,138,189,.10); }
    .upload-real-card [data-testid="stFileUploaderDropzone"] { background:transparent !important; border:0 !important; min-height:310px !important; width:100% !important; display:flex !important; align-items:center !important; justify-content:center !important; flex-direction:column !important; text-align:center !important; color:#2F2330 !important; }
    .upload-real-card [data-testid="stFileUploaderDropzone"]::before { content:"⇧"; width:86px; height:86px; border-radius:1.45rem; background:#F9D1D9; color:#F48ABD; display:flex; align-items:center; justify-content:center; font-size:3rem; font-weight:900; margin-bottom:1rem; }
    .upload-real-card [data-testid="stFileUploaderDropzone"] button { background:#FFF0F5 !important; border:1px solid rgba(255,168,214,.55) !important; border-radius:999px !important; color:#D94E91 !important; font-weight:900 !important; }

    .camera-real-card [data-testid="stCameraInput"] { background:rgba(255,255,255,.85) !important; border:1.5px solid rgba(255,168,214,.78) !important; border-radius:1.35rem !important; min-height:330px !important; padding:1.1rem !important; box-shadow:0 16px 35px rgba(244,138,189,.10); color:#2F2330 !important; overflow:hidden !important; }
    .camera-real-card [data-testid="stCameraInput"] * { color:#2F2330 !important; }
    .camera-real-card [data-testid="stCameraInput"] video, .camera-real-card [data-testid="stCameraInput"] img { transform:none !important; max-height:300px !important; object-fit:contain !important; border-radius:1rem !important; }
    .camera-real-card [data-testid="stCameraInput"] button { background:linear-gradient(135deg,#BADF93,#838F58) !important; color:white !important; border:0 !important; border-radius:.85rem !important; font-weight:900 !important; }


    .analysis-preview-img img { border-radius:1rem !important; border:1px solid rgba(255,168,214,.35); box-shadow:0 12px 24px rgba(0,0,0,.08); }
    .html-product-card{ background:rgba(255,255,255,.80); border:1px solid rgba(255,168,214,.45); border-radius:1.15rem; padding:1rem; min-height:235px; box-shadow:0 16px 30px rgba(200,107,133,.08); margin-bottom:1rem; }
    .html-product-top{ display:grid; grid-template-columns:92px 1fr 120px; gap:1rem; align-items:start; }
    .html-product-img{ width:82px; height:104px; object-fit:contain; border-radius:.8rem; background:#FFF0F5; border:1px solid rgba(232,192,197,.45); }
    .html-brand{ font-size:.78rem; color:#7B6472; letter-spacing:.08em; text-transform:uppercase; font-weight:900; margin-bottom:.2rem; }
    .html-name{ font-weight:900; color:#2F2330; font-size:1rem; margin-bottom:.45rem; }
    .html-price{ text-align:right; font-weight:900; color:#2F2330; font-size:1.02rem; }
    .html-reason{ background:rgba(255,240,245,.75); border-radius:.75rem; padding:.55rem .7rem; color:#7B6472; font-size:.82rem; margin:.65rem 0; }
    .html-bar{ height:9px; border-radius:999px; background:rgba(232,192,197,.45); overflow:hidden; margin-top:.35rem; }
    .html-bar > div{ height:100%; border-radius:999px; background:linear-gradient(90deg,#BADF93,#758952); }

    /* Camera card dibuat lebih ringkas + foto kamera tidak mirror */
    .camera-intro-card{ padding:.72rem .9rem !important; margin-top:.75rem !important; margin-bottom:.65rem !important; text-align:center; }
    .camera-intro-card .upload-symbol{ width:42px !important; height:42px !important; font-size:1.1rem !important; margin:0 auto .35rem !important; border-radius:1rem !important; }
    .camera-intro-card h3{ font-size:1.05rem !important; margin:0 !important; }
    .camera-intro-card .small-text{ font-size:.78rem !important; }
    .camera-real-card [data-testid="stCameraInput"]{ min-height:235px !important; padding:.65rem !important; }
    .camera-real-card [data-testid="stCameraInput"] > div,
    [data-testid="stCameraInput"] > div{ width:100% !important; }
    /* Live preview dan hasil foto di widget kamera dibuat tidak mirror + ukurannya normal */
    .camera-real-card [data-testid="stCameraInput"] video,
    .camera-real-card [data-testid="stCameraInput"] img,
    .camera-real-card [data-testid="stCameraInput"] canvas,
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img,
    [data-testid="stCameraInput"] canvas,
    [data-testid="stCameraInput"] video[playsinline]{
        width:100% !important;
        max-width:100% !important;
        height:auto !important;
        max-height:none !important;
        object-fit:contain !important;
        display:block !important;
        margin:0 auto !important;
        transform:scaleX(-1) !important;
        -webkit-transform:scaleX(-1) !important;
        border-radius:1rem !important;
    }

    /* Radio filter dan input mode dibuat seperti pill/elips kecil */
    [data-testid="stRadio"] div[role="radiogroup"]{ gap:.65rem 1.1rem !important; align-items:center !important; flex-wrap:wrap !important; }
    [data-testid="stRadio"] label{ border-radius:999px !important; padding:.56rem 1.05rem !important; border:1px solid transparent !important; background:transparent !important; min-height:unset !important; color:#758952 !important; }
    [data-testid="stRadio"] label:has(input:checked){ background:#F9C3DE !important; color:#2F2330 !important; border-color:#F9C3DE !important; box-shadow:0 10px 20px rgba(244,138,189,.14) !important; }
    [data-testid="stRadio"] label > div:first-child{ display:none !important; }
    [data-testid="stRadio"] p{ font-weight:800 !important; color:#5E4A59 !important; }

    /* Sidebar menu rata kiri */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]{ align-items:stretch !important; gap:.55rem !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label{ width:100% !important; justify-content:flex-start !important; text-align:left !important; padding:.82rem 1rem !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] p{ width:100% !important; text-align:left !important; }

    /* Panel filter recommendation seperti mockup */
    .filters-header-box{ border:1.6px solid rgba(255,168,214,.58); border-radius:1.2rem; padding:1.05rem 1.35rem; background:rgba(255,255,255,.48); margin-bottom:.95rem; }
    .filters-shell{ padding:0 .2rem .2rem .2rem; }
    .filter-group{ margin-bottom:1.05rem; }
    .filter-group-title{ font-weight:700; font-size:1rem; margin-bottom:.45rem; color:#5E4A59; }

    /* Processing Pipeline horizontal seperti mockup */
    .pipeline-card{ padding:1.4rem 1.6rem 1.8rem !important; overflow:hidden !important; margin-bottom:1.55rem !important; }
    .method-timeline{ display:grid !important; grid-template-columns:repeat(6,minmax(110px,1fr)) !important; gap:2.4rem !important; align-items:start !important; text-align:center !important; margin-top:.85rem !important; }
    .method-step{ position:relative !important; min-height:145px !important; }
    .method-step:not(:last-child)::after{ content:'⟶'; position:absolute; right:-2.05rem; top:48px; color:#F48ABD; font-weight:900; opacity:.92; font-size:2rem; line-height:1; }
    .method-step .step-badge{ width:22px !important; height:22px !important; border-radius:999px !important; display:flex !important; align-items:center !important; justify-content:center !important; color:white !important; font-size:.78rem !important; font-weight:900 !important; margin:0 auto .45rem !important; }
    .method-step .method-icon{ width:58px !important; height:58px !important; border-radius:1rem !important; display:flex !important; align-items:center !important; justify-content:center !important; font-size:1.75rem !important; margin:.25rem auto .55rem !important; border:1px solid currentColor !important; }
    .method-step .method-title{ font-weight:900 !important; margin-top:.35rem !important; line-height:1.25 !important; }
    .method-cards-row{ margin-top:.3rem !important; }

    /* Technology stack diperkecil */
    .tech-stack-box{ padding:1rem 1.15rem !important; margin-top:1rem !important; }
    .tech-stack-box h3{ font-size:1.7rem !important; margin-bottom:.5rem !important; }
    .tech-card.compact{ padding:.75rem .85rem !important; border-radius:1rem !important; min-height:unset !important; }
    .tech-card.compact strong{ font-size:.92rem !important; }
    .tech-card.compact .small-text{ font-size:.72rem !important; line-height:1.4 !important; }
    .tech-icon.compact{ width:36px !important; height:36px !important; font-size:1rem !important; }

    @media(max-width:900px){ .method-timeline{ grid-template-columns:repeat(2,1fr) !important; } .method-step::after{ display:none !important; } }



    /* =========================================================
       RESPONSIVE + DARK/LIGHT ADAPTIVE PATCH
       - Membuat UI lebih aman di HP
       - Menghindari icon/teks hitam hilang saat browser dark mode
       - Menjaga card tidak overflow di layar kecil
       ========================================================= */
    :root{
        color-scheme: light dark;
        --sm-bg:#FFF6FA;
        --sm-bg-soft:#FFF0F5;
        --sm-card:rgba(255,255,255,.86);
        --sm-card-solid:#FFFFFF;
        --sm-text:#2F2330;
        --sm-muted:#7B6472;
        --sm-border:rgba(255,168,214,.55);
        --sm-pink:#F48ABD;
        --sm-pink-strong:#E7569F;
        --sm-green:#758952;
        --sm-green-soft:#EEF3E5;
        --sm-shadow:0 18px 38px rgba(200,107,133,.12);
    }

    @media (prefers-color-scheme: dark){
        :root{
            --sm-bg:#1E171D;
            --sm-bg-soft:#271D25;
            --sm-card:rgba(45,34,43,.92);
            --sm-card-solid:#2D222B;
            --sm-text:#F8EEF4;
            --sm-muted:#D8C5D0;
            --sm-border:rgba(244,138,189,.45);
            --sm-pink:#FF9ED0;
            --sm-pink-strong:#F06BAF;
            --sm-green:#C8DDA5;
            --sm-green-soft:#33402B;
            --sm-shadow:0 18px 38px rgba(0,0,0,.32);
        }
    }

    .stApp,
    [data-testid="stAppViewContainer"]{
        background:linear-gradient(135deg,var(--sm-bg) 0%, var(--sm-bg-soft) 55%, rgba(117,137,82,.12) 100%) !important;
        color:var(--sm-text) !important;
    }

    [data-testid="stSidebar"]{
        background:linear-gradient(180deg,var(--sm-bg-soft),var(--sm-bg)) !important;
        border-right:1px solid var(--sm-border) !important;
    }

    .custom-card,
    .tip-card,
    .html-product-card,
    .pipeline-card,
    .upload-real-card [data-testid="stFileUploader"],
    .camera-real-card [data-testid="stCameraInput"],
    .filters-header-box{
        background:var(--sm-card) !important;
        color:var(--sm-text) !important;
        border-color:var(--sm-border) !important;
        box-shadow:var(--sm-shadow) !important;
    }

    .page-title,
    .hero-title,
    .brand-name,
    .sidebar-brand,
    .html-name,
    .html-price,
    .method-title,
    .step-title,
    .feature-card h3,
    .tip-card h3,
    .custom-card h1,
    .custom-card h2,
    .custom-card h3,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3{
        color:var(--sm-text) !important;
    }

    .subtitle,
    .small-text,
    .html-brand,
    .html-reason,
    .sidebar-footer,
    .stat-label{
        color:var(--sm-muted) !important;
    }

    .logo-box,
    .feature-icon,
    .upload-symbol,
    .method-icon,
    .tech-icon,
    .tip-emoji,
    .step-icon{
        color:var(--sm-text) !important;
        background:rgba(255,168,214,.24) !important;
    }

    .chip,
    .pill,
    .match-badge{
        color:var(--sm-text) !important;
        border-color:var(--sm-border) !important;
    }

    .swatch{
        border:1px solid rgba(255,255,255,.45) !important;
        box-shadow:0 3px 10px rgba(0,0,0,.12) !important;
    }

    /* Streamlit widget text agar tidak hilang di dark mode */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stFileUploader"] *,
    [data-testid="stCameraInput"] *,
    .stButton button,
    .stDownloadButton button{
        color:var(--sm-text) !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton button{
        background:linear-gradient(135deg,var(--sm-pink),var(--sm-pink-strong)) !important;
        color:white !important;
        border:0 !important;
    }

    .stButton > button[kind="primary"] *,
    .stDownloadButton button *{
        color:white !important;
    }

    /* Dark mode khusus product card */
    @media (prefers-color-scheme: dark){
        .html-product-img{
            background:rgba(255,240,245,.12) !important;
            border-color:rgba(244,138,189,.35) !important;
        }
        .html-bar{
            background:rgba(255,255,255,.18) !important;
        }
        .html-bar > div{
            background:linear-gradient(90deg,#C8DDA5,#91A56D) !important;
        }
    }

    /* ======================= MOBILE LAYOUT ======================= */
    @media(max-width: 900px){
        .main .block-container,
        [data-testid="stMainBlockContainer"]{
            padding-left:1rem !important;
            padding-right:1rem !important;
            max-width:100% !important;
        }

        .hero{
            padding:2rem 1rem !important;
            text-align:center !important;
        }
        .hero-title{
            font-size:2.2rem !important;
            line-height:1.12 !important;
        }
        .page-title{
            font-size:2rem !important;
            line-height:1.15 !important;
        }
        .subtitle{
            font-size:.9rem !important;
        }

        .custom-card,
        .tip-card,
        .html-product-card,
        .filters-header-box{
            border-radius:1rem !important;
            padding:1rem !important;
        }

        .html-product-top{
            display:grid !important;
            grid-template-columns:72px 1fr !important;
            gap:.75rem !important;
        }
        .html-product-img{
            width:68px !important;
            height:86px !important;
        }
        .html-price{
            grid-column:1 / -1 !important;
            text-align:left !important;
            display:flex !important;
            justify-content:space-between !important;
            align-items:center !important;
            gap:.5rem !important;
            margin-top:.35rem !important;
            font-size:.9rem !important;
        }
        .html-name{
            font-size:.92rem !important;
            line-height:1.35 !important;
        }
        .match-badge{
            font-size:.75rem !important;
            padding:.35rem .55rem !important;
        }

        .filters-shell [data-testid="stRadio"] div[role="radiogroup"]{
            gap:.45rem !important;
        }
        .filters-shell [data-testid="stRadio"] label{
            padding:.45rem .75rem !important;
            font-size:.82rem !important;
        }

        .method-timeline{
            grid-template-columns:repeat(2,1fr) !important;
            gap:1rem !important;
        }
        .method-step::after{
            display:none !important;
        }
        .method-step .method-icon{
            width:46px !important;
            height:46px !important;
            font-size:1.25rem !important;
        }

        .camera-intro-card{
            padding:.7rem !important;
        }
        .camera-real-card [data-testid="stCameraInput"]{
            min-height:auto !important;
            padding:.55rem !important;
        }
        .camera-real-card [data-testid="stCameraInput"] video,
        .camera-real-card [data-testid="stCameraInput"] img,
        .camera-real-card [data-testid="stCameraInput"] canvas{
            width:100% !important;
            max-height:none !important;
        }

        .tip-item{
            gap:.6rem !important;
        }
        .tip-emoji{
            min-width:28px !important;
            width:28px !important;
            height:28px !important;
            font-size:.9rem !important;
        }
    }

    @media(max-width: 520px){
        .hero-title{ font-size:1.75rem !important; }
        .page-title{ font-size:1.7rem !important; }
        .custom-card{ padding:.85rem !important; }
        .stats-grid{ grid-template-columns:repeat(2,1fr) !important; gap:.7rem !important; }
        .stat-number{ font-size:1.4rem !important; }
        .html-product-top{ grid-template-columns:60px 1fr !important; }
        .html-product-img{ width:58px !important; height:76px !important; }
        .swatch{ width:28px !important; height:24px !important; }
    }



    /* ===== Results page readability + mobile order patch ===== */
    .result-preview-card{ padding:1.05rem 1.1rem 1.15rem !important; margin-bottom:.85rem !important; }
    .result-preview-card .analysis-preview-img{ margin:.55rem 0 .95rem !important; }
    .result-preview-card .analysis-preview-img img{ display:block !important; margin:0 auto !important; max-height:360px !important; object-fit:contain !important; }
    .detected-skin-swatch{ width:100% !important; min-width:100% !important; height:86px !important; border-radius:1rem !important; margin:.55rem 0 .72rem !important; box-shadow:0 12px 24px rgba(0,0,0,.12) !important; border:1px solid rgba(255,255,255,.38) !important; }
    .result-section-title{ margin:1.15rem 0 .75rem !important; font-size:1.35rem !important; font-weight:900 !important; color:var(--sm-text) !important; }
    .result-grid{ display:grid !important; grid-template-columns:repeat(4,minmax(0,1fr)) !important; gap:1rem !important; margin-bottom:1rem !important; }
    .metric-card{ background:var(--sm-card) !important; color:var(--sm-text) !important; border:1px solid var(--sm-border) !important; border-radius:1.25rem !important; padding:1.15rem !important; min-height:145px !important; box-shadow:var(--sm-shadow) !important; }
    .metric-card .metric-value{ color:var(--sm-text) !important; font-size:1.85rem !important; font-weight:950 !important; margin:.65rem 0 .25rem !important; line-height:1.1 !important; }
    .metric-card .small-text, .metric-card .metric-sub, .result-card-muted{ color:var(--sm-muted) !important; }
    .result-card, .result-about-card{ background:var(--sm-card) !important; color:var(--sm-text) !important; border:1px solid var(--sm-border) !important; border-radius:1.25rem !important; box-shadow:var(--sm-shadow) !important; }
    .result-card strong, .result-about-card strong{ color:var(--sm-text) !important; }
    .result-card .pink-tint, .result-about-card .pink-tint{ background:rgba(244,138,189,.14) !important; color:var(--sm-text) !important; border:1px solid rgba(244,138,189,.22) !important; }
    .result-card .green-tint, .result-about-card .green-tint{ background:rgba(117,137,82,.16) !important; color:var(--sm-text) !important; border:1px solid rgba(117,137,82,.22) !important; }
    .color-grid{ display:grid !important; grid-template-columns:repeat(3,minmax(0,1fr)) !important; gap:.9rem !important; margin-top:1rem !important; }
    .color-box{ border-radius:.95rem !important; padding:1rem !important; min-height:120px !important; overflow:hidden !important; }
    .color-hex-value{ display:block !important; float:none !important; width:100% !important; margin-top:.45rem !important; color:var(--sm-text) !important; font-size:1.05rem !important; overflow-wrap:anywhere !important; word-break:normal !important; white-space:nowrap !important; }
    .result-button-spacer{ height:1rem !important; }
    .mst-strip{ overflow-x:auto !important; padding:.45rem .15rem .75rem !important; }
    .mst-strip::-webkit-scrollbar{ height:6px; }
    @media(max-width:900px){ .result-grid{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; gap:.85rem !important; } .color-grid{ grid-template-columns:repeat(3,minmax(0,1fr)) !important; gap:.65rem !important; } .color-box{ padding:.85rem .65rem !important; min-height:112px !important; } .color-hex-value{ font-size:.95rem !important; } .result-preview-card .analysis-preview-img img{ max-height:280px !important; } }
    @media(max-width:520px){ .result-grid{ grid-template-columns:1fr !important; } .metric-card{ min-height:118px !important; padding:1rem !important; } .metric-card .metric-value{ font-size:1.65rem !important; } .color-grid{ grid-template-columns:1fr !important; } .color-box{ min-height:auto !important; } .detected-skin-swatch{ width:100% !important; min-width:100% !important; height:74px !important; } .result-preview-card{ padding:.85rem !important; } }


    /* ===== Dark-mode fixes: privacy notice, tech stack, references ===== */
    .notice{
        background:var(--sm-card) !important;
        color:var(--sm-text) !important;
        border:1px solid var(--sm-border) !important;
        box-shadow:var(--sm-shadow) !important;
    }
    .notice strong{ color:var(--sm-text) !important; }
    .tech-stack-box{
        background:var(--sm-card) !important;
        color:var(--sm-text) !important;
        border:1px solid var(--sm-border) !important;
        box-shadow:var(--sm-shadow) !important;
    }
    .tech-card.compact{
        background:rgba(255,255,255,.72) !important;
        color:var(--sm-text) !important;
        border:1px solid rgba(244,138,189,.28) !important;
        box-shadow:none !important;
    }
    .tech-card.compact strong{ color:var(--sm-text) !important; }
    .tech-card.compact .small-text{ color:var(--sm-muted) !important; }
    .ref-box{
        background:var(--sm-card) !important;
        color:var(--sm-text) !important;
        border:1px solid var(--sm-border) !important;
        border-radius:1.25rem !important;
        box-shadow:var(--sm-shadow) !important;
    }
    .ref-box strong{ color:var(--sm-text) !important; }
    .ref-box .small-text{ color:var(--sm-muted) !important; }
    @media (prefers-color-scheme: dark){
        .tech-card.compact{
            background:rgba(255,255,255,.08) !important;
            border-color:rgba(244,138,189,.34) !important;
        }
        .tech-icon.compact{
            background:rgba(244,138,189,.20) !important;
        }
        .ref-box,
        .notice{
            background:rgba(45,34,43,.94) !important;
        }
    }

    #MainMenu, footer, [data-testid="stDecoration"] { visibility:hidden !important; display:none !important; }
    </style>
    """, unsafe_allow_html=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def product_image_path(brand, shade=None):
    """Cari gambar produk secara robust.
    - folder brand dicari case-insensitive, jadi brand "Mop" tetap bisa masuk ke folder "MOP"
    - nama shade dicari case-insensitive
    - support jpg/png/jpeg/webp/avif
    """
    try:
        brand_str = str(brand).strip()
        shade_str = str(shade).strip() if shade is not None else ""

        # 1) cari folder brand exact dulu, lalu case-insensitive
        brand_dir = PRODUCT_DIR / brand_str
        if not brand_dir.exists():
            brand_dir = None
            if PRODUCT_DIR.exists():
                for child in PRODUCT_DIR.iterdir():
                    if child.is_dir() and child.name.lower() == brand_str.lower():
                        brand_dir = child
                        break

        # 2) cari file berdasarkan shade di dalam folder brand
        if brand_dir and brand_dir.exists() and shade_str:
            allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

            # exact candidate
            for ext in allowed_ext:
                candidate = brand_dir / f"{shade_str}{ext}"
                if candidate.exists():
                    return str(candidate)

            # case-insensitive exact stem
            shade_key = shade_str.lower().strip()
            for file in brand_dir.iterdir():
                if file.is_file() and file.suffix.lower() in allowed_ext:
                    if file.stem.lower().strip() == shade_key:
                        return str(file)

            # fallback contains matching: berguna kalau shade di CSV "C10 Pearl" dan file ada variasi nama
            for file in brand_dir.iterdir():
                if file.is_file() and file.suffix.lower() in allowed_ext:
                    stem = file.stem.lower().strip()
                    if shade_key in stem or stem in shade_key:
                        return str(file)

        # 3) fallback per brand png di root products
        if PRODUCT_DIR.exists():
            for file in PRODUCT_DIR.iterdir():
                if file.is_file() and file.stem.lower() == slug(brand_str).lower():
                    return str(file)

    except Exception:
        pass

    for dummy in [PRODUCT_DIR / "dummy_product.png", ASSETS_DIR / "dummy_product.png"]:
        if dummy.exists():
            return str(dummy)
    return None

def encode_image_for_html(img_path):
    try:
        p = Path(img_path)
        if not p.exists():
            return ""
        ext = p.suffix.lower().replace('.', '')
        if ext == 'jpg':
            ext = 'jpeg'
        data = base64.b64encode(p.read_bytes()).decode('utf-8')
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return ""

def ehtml(value):
    return html.escape(str(value)) if value is not None else "-"

def safe_similarity(delta_e):
    try:
        return float(np.clip(100 - float(delta_e) * 2.8, 55, 99.2))
    except Exception:
        return 88.0


def ImageColor_get_rgb(hex_color):
    hex_color = str(hex_color).strip().replace("#", "")
    if len(hex_color) != 6:
        return (196, 149, 106)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def render_sidebar():
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="logo-box">✿</div>
        <div>
            <div class="brand-kicker">Capstone 27</div>
            <div class="brand-name">ShadeMate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    pages = ["Home", "Skin Analysis", "Results", "Recommendations", "About Method"]
    icons = {"Home":"🏠", "Skin Analysis":"📷", "Results":"📊", "Recommendations":"✨", "About Method":"📖"}
    current = st.session_state.get("page", "Home")
    page = st.sidebar.radio("Menu", pages, index=pages.index(current), format_func=lambda x: f"{icons[x]}  {x}", label_visibility="collapsed")
    st.sidebar.markdown("""
    <div class="sidebar-footer"><div>ShadeMate v1.0</div><div>Capstone 27</div></div>
    """, unsafe_allow_html=True)
    st.session_state["page"] = page
    return page


def get_resources_or_stop():
    if "resources_loaded" not in st.session_state:
        with st.spinner("Memuat model & database foundation..."):
            try:
                (face_mesh, ensemble, scaler, kmeans, df_found, centroids, mst_hex_lookup) = load_resources()
                st.session_state["resources"] = {
                    "face_mesh": face_mesh, "ensemble": ensemble, "scaler": scaler,
                    "kmeans": kmeans, "df_found": df_found, "centroids": centroids,
                    "mst_hex_lookup": mst_hex_lookup,
                }
                st.session_state["resources_loaded"] = True
            except Exception as e:
                st.error(f"❌ Gagal load model: {e}")
                st.stop()
    return st.session_state["resources"]


def home_page(df_found):
    st.markdown("""
    <div class="hero">
        <div class="pill">✧ Skin Tone Analysis ✧</div>
        <div class="hero-title">Find Your Perfect <span class="pink">Foundation</span> <span class="green">Match</span></div>
        <div class="subtitle">Analyze your skin tone and undertone to discover foundation shades that suit you.<br>Powered by computer vision and color science.</div>
    </div>
    """, unsafe_allow_html=True)
    _, center, _ = st.columns([1.2, 1.1, 1.2])
    with center:
        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("Start Analysis →", type="primary", use_container_width=True):
                st.session_state["page"] = "Skin Analysis"; st.rerun()
        with cta2:
            if st.button("Learn More", use_container_width=True):
                st.session_state["page"] = "About Method"; st.rerun()
    st.markdown("<div style='height:1.3rem;'></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    cards = [("📷","Upload Photo","Upload a selfie or use your webcam for real-time skin tone analysis.","pink-tint"),("🎨","Skin Tone Analysis","AI extracts dominant skin color using K-Means clustering and LAB color space.","green-tint"),("✨","Foundation Match","Get personalized recommendations from foundation shades across available brands.","purple-tint")]
    for col, (icon, title, text, tint) in zip(cols, cards):
        with col:
            st.markdown(f"""<div class="custom-card feature-card {tint}" style="text-align:center;"><div class="feature-icon" style="background:rgba(255,168,214,.28);margin:0 auto 1rem;display:flex;justify-content:center;align-items:center;">{icon}</div><h3>{title}</h3><p>{text}</p></div>""", unsafe_allow_html=True)
    total_shades = len(df_found)
    total_brands = df_found["Brand"].nunique() if "Brand" in df_found else 10
    st.markdown(f"""<div class="custom-card stats-card"><div class="stats-grid"><div><div class="stat-number">{total_shades}+</div><div class="stat-label">Foundation Shades</div></div><div><div class="stat-number">{total_brands}</div><div class="stat-label">Brands Covered</div></div><div><div class="stat-number">10</div><div class="stat-label">Monk Skin Tones</div></div><div><div class="stat-number">98%</div><div class="stat-label">Analysis Accuracy</div></div></div></div>""", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;margin-top:1.8rem;'>How It Works</h3>", unsafe_allow_html=True)
    steps = [("01","⇧","Upload","Take or upload a photo in natural lighting.","#F48ABD"),("02","⌗","Analyze","AI detects face and extracts skin pixels.","#BADF93"),("03","✧","Match","Euclidean distance finds closest shades.","#F48ABD"),("04","▯","Discover","Browse curated foundation recommendations.","#BADF93")]
    for col, (num, icon, title, desc, color) in zip(st.columns(4), steps):
        with col:
            icon_color = "#D94E91" if color == "#F48ABD" else "#758952"
            st.markdown(f"""<div class="step-node"><div class="step-badge" style="background:{color};">{num}</div><div class="step-icon" style="background:{color}55;color:{icon_color};">{icon}</div><div class="step-title">{title}</div><div class="small-text" style="text-align:center;max-width:170px;">{desc}</div></div>""", unsafe_allow_html=True)



def skin_analysis_page(resources):
    st.markdown('<div class="pill">Step 1 of 3</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Skin Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="margin:0;max-width:760px;">Upload your photo or use webcam to begin skin tone analysis.</div>', unsafe_allow_html=True)
    mode = st.radio("Input mode", ["Upload Photo", "Camera Capture"], horizontal=True, label_visibility="collapsed")
    left, right = st.columns([2.4, 1.1], gap="large")
    image_source = None
    with left:
        if mode == "Upload Photo":
            st.markdown('<div class="upload-real-card">', unsafe_allow_html=True)
            uploaded = st.file_uploader("Drag & drop your photo here", type=["png", "jpg", "jpeg", "webp"], help="PNG, JPG, JPEG, atau WEBP", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            if uploaded:
                image_source = uploaded
                st.image(uploaded, caption="Preview foto", use_container_width=True)
        else:
            st.markdown("""
            <div class="custom-card camera-intro-card">
                <div class="upload-symbol" style="background:#D4EBC2;color:#758952;">▣</div>
                <h3 style="font-family:Inter;margin:.1rem 0;">Camera Capture</h3>
                <div class="small-text">Allow camera access and take a photo.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="camera-real-card">', unsafe_allow_html=True)
            cam = st.camera_input("Take Photo", help="Izinkan akses kamera di browser jika diminta")
            st.markdown('</div>', unsafe_allow_html=True)
            if cam:
                image_source = cam
                cam_bytes_preview = np.frombuffer(cam.getvalue(), dtype=np.uint8)
                cam_bgr_preview = cv2.imdecode(cam_bytes_preview, cv2.IMREAD_COLOR)
                if cam_bgr_preview is not None:
                    cam_rgb_preview = cv2.cvtColor(cam_bgr_preview, cv2.COLOR_BGR2RGB)
                    cam_rgb_preview = cv2.flip(cam_rgb_preview, 1)  # mengikuti logic app_backup: hasil kamera tidak mirror
                    st.image(cam_rgb_preview, caption="Captured photo", use_container_width=True)
                else:
                    st.image(cam, caption="Captured photo", use_container_width=True)
        if st.button("Analyze Now  →", type="primary", use_container_width=True):
            if image_source is None:
                st.warning("upload or take a photo first")
            else:
                file_bytes = np.frombuffer(image_source.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    st.error("Gambar tidak bisa dibaca. Coba upload ulang dengan format JPG/PNG.")
                    st.stop()
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if mode == "Camera Capture":
                    img_rgb = cv2.flip(img_rgb, 1)  # referensi app_backup: koreksi mirror dari st.camera_input
                st.session_state["last_input_image"] = img_rgb.copy()
                with st.spinner("Menganalisis wajah..."):
                    result, error = run_pipeline(img_rgb, resources["face_mesh"], resources["ensemble"], resources["scaler"], resources["kmeans"], resources["centroids"], resources["df_found"], resources["mst_hex_lookup"], FEATURE_COLS)
                if error:
                    st.session_state["analysis_error"] = error
                    st.warning(error)
                else:
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_error"] = None
                    st.session_state["page"] = "Results"
                    st.rerun()
    with right:
        st.markdown("""
        <div class="tip-card">
            <h3 style="font-family:Inter;margin-top:0;">💡 Photo Tips</h3>
            <div class="tip-item"><div class="tip-emoji">☀️</div><div><strong>Natural Lighting</strong><span class="small-text">Use daylight or soft indoor light. Avoid flash and harsh shadows.</span></div></div>
            <div class="tip-item"><div class="tip-emoji">🚫</div><div><strong>No Filters</strong><span class="small-text">Upload the original photo without any color filters or edits.</span></div></div>
            <div class="tip-item"><div class="tip-emoji">👤</div><div><strong>Face Visible</strong><span class="small-text">Your face should be clearly visible and centered in the frame.</span></div></div>
            <div class="tip-item"><div class="tip-emoji">📐</div><div><strong>Straight Angle</strong><span class="small-text">Face the camera directly for best skin tone extraction.</span></div></div>
            <div class="tip-item"><div class="tip-emoji">💄</div><div><strong>Minimal Makeup</strong><span class="small-text">Less makeup gives more accurate skin color readings.</span></div></div>
        </div>
        <div class="notice">🔒 <strong>Your privacy matters.</strong> Photos are processed locally and are not stored or shared.</div>
        """, unsafe_allow_html=True)


def results_page():
    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("There are no analysis results yet. Please run analysis first.")
        if st.button("Go to Skin Analysis →", type="primary"):
            st.session_state["page"] = "Skin Analysis"
            st.rerun()
        return
    skin_hex = result.get("skin_hex", "#C4956A")
    user_skintone = result.get("user_skintone", "-")
    user_undertone = result.get("user_undertone", "-")
    mst = int(result.get("mst_pred", 4))
    confidence = float(result.get("confidence", 0))
    lab = result.get("cielab", {"L": 0, "a": 0, "b": 0})
    rgb_tuple = tuple(int(x) for x in ImageColor_get_rgb(skin_hex))
    display_skintone = "Medium Beige" if str(user_skintone).lower() == "medium" else user_skintone
    st.markdown('<span class="pill green">Step 2 of 3</span> <span class="pill green">✓ Analysis Complete</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Analysis Results</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="margin:0 0 1.2rem;max-width:760px;">Here’s what we found from your photo.</div>', unsafe_allow_html=True)
    main, side = st.columns([2.2, 1], gap="large")
    with main:
        cards = [("🌸 Skin Tone", display_skintone, "Fitzpatrick scale approximation", "#D94E91"), ("🍃 Undertone", user_undertone, "Golden–yellow hue bias detected" if user_undertone == "Warm" else "Estimated from CIELAB a* and b*", "#758952"), ("▥ MST Score", f"MST-{mst}", "Monk Skin Tone Scale", "#A66BCF"), ("✨ Confidence", f"{confidence}%", "Model prediction confidence", "#F28C43")]
        for i in range(0,4,2):
            cols = st.columns(2)
            for col, (label, value, sub, color) in zip(cols, cards[i:i+2]):
                with col:
                    st.markdown(f'<div class="metric-card" style="margin-bottom:1rem;"><div class="metric-label" style="color:{color};">{label}</div><div class="metric-value">{value}</div><div class="small-text">{sub}</div></div>', unsafe_allow_html=True)
        mst_items = ""
        for i, color in MST_COLORS.items():
            match = "match" if i == mst else ""
            bubble = '<div class="match-label">Your Match</div>' if i == mst else ""
            label_color = "#F48ABD" if i == mst else "#7B6472"
            weight = "900" if i == mst else "700"
            mst_items += f'<div class="mst-item {match}">{bubble}<div class="mst-color" style="background:{color};"></div><div style="font-size:.78rem;margin-top:.45rem;color:{label_color};font-weight:{weight};">MST-{i}</div></div>'
        st.markdown(f'<div class="custom-card" style="padding:1.6rem;margin-top:.1rem;"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;"><strong>Prediction Confidence</strong><strong style="color:#F48ABD;font-size:1.25rem;">{confidence}%</strong></div><div class="progress-track"><div class="progress-fill pink" style="width:{confidence}%;"></div></div><div style="display:flex;justify-content:space-between;color:#7B6472;font-size:.8rem;margin-top:.45rem;"><span>0%</span><span>100%</span></div><div style="margin-top:1.5rem;margin-bottom:.75rem;color:#7B6472;">Monk Skin Tone Scale</div><div class="mst-strip">{mst_items}</div><div style="text-align:center;color:#F48ABD;font-weight:900;margin-top:.9rem;">←──── MST-{mst} (Your Tone) ────→</div></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="custom-card result-card" style="padding:1.4rem;margin-top:1rem;"><strong>Color Space Analysis</strong><div class="color-grid"><div class="pink-tint color-box"><div class="metric-label">HEX</div><div class="small-text">Detected</div><strong class="color-hex-value">{skin_hex}</strong></div><div class="pink-tint color-box"><div class="metric-label">RGB</div><div>R <strong style="float:right;">{rgb_tuple[0]}</strong></div><div>G <strong style="float:right;">{rgb_tuple[1]}</strong></div><div>B <strong style="float:right;">{rgb_tuple[2]}</strong></div></div><div class="pink-tint color-box"><div class="metric-label">LAB</div><div>L* <strong style="float:right;">{lab.get('L')}</strong></div><div>a* <strong style="float:right;">{lab.get('a')}</strong></div><div>b* <strong style="float:right;">{lab.get('b')}</strong></div></div></div></div>""", unsafe_allow_html=True)
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        if st.button("View Foundation Recommendations  →", type="primary", use_container_width=True, key="view_recommendation_from_results"):
            st.session_state["page"] = "Recommendations"
            st.rerun()
    with side:
        with st.container(border=True):
            st.markdown('<strong>Preview Photo</strong>', unsafe_allow_html=True)
            preview = result.get("vis_frame", st.session_state.get("last_input_image", None))
            if preview is not None:
                st.markdown('<div class="analysis-preview-img" style="margin:.55rem 0 .75rem;">', unsafe_allow_html=True)
                st.image(preview, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<strong>Detected Skin Color</strong><div class="detected-skin-swatch" style="height:76px;background:{skin_hex};margin:.65rem 0 .75rem;"></div><div style="text-align:center;"><span class="pill" style="letter-spacing:0;text-transform:none;white-space:nowrap;">{skin_hex}</span><div class="small-text" style="margin-top:.42rem;">{display_skintone}</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)
        if st.button("📷 Re-analyze Photo", use_container_width=True, key="reanalyze_photo_results"):
            st.session_state["page"] = "Skin Analysis"
            st.rerun()
        st.markdown(f'<div class="custom-card result-about-card" style="padding:1.35rem;margin-top:.75rem;"><strong>ⓘ About Your Result</strong><div class="pink-tint" style="border-radius:.9rem;padding:1rem;margin-top:1rem;"><strong style="color:#F48ABD;">{user_undertone} Undertone</strong><div class="small-text">Foundation shades with matching undertone labels will complement your detected skin color better.</div></div><div class="green-tint" style="border-radius:.9rem;padding:1rem;margin-top:.75rem;"><strong style="color:#758952;">Best Finishes</strong><div class="small-text">Dewy and satin finishes often enhance the result with a natural radiance.</div></div></div>', unsafe_allow_html=True)



def recommendations_page():
    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("There are no recommendation results yet. Please run a skin analysis first.")
        if st.button("Go to Skin Analysis →", type="primary"):
            st.session_state["page"] = "Skin Analysis"
            st.rerun()
        return
    recs = pd.DataFrame(result.get("top5_recs", []))
    if recs.empty:
        st.warning("Tidak ada rekomendasi foundation yang tersedia.")
        return
    skin_hex = result.get("skin_hex", "#C4956A")
    mst_pred = result.get("mst_pred", "-")
    display_skintone = result.get("user_skintone", "-")
    if str(display_skintone).lower() == "medium": display_skintone = "Medium Beige"
    st.markdown('<div class="pill">Step 3 of 3</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Foundation Recommendations</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="margin:0 0 1.1rem;max-width:760px;">Showing shades matched to your skin tone • Sorted by similarity</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="custom-card" style="padding:1.2rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1rem;"><div style="display:flex;align-items:center;gap:1rem;"><div class="swatch" style="background:{skin_hex};width:64px;height:64px;"></div><div><div class="small-text">Your detected skin color</div><div style="font-weight:900;font-size:1.15rem;">{display_skintone} · {result.get("user_undertone","-")} Undertone · MST-{mst_pred}</div><div class="small-text">{skin_hex}</div></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="filters-header-box"><strong style="font-size:1.05rem;">Filters</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="filters-shell">', unsafe_allow_html=True)
    brand_options = ["All"] + sorted(recs["Brand"].dropna().astype(str).unique().tolist())
    st.markdown('<div class="filter-group"><div class="filter-group-title">Brand</div></div>', unsafe_allow_html=True)
    brand_filter = st.radio("Brand", brand_options, horizontal=True, key="rec_brand_filter", label_visibility="collapsed")
    undertone_options = ["All"] + sorted(recs["Undertone"].dropna().astype(str).unique().tolist()) if "Undertone" in recs else ["All"]
    st.markdown('<div class="filter-group"><div class="filter-group-title">Undertone</div></div>', unsafe_allow_html=True)
    undertone_filter = st.radio("Undertone", undertone_options, horizontal=True, key="rec_undertone_filter", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    filtered = recs.copy()
    if brand_filter != "All": filtered = filtered[filtered["Brand"].astype(str).eq(brand_filter)]
    if undertone_filter != "All" and "Undertone" in filtered: filtered = filtered[filtered["Undertone"].astype(str).eq(undertone_filter)]
    if "delta_e" not in filtered: filtered["delta_e"] = 6
    filtered["similarity"] = filtered["delta_e"].apply(safe_similarity)
    cols = st.columns(2, gap="large")
    for idx, (_, row) in enumerate(filtered.head(6).iterrows()):
        brand, product, shade = str(row.get("Brand","-")), str(row.get("Product","-")), str(row.get("Shade","-"))
        undertone = str(row.get("Undertone","-")); skintone = str(row.get("Skin tone", row.get("skintone_norm", "-")))
        price = format_rupiah(row.get("Price", "-"))
        hex_color = cielab_to_hex(row.get("lab_L",65), row.get("lab_a",10), row.get("lab_b",20))
        sim = float(row.get("similarity",88))
        img_path = product_image_path(brand, shade)
        img_src = encode_image_for_html(img_path) if img_path else ""
        ml = "25 ml" if brand.lower() == "omg" else "30 ml"
        img_tag = ("<img class='html-product-img' src='" + img_src + "'/>" if img_src else "<div class='html-product-img'></div>")
        html_card = f"""<div class="html-product-card"><div class="html-product-top"><div>{img_tag}</div><div><div class="html-brand">{ehtml(brand)}</div><div class="html-name">{ehtml(product)} - {ehtml(shade)}</div><div style="display:flex;align-items:center;gap:.55rem;"><div class="swatch" style="width:34px;height:28px;background:{hex_color};"></div><span class="small-text">{hex_color}</span></div><div style="margin-top:.6rem;"><span class="chip">{ehtml(undertone)}</span> <span class="chip">{ehtml(skintone)}</span></div></div><div class="html-price"><div>{ehtml(price)}</div><div class="small-text">{ml}</div><div style="margin-top:.6rem;" class="match-badge">▲ {sim:.1f}% match</div></div></div><div style="display:flex;justify-content:space-between;"><span class="small-text">Match Score</span><strong>{sim:.1f}%</strong></div><div class="html-bar"><div style="width:{sim:.1f}%;"></div></div></div>"""
        with cols[idx % 2]: st.markdown(html_card, unsafe_allow_html=True)
    report_img = create_analysis_report(result, dark_mode=True)
    buffer = BytesIO(); report_img.save(buffer, format="PNG"); buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(label="📥 Download Hasil Analisis", data=buffer, file_name=f"hasil_analisis_foundation_{timestamp}.png", mime="image/png", use_container_width=True)



def about_method_page():
    st.markdown('<div class="pill green">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">About the Method</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="margin:0 0 1.3rem;max-width:760px;">ShadeMate uses a computer vision pipeline to analyze facial skin color and match it to foundation shades using color science.</div>', unsafe_allow_html=True)
    steps = [("1","⇧","Input","Upload Image","#FFA8D6"),("2","⌗","Detection","Face Detection","#C58CE0"),("3","▰","Segmentation","Skin Area<br>Extraction","#FF9A57"),("4","↻","Transform","RGB → LAB<br>Conversion","#66B4E8"),("5","▦","Clustering","K-Means<br>Clustering","#758952"),("6","⌘","Matching","Euclidean Distance<br>Matching","#F48ABD")]
    pipeline_html = '<div class="custom-card pipeline-card"><h3 style="font-family:Inter;margin-top:0;">Processing Pipeline</h3><div class="method-timeline">'
    for num, icon, tag, title, color in steps:
        pipeline_html += f'<div class="method-step"><div class="step-badge" style="background:{color};">{num}</div><div class="method-icon" style="background:{color}22;color:{color};">{icon}</div><div class="chip active" style="background:{color}22;color:{color};border-color:{color}33;">{tag}</div><div class="method-title">{title}</div></div>'
    pipeline_html += '</div></div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)
    st.markdown('<div style="height:1.1rem;"></div>', unsafe_allow_html=True)
    method_cards = [("STEP 1","Upload Image","Input Layer","User provides a facial photograph via file upload or webcam capture. Accepted formats include PNG and JPG.","Resolution ≥ 480×480px recommended for accurate face detection.","#FFA8D6","⇧"),("STEP 2","Face Detection","Computer Vision","MediaPipe Face Landmarker detects the face landmarks within the image frame.","Model: face_landmarker.task with confidence threshold 0.3","#C58CE0","⌗"),("STEP 3","Skin Area Extraction","Segmentation","Within the detected face region, cheek, forehead, and nose skin areas are isolated with landmark-based masks.","Skin-like LAB pixels are filtered before feature extraction.","#FF9A57","▰"),("STEP 4","RGB → LAB Conversion","Color Science","Skin pixels are converted from RGB to the CIELAB color space. LAB is perceptually uniform for color difference comparison.","Using D65 illuminant. L*: lightness, a*: green-red axis, b*: blue-yellow axis.","#66B4E8","↻"),("STEP 5","K-Means Clustering","Machine Learning","K-Means features and scaled LAB statistics support the model in predicting the closest Monk Skin Tone.","kmeans_k*.pkl + scaler.pkl + best_model.pkl","#758952","▦"),("STEP 6","Euclidean Distance Matching","Recommendation Engine","The dominant LAB color is compared to foundation shade LAB values. The Euclidean distance (ΔE) determines similarity.","ΔE = √[(ΔL*)² + (Δa*)² + (Δb*)²] — lower ΔE means closer match.","#F48ABD","⌘")]
    for i in range(0,6,3):
        st.markdown('<div class="method-cards-row">', unsafe_allow_html=True)
        cols=st.columns(3,gap="large")
        for col,(step,title,tag,desc,note,color,icon) in zip(cols,method_cards[i:i+3]):
            with col:
                st.markdown(f'<div class="custom-card method-card" style="border-color:{color}66;"><div style="display:flex;align-items:center;gap:.9rem;"><div class="method-icon" style="background:{color}22;color:{color};margin:0;">{icon}</div><div><div class="metric-label" style="color:{color};">{step}</div><div style="font-weight:900;font-size:1.05rem;">{title}</div></div></div><div class="chip" style="margin-top:1rem;background:{color}18;color:{color};border-color:{color}33;">{tag}</div><div class="small-text" style="margin-top:.9rem;">{desc}</div><div class="code-note" style="background:{color}12;border:1px solid {color}55;color:{color};">{note}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-card tech-stack-box"><h3 style="font-family:Inter;margin-top:0;">Technology Stack</h3>', unsafe_allow_html=True)
    techs=[("🐍","Python","Core Language","#66B4E8"),("👑","Streamlit","Frontend Framework","#F48ABD"),("🔶","OpenCV","Computer Vision","#FF9A57"),("🌿","scikit-learn","Machine Learning","#758952"),("🧊","NumPy","Numeric Computing","#A66BCF"),("📊","Pandas","Data Processing","#66B4E8"),("💧","CIELAB ΔE","Color Space & Metric","#FF9A57"),("✣","K-Means","Clustering Algorithm","#758952")]
    for i in range(0,8,4):
        cols=st.columns(4, gap="medium")
        for col,(icon,name,desc,color) in zip(cols,techs[i:i+4]):
            with col: st.markdown(f'<div class="tech-card compact"><div class="tech-icon compact" style="background:{color}18;color:{color};">{icon}</div><div><strong>{name}</strong><div class="small-text">{desc}</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="ref-box" style="margin-top:1.25rem;"><strong>References</strong><div class="small-text" style="margin-top:1rem;line-height:2;"><span style="color:#F48ABD;font-weight:900;">[1]</span> Monk, D. S. (2019). Monk Skin Tone Scale. Google Research.<br><span style="color:#F48ABD;font-weight:900;">[2]</span> CIE (2004). Colorimetry, 3rd ed. — CIELAB color model specification.<br><span style="color:#F48ABD;font-weight:900;">[3]</span> MacAdam, D. L. (1942). Visual Sensitivities to Color Differences. JOSA.<br><span style="color:#F48ABD;font-weight:900;">[4]</span> MediaPipe Face Landmarker documentation.<br><span style="color:#F48ABD;font-weight:900;">[5]</span> scikit-learn documentation for K-Means and ensemble modeling.</div></div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="ShadeMate", page_icon="🌸", layout="wide", initial_sidebar_state="expanded")
    load_css()
    inject_ui_hotfix_css()
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    resources = get_resources_or_stop()
    page = render_sidebar()
    if page == "Home": home_page(resources["df_found"])
    elif page == "Skin Analysis": skin_analysis_page(resources)
    elif page == "Results": results_page()
    elif page == "Recommendations": recommendations_page()
    elif page == "About Method": about_method_page()


if __name__ == "__main__":
    main()
