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
def recommend_foundation(mst_pred, L, a, b, df_found, top_n=5):
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

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(img_rgb, face_mesh, ensemble, scaler,
                 kmeans, centroids, df_found, mst_hex_lookup, feature_cols):
    t0 = time.time()

    # Guard: pastikan RGB 3 channel
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
    
    return {
        "mst_pred"  : mst,
        "confidence": conf,
        "top3"      : top3_hex,
        "shade_name": top_rec["Shade"],
        "brand"     : top_rec["Brand"],
        "product"   : top_rec["Product"],
        # FIX Bug 2: Gunakan warna LAB spesifik shade produk, bukan warna rata-rata grup MST.
        # mst_hex adalah warna representatif seluruh grup MST yang sama untuk semua produk
        # dalam grup tersebut, sehingga tidak mencerminkan warna shade yang direkomendasikan.
        "hex_color" : cielab_to_hex(top_rec["lab_L"], top_rec["lab_a"], top_rec["lab_b"]),
        "skin_hex"  : skin_hex,
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

def create_analysis_report(result):
    # Canvas report
    W, H = 1400, 1000
    bg = (255, 255, 255)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Font fallback
    try:
        font_title = load_font(42, bold=True)
        font_h1 = load_font(34, bold=True)
        font_h2 = load_font(28, bold=True)
        font_text = load_font(24)
        font_small = load_font(21)
        font_table = load_font(20)
        font_big = load_font(54, bold=True)
    except:
        font_title = ImageFont.load_default()
        font_h2 = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_big = ImageFont.load_default()

    # Title
    draw.text((50, 35), "Foundation Shade Detector - Hasil Analisis", fill=(30, 30, 30), font=font_title)

    # Frame landmark
    frame = Image.fromarray(result["vis_frame"]).convert("RGB")
    frame.thumbnail((520, 380))
    draw.text((50, 115), "Frame + Landmark", fill=(30, 30, 30), font=font_h2)
    img.paste(frame, (50, 165))

    # Prediksi MST
    x = 630
    y = 115
    draw.text((x, y), "Prediksi MST", fill=(30, 30, 30), font=font_h2)

    # Color boxes
    skin_hex = result.get("skin_hex", "")
    detected_hex = skin_hex if skin_hex else "-"
    foundation_hex = result["hex_color"]

    draw.text((x, y + 60), "Warna Kulit Terdeteksi", fill=(40, 40, 40), font=font_text)
    draw.rounded_rectangle((x, y + 100, x + 260, y + 150), radius=12, fill=detected_hex if detected_hex != "-" else "#cccccc", outline=(200, 200, 200))
    draw.text((x + 285, y + 112), detected_hex, fill=(40, 40, 40), font=font_text)

    draw.text((x, y + 180), "Foundation Cocok", fill=(40, 40, 40), font=font_text)
    draw.rounded_rectangle((x, y + 220, x + 260, y + 270), radius=12, fill=foundation_hex, outline=(200, 200, 200))
    draw.text((x + 285, y + 232), foundation_hex, fill=(40, 40, 40), font=font_text)

    # MST card
    card_x = 1050
    card_y = 175
    draw.rounded_rectangle((card_x, card_y, card_x + 280, card_y + 150), radius=20, fill=(245, 245, 245), outline=(220, 220, 220))
    draw.text((card_x + 55, card_y + 35), f"MST {result['mst_pred']}", fill=(25, 25, 25), font=font_big)
    draw.text((card_x + 55, card_y + 105), f"Confidence: {result['confidence']}%", fill=(80, 80, 80), font=font_small)

    # Top 3
    draw.text((1050, 360), "Top-3 Alternatif MST", fill=(40, 40, 40), font=font_text)
    yy = 405
    for t in result["top3"]:
        draw.rounded_rectangle((1050, yy, 1085, yy + 35), radius=6, fill=t["hex"], outline=(180, 180, 180))
        draw.text((1100, yy + 3), f"MST {t['mst']} - {t['conf']}%", fill=(40, 40, 40), font=font_small)
        yy += 48

    # CIELAB
    draw.text((630, 470), "Nilai CIELAB Kulit", fill=(30, 30, 30), font=font_h2)
    draw.text((630, 525), f"L*  : {result['cielab']['L']}", fill=(40, 40, 40), font=font_text)
    draw.text((830, 525), f"a*  : {result['cielab']['a']}", fill=(40, 40, 40), font=font_text)
    draw.text((1030, 525), f"b*  : {result['cielab']['b']}", fill=(40, 40, 40), font=font_text)

    # Rekomendasi utama
    y2 = 620
    draw.line((50, y2 - 30, 1350, y2 - 30), fill=(220, 220, 220), width=2)
    draw.text((50, y2), "Rekomendasi Foundation", fill=(30, 30, 30), font=font_h2)

    rec_lines = [
        f"Brand      : {result['brand']}",
        f"Produk     : {result['product']}",
        f"Shade      : {result['shade_name']}",
        f"Undertone  : {result['undertone']}",
        f"Price      : {result['price']}",
    ]

    yy = y2 + 55
    for line in rec_lines:
        draw.text((50, yy), line, fill=(40, 40, 40), font=font_text)
        yy += 38

    # Top 5
    draw.text((700, y2), "Top-5 Rekomendasi Foundation", fill=(30, 30, 30), font=font_h2)
    yy = y2 + 55

    for i, rec in enumerate(result["top5_recs"][:5], start=1):
        brand = str(rec.get("Brand", "-"))
        shade = str(rec.get("Shade", "-"))
        undertone = str(rec.get("Undertone", "-"))
        price = format_rupiah(rec.get("Price", "-"))

        line = f"{i}. {brand} | {shade} | {undertone} | {price}"
        draw.text((700, yy), line[:55], fill=(40, 40, 40), font=font_small)
        yy += 40

    return img

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Foundation Shade Detector",
        page_icon="🎨",
        layout="wide",
    )

    st.markdown(
    """
    <style>
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {
        transform: scaleX(-1);
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.title("🎨 Foundation Shade Detector")
    st.caption("Monk Skin Tone (MST) Detection + Rekomendasi Foundation via Webcam")

    with st.spinner("Memuat model & database foundation..."):
        try:
            (face_mesh, ensemble, scaler,
             kmeans, df_found, centroids, mst_hex_lookup) = load_resources()
            st.success(f"✅ Model siap | Foundation DB: {len(df_found)} produk")
        except Exception as e:
            st.error(f"❌ Gagal load model: {e}")
            st.stop()

    with st.sidebar:
        st.header("ℹ️ Petunjuk")
        st.markdown("""
        1. Klik **'Ambil Foto'** di bawah
        2. Izinkan akses kamera jika diminta
        3. Pastikan wajah terlihat jelas & cahaya cukup
        4. Klik tombol kamera untuk mengambil foto
        5. Hasil prediksi MST & rekomendasi foundation akan muncul
        """)
        st.divider()
        st.markdown("**Tentang MST (Monk Skin Tone)**")
        st.markdown("Skala 1–10 untuk mengukur warna kulit secara inklusif, "
                    "dikembangkan oleh Dr. Ellis Monk (Google).")

        st.markdown("**Referensi Warna MST:**")
        mst_colors = {
            1: "#f6ede4", 2: "#f3e7db", 3: "#f7ead0", 4: "#eadaba",
            5: "#d7bd96", 6: "#a07850", 7: "#825c43", 8: "#604134",
            9: "#3a312a", 10: "#292420"
        }
        cols_mst = st.columns(5)
        for i, (mst_id, hex_c) in enumerate(mst_colors.items()):
            with cols_mst[i % 5]:
                st.markdown(
                    f'<div style="background:{hex_c};border-radius:6px;'
                    f'height:28px;display:flex;align-items:center;'
                    f'justify-content:center;color:{"#000" if i < 5 else "#fff"};'
                    f'font-size:11px;font-weight:bold">MST {mst_id}</div>',
                    unsafe_allow_html=True
                )

    st.subheader("📷 Input Foto")

    input_mode = st.radio(
        "Pilih metode input:",
        ["Kamera", "Upload Foto"],
        horizontal=True
    )

    camera_image = None
    uploaded_image = None

    if input_mode == "Kamera":
        camera_image = st.camera_input(
            "Ambil foto wajah",
            help="Izinkan akses kamera di browser jika diminta"
        )
    else:
        uploaded_image = st.file_uploader(
            "Upload foto wajah",
            type=["jpg", "jpeg", "png"]
        )

    image_source = camera_image if camera_image is not None else uploaded_image

    if image_source is not None:
        file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if input_mode == "Kamera":
            img_rgb = cv2.flip(img_rgb, 1)

        st.subheader("🔍 Hasil Analisis")

        with st.spinner("Menganalisis wajah..."):
            result, error = run_pipeline(
                img_rgb, face_mesh, ensemble, scaler,
                kmeans, centroids, df_found, mst_hex_lookup, FEATURE_COLS
            )

        if error:
            st.warning(error)
        else:
            # =========================
            # ROW 1: GAMBAR + PREDIKSI
            # =========================
            left_col, right_col = st.columns([1.15, 1.85], gap="large")

            with left_col:
                st.markdown("### Frame + Landmark")
                st.image(result["vis_frame"], width="stretch")

            with right_col:
                st.markdown("### Prediksi MST")

                skin_hex = cielab_to_hex(
                    result["cielab"]["L"],
                    result["cielab"]["a"],
                    result["cielab"]["b"]
                )

                pred_left, pred_right = st.columns([1.12, 1], gap="medium")

                # =========================
                # KIRI: WARNA KULIT + FOUNDATION
                # =========================
            with pred_left:
                st.markdown(
                    f"""
                    <div style="font-size:17px;font-weight:700;margin-bottom:6px;">
                        Warna Kulit Terdeteksi
                    </div>

                    <div style="
                        display:flex;
                        align-items:center;
                        gap:12px;
                        margin-bottom:18px;">
                        <div style="
                            background:{skin_hex};
                            border-radius:8px;
                            height:38px;
                            width:58%;
                            border:1px solid #ddd;">
                        </div>
                        <div style="
                            font-size:17px;
                            font-weight:700;
                            color:#333;">
                            {skin_hex}
                        </div>
                    </div>

                    <div style="font-size:17px;font-weight:700;margin-bottom:6px;">
                        Foundation Cocok
                    </div>

                    <div style="
                        display:flex;
                        align-items:center;
                        gap:12px;">
                        <div style="
                            background:{result["hex_color"]};
                            border-radius:8px;
                            height:44px;
                            width:58%;
                            border:1px solid #ddd;">
                        </div>
                        <div style="
                            font-size:17px;
                            font-weight:700;
                            color:#333;">
                            {result["hex_color"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # =========================
                # KANAN: MST + TOP 3
                # =========================
                with pred_right:
                    conf_pct = result["confidence"]
                    bar_color = "#2e8b57" if conf_pct >= 60 else "#e07b39" if conf_pct >= 40 else "#cc2222"

                    st.markdown(
                        f"""
                        <div style="
                            background:#f7f7f7;
                            border-radius:12px;
                            padding:15px 12px;
                            text-align:center;
                            border:1px solid #e0e0e0;
                            margin-bottom:6px;">
                            <div style="
                                font-size:38px;
                                font-weight:800;
                                line-height:1;">
                                MST {result["mst_pred"]}
                            </div>
                            <div style="
                                font-size:17px;
                                color:#555;
                                margin-top:5px;">
                                Confidence: {result["confidence"]}%
                            </div>
                        </div>

                        <div style="
                            background:#eeeeee;
                            border-radius:999px;
                            height:6px;
                            overflow:hidden;
                            margin-bottom:8px;">
                            <div style="
                                background:{bar_color};
                                width:{conf_pct}%;
                                height:100%;
                                border-radius:999px;">
                            </div>
                        </div>

                        <div style="
                            font-size:17px;
                            font-weight:700;
                            margin-bottom:4px;">
                            Top 3 Alternatif MST
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    for t in result["top3"]:
                        hex_c = t["hex"]
                        st.markdown(
                            f"""
                            <div style="
                                display:flex;
                                align-items:center;
                                gap:7px;
                                margin:3px 0;">
                                <div style="
                                    background:{hex_c};
                                    width:30px;
                                    height:30px;
                                    border-radius:5px;
                                    border:1px solid #ccc;
                                    flex-shrink:0;">
                                </div>
                                <span style="font-size:17px;">
                                    MST {t["mst"]} — {t["conf"]}%
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # =========================
                # CIELAB COMPACT
                # =========================
                st.markdown(
                    f"""
                    <div style="
                        margin-top:12px;
                        padding-top:10px;
                        border-top:1px solid #e2e2e2;">
                        <div style="
                            font-size:23px;
                            font-weight:800;
                            margin-bottom:8px;">
                            Nilai CIELAB Kulit
                        </div>
                        <div style="
                            display:grid;
                            grid-template-columns:repeat(3, 1fr);
                            gap:10px;">
                            <div>
                                <div style="font-size:15px;color:#444;">L* (kecerahan)</div>
                                <div style="font-size:24px;font-weight:500;line-height:1.1;">
                                    {result["cielab"]["L"]}
                                </div>
                            </div>
                            <div>
                                <div style="font-size:15px;color:#444;">a* (merah-hijau)</div>
                                <div style="font-size:24px;font-weight:500;line-height:1.1;">
                                    {result["cielab"]["a"]}
                                </div>
                            </div>
                            <div>
                                <div style="font-size:15px;color:#444;">b* (kuning-biru)</div>
                                <div style="font-size:24px;font-weight:500;line-height:1.1;">
                                    {result["cielab"]["b"]}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # ==========================================
            # ROW 2: REKOMENDASI UTAMA + TOP 5 PRODUK
            # ==========================================
            rec_col, top5_col = st.columns([1.1, 1.6], gap="large")

            with rec_col:
                st.markdown("### Rekomendasi Foundation")

                st.markdown(f"""
                | Info | Detail |
                |------|--------|
                | 🏷️ Brand | **{result['brand']}** |
                | 💄 Produk | {result['product']} |
                | 🎨 Shade | **{result['shade_name']}** |
                | 🌡️ Undertone | {result['undertone']} |
                | 💰 Price | {result['price']} |
                """)

            with top5_col:
                st.markdown("### Top-5 Rekomendasi Foundation")

                df_recs = pd.DataFrame(result["top5_recs"])[
                    ["Brand", "Product", "Shade", "Undertone", "Price"]
                ]

                df_recs["Price"] = df_recs["Price"].apply(format_rupiah)

                st.dataframe(
                    df_recs,
                    width="stretch",
                    hide_index=True
                )

                report_img = create_analysis_report(result)

                buffer = BytesIO()
                report_img.save(buffer, format="PNG")
                buffer.seek(0)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                st.download_button(
                    label="📥 Download Hasil Analisis",
                    data=buffer,
                    file_name=f"hasil_analisis_foundation_{timestamp}.png",
                    mime="image/png",
                    use_container_width=True
                )

            st.caption(f"⏱️ Waktu analisis: {result['latency_ms']} ms")


if __name__ == "__main__":
    main()