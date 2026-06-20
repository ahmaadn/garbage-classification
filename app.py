from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────── Konstanta ───────────────────────────────────────

IMAGE_SIZE: tuple[int, int] = (128, 128)
MODEL_PATH: str = "CNN_KAN.keras"
CONFIDENCE_THRESHOLD: float = 0.5

CLASS_META: dict[str, dict[str, str]] = {
    "battery": {"emoji": "🔋", "label": "Baterai", "label_en": "Battery"},
    "biological": {"emoji": "🌿", "label": "Organik", "label_en": "Organic"},
    "brown-glass": {"emoji": "🟤", "label": "Kaca Coklat", "label_en": "Brown Glass"},
    "cardboard": {"emoji": "📦", "label": "Kardus", "label_en": "Cardboard"},
    "clothes": {"emoji": "👕", "label": "Pakaian", "label_en": "Clothes"},
    "green-glass": {"emoji": "🍶", "label": "Kaca Hijau", "label_en": "Green Glass"},
    "metal": {"emoji": "🔩", "label": "Logam", "label_en": "Metal"},
    "paper": {"emoji": "📄", "label": "Kertas", "label_en": "Paper"},
    "plastic": {"emoji": "🧴", "label": "Plastik", "label_en": "Plastic"},
    "shoes": {"emoji": "👟", "label": "Sepatu", "label_en": "Shoes"},
    "trash": {"emoji": "🗑️", "label": "Sampah Umum", "label_en": "General Trash"},
    "white-glass": {"emoji": "🪟", "label": "Kaca Putih", "label_en": "White Glass"},
}

DEFAULT_CLASS_NAMES: list[str] = list(CLASS_META.keys())


# ─────────────────────────── i18n ──────────────────────────────────────────────


def _get_label(kelas: str) -> str:
    """Ambil label kelas sesuai bahasa aktif."""
    lang = st.session_state.get("lang", "id")
    meta = CLASS_META.get(kelas, {"emoji": "📌", "label": kelas, "label_en": kelas})
    return meta["label_en"] if lang == "en" else meta["label"]


TRANSLATIONS: dict[str, dict[str, str]] = {
    "id": {
        "page_title": "EcoScan — Klasifikasi Sampah",
        "hero_eyebrow": "CNN + Kolmogorov-Arnold Network",
        "hero_title_1": "Klasifikasi",
        "hero_title_span": "Sampah",
        "hero_title_2": "Berbasis AI",
        "hero_sub": "Unggah foto sampah dan model CNN-KAN akan mengidentifikasi kategorinya dari 12 kelas secara otomatis.",
        "upload_label": "📂 Sumber Gambar",
        "source_upload": "📂 Unggah File",
        "source_camera": "📷 Kamera",
        "camera_label": "Ambil foto",
        "file_uploader_label": "Unggah Gambar",
        "metadata_mode": "Mode",
        "empty_state": "Unggah gambar sampah<br>untuk memulai klasifikasi",
        "btn_classify": "🔍  Klasifikasikan Sekarang",
        "spinner_inference": "Menjalankan inferensi…",
        "spinner_load_model": "⚙️ Memuat model CNN-KAN…",
        "result_eyebrow": "Hasil Prediksi",
        "confidence_label": "Confidence Score",
        "meta_inference_time": "Waktu inferensi",
        "meta_input_size": "Input size",
        "meta_class_no": "Kelas ke-",
        "warn_low_confidence": "⚠️ Confidence di bawah 50% — model kurang yakin. Coba gambar yang lebih jelas dan terang.",
        "prob_title": "Distribusi Probabilitas",
        "prob_expander": "Lihat {n} kelas lainnya",
        "batch_expander": "🗂️  Prediksi Batch — Beberapa Gambar Sekaligus",
        "batch_desc": "Unggah beberapa file, model akan memproses semuanya dan menampilkan ringkasan.",
        "batch_uploader": "Pilih gambar batch",
        "batch_progress": "Memproses…",
        "batch_progress_n": "Memproses {i}/{n}…",
        "batch_summary": "Ringkasan Batch",
        "tbl_file": "File",
        "tbl_prediction": "Prediksi",
        "tbl_label": "Label ID",
        "tbl_confidence": "Confidence",
        "tbl_time": "Waktu (ms)",
        "sidebar_header": "♻️ Klasifikasi Sampah",
        "sidebar_sub": "CNN-KAN · 12 Kelas",
        "sidebar_arch": "Arsitektur",
        "sidebar_categories": "Kategori Sampah",
        "sidebar_footer": "TensorFlow · tf-kan-latest",
        "lang_label": "🌐 Bahasa",
        "lang_id": "🇮🇩 Indonesia",
        "lang_en": "🇬🇧 English",
        "err_no_tfkan": "❌ `tf-kan-latest` tidak ditemukan. Jalankan: `pip install tf-kan-latest`",
        "err_load_weights": "❌ Gagal memuat weights",
    },
    "en": {
        "page_title": "EcoScan — Waste Classification",
        "hero_eyebrow": "CNN + Kolmogorov-Arnold Network",
        "hero_title_1": "AI-Powered",
        "hero_title_span": "Waste",
        "hero_title_2": "Classification",
        "hero_sub": "Upload a photo of waste and the CNN-KAN model will automatically identify its category from 12 classes.",
        "upload_label": "📂 Image Source",
        "source_upload": "📂 Upload File",
        "source_camera": "📷 Camera",
        "camera_label": "Take photo",
        "file_uploader_label": "Upload Image",
        "metadata_mode": "Mode",
        "empty_state": "Upload a waste image<br>to start classification",
        "btn_classify": "🔍  Classify Now",
        "spinner_inference": "Running inference…",
        "spinner_load_model": "⚙️ Loading CNN-KAN model…",
        "result_eyebrow": "Prediction Result",
        "confidence_label": "Confidence Score",
        "meta_inference_time": "Inference time",
        "meta_input_size": "Input size",
        "meta_class_no": "Class #",
        "warn_low_confidence": "⚠️ Confidence below 50% — model is uncertain. Try a clearer, well-lit image.",
        "prob_title": "Probability Distribution",
        "prob_expander": "See {n} more classes",
        "batch_expander": "🗂️  Batch Prediction — Multiple Images",
        "batch_desc": "Upload multiple files, the model will process all of them and display a summary.",
        "batch_uploader": "Select batch images",
        "batch_progress": "Processing…",
        "batch_progress_n": "Processing {i}/{n}…",
        "batch_summary": "Batch Summary",
        "tbl_file": "File",
        "tbl_prediction": "Prediction",
        "tbl_label": "Label ID",
        "tbl_confidence": "Confidence",
        "tbl_time": "Time (ms)",
        "sidebar_header": "♻️ Waste Classification",
        "sidebar_sub": "CNN-KAN · 12 Classes",
        "sidebar_arch": "Architecture",
        "sidebar_categories": "Waste Categories",
        "sidebar_footer": "TensorFlow · tf-kan-latest",
        "lang_label": "🌐 Language",
        "lang_id": "🇮🇩 Indonesian",
        "lang_en": "🇬🇧 English",
        "err_no_tfkan": "❌ `tf-kan-latest` not found. Run: `pip install tf-kan-latest`",
        "err_load_weights": "❌ Failed to load weights",
    },
}


def _t(key: str, **kwargs: Any) -> str:
    """Terjemahkan kunci ke bahasa aktif."""
    try:
        lang = st.session_state.get("lang", "id")
    except Exception:
        lang = "id"
    text = TRANSLATIONS.get(lang, TRANSLATIONS["id"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text

# ─────────────────────────── CSS ─────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #F8FAFC;
    color: #1E293B;
}

/* ── Header ── */

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 999 !important;
}
[data-testid="collapsedControl"] button {
    color: #0F172A !important;
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* ── Sembunyikan elemen Streamlit default ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-eyebrow {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #16A34A;
    margin-bottom: 12px;
    background: #DCFCE7;
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 800;
    color: #0F172A;
    line-height: 1.15;
    margin: 0 0 10px;
    letter-spacing: -0.5px;
}
.hero-title span {
    background: linear-gradient(135deg, #16A34A, #22C55E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px;
    color: #64748B;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Divider ── */
.eco-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #16A34A;
    margin-bottom: 8px;
    display: block;
}
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #BBF7D0 !important;
    border-radius: 16px !important;
    padding: 8px !important;
    transition: border-color 0.25s, box-shadow 0.25s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #22C55E !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.12) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Tombol prediksi ── */
.stButton > button {
    background: linear-gradient(135deg, #16A34A, #22C55E) !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(22,163,74,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(22,163,74,0.35) !important;
}

/* ── Kartu hasil ── */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #16A34A, #22C55E, #4ADE80);
    border-radius: 4px 0 0 4px;
}
.result-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #16A34A;
    margin: 0 0 8px;
    background: #DCFCE7;
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
}
.result-kelas {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 4px;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.result-label-id {
    font-size: 14px;
    color: #64748B;
    margin: 0 0 20px;
    font-weight: 500;
}
.confidence-row {
    display: flex;
    align-items: center;
    gap: 16px;
}
.confidence-gauge {
    width: 72px;
    height: 72px;
    flex-shrink: 0;
}
.confidence-text-group {}
.confidence-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 30px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.confidence-sublabel {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
}
.meta-row {
    display: flex;
    gap: 20px;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #F1F5F9;
}
.meta-item {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
}
.meta-item span {
    color: #334155;
    font-weight: 600;
    background: #F1F5F9;
    padding: 2px 8px;
    border-radius: 6px;
}

/* ── Bar probabilitas custom ── */
.prob-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #16A34A;
    margin: 24px 0 14px;
}
.prob-row {
    display: grid;
    grid-template-columns: 120px 1fr 52px;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.prob-name {
    font-size: 13px;
    color: #334155;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-track {
    background: #F1F5F9;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.prob-fill.top { background: linear-gradient(90deg, #16A34A, #22C55E); }
.prob-fill.other { background: #CBD5E1; }
.prob-pct {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: #94A3B8;
    text-align: right;
}
.prob-pct.top { color: #16A34A; }

/* ── Warning confidence ── */
.warn-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400E;
    margin-top: 12px;
    font-weight: 500;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem;
}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    padding: 0.5rem 1rem !important;
    min-height: auto !important;
}
[data-testid="stSidebar"] button[kind="header"] {
    color: #0F172A !important;
}
[data-testid="stSidebar"] button[kind="header"]:hover {
    background: #F1F5F9 !important;
}
.sidebar-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #16A34A;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
}
.sidebar-logo-sub {
    font-size: 11px;
    color: #94A3B8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 20px;
}
.sidebar-section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 16px 0 8px;
}
.kelas-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #166534;
    font-weight: 500;
    margin: 3px 3px 3px 0;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #16A34A !important; }

/* ── Expander batch ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #334155 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
[data-testid="stMetric"] label {
    color: #94A3B8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Info kosong ── */
.empty-state {
    background: #FFFFFF;
    border: 2px dashed #E2E8F0;
    border-radius: 16px;
    padding: 48px 24px;
    text-align: center;
}
.empty-state-icon { font-size: 40px; margin-bottom: 12px; }
.empty-state-text { font-size: 14px; color: #94A3B8; font-weight: 500; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div {
    background: #F1F5F9 !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #16A34A, #22C55E) !important;
}
</style>
"""

# ─────────────────────────── Dataclass ───────────────────────────────────────


@dataclass(frozen=True)
class HasilPrediksi:
    kelas: str
    indeks: int
    confidence: float
    semua_probabilitas: dict[str, float]
    waktu_inferensi_ms: float


# ─────────────────────────── Model ───────────────────────────────────────────


def _bangun_arsitektur_cnn_kan() -> Any:
    import tensorflow as tf
    from tfkan.layers import DenseKAN

    silu = tf.keras.activations.swish
    l2 = tf.keras.regularizers.L2(1e-4)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.027777),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    def conv_block(filters: int, dropout_rate: float = 0.1) -> list[Any]:
        return [
            tf.keras.layers.Conv2D(
                filters, 3, padding="same", activation=silu, kernel_regularizer=l2
            ),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Conv2D(
                filters, 3, padding="same", activation=silu, kernel_regularizer=l2
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.SpatialDropout2D(dropout_rate),
        ]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(128, 128, 3)),
            augmentation,
            *conv_block(32, dropout_rate=0.1),
            *conv_block(64, dropout_rate=0.2),
            *conv_block(128, dropout_rate=0.2),
            *conv_block(128, dropout_rate=0.2),
            tf.keras.layers.Conv2D(
                256, 3, padding="same", activation=silu, kernel_regularizer=l2
            ),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Conv2D(
                256, 1, padding="same", activation=silu, kernel_regularizer=l2
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            DenseKAN(256, grid_size=3, basis_activation=silu),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Dropout(0.3),
            DenseKAN(128, grid_size=3, basis_activation=silu),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(12, activation="softmax"),
        ],
        name="CNN_KAN",
    )

    model(tf.zeros((1, 128, 128, 3)), training=False)
    return model


@st.cache_resource(show_spinner="⚙️ Memuat model CNN-KAN…")
def muat_model(model_path: str) -> Any:
    try:
        import tensorflow as tf  # noqa: F401
        from tfkan.layers import DenseKAN  # noqa: F401
    except ImportError:
        st.error(_t("err_no_tfkan"))
        return None
    try:
        model = _bangun_arsitektur_cnn_kan()
        model.load_weights(model_path)
        return model
    except Exception as exc:
        st.error(f"{_t('err_load_weights')}: {exc}")
        return None


def pra_proses(gambar: Image.Image) -> np.ndarray:
    arr = (
        np.array(
            gambar.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    return np.expand_dims(arr, 0)


def prediksi(model: Any, gambar: Image.Image) -> HasilPrediksi:
    t0 = time.perf_counter()
    prob: np.ndarray = model.predict(pra_proses(gambar), verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    idx = int(np.argmax(prob))
    return HasilPrediksi(
        kelas=DEFAULT_CLASS_NAMES[idx],
        indeks=idx,
        confidence=float(prob[idx]),
        semua_probabilitas={n: float(p) for n, p in zip(DEFAULT_CLASS_NAMES, prob)},
        waktu_inferensi_ms=ms,
    )


# ─────────────────────────── Komponen UI ─────────────────────────────────────


def gauge_svg(pct: float, warna: str) -> str:
    """SVG lingkaran gauge untuk confidence."""
    r = 26
    circ = 2 * 3.14159 * r
    dash = circ * pct
    return f"""
    <svg class="confidence-gauge" viewBox="0 0 68 68">
      <circle cx="34" cy="34" r="{r}" fill="none" stroke="#E2E8F0" stroke-width="6"/>
      <circle cx="34" cy="34" r="{r}" fill="none" stroke="{warna}" stroke-width="6"
        stroke-dasharray="{dash:.1f} {circ:.1f}"
        stroke-dashoffset="{circ / 4:.1f}"
        stroke-linecap="round"/>
    </svg>"""


def kartu_hasil(h: HasilPrediksi) -> None:
    meta = CLASS_META.get(h.kelas, {"emoji": "📌", "label": h.kelas, "label_en": h.kelas})
    label_display = _get_label(h.kelas)
    warna = "#16A34A" if h.confidence >= CONFIDENCE_THRESHOLD else "#F59E0B"
    st.markdown(
        f"""
    <div class="result-card">
        <p class="result-eyebrow">{_t("result_eyebrow")}</p>
        <p class="result-kelas">{meta["emoji"]} {h.kelas}</p>
        <p class="result-label-id">{label_display}</p>
        <div class="confidence-row">
            {gauge_svg(h.confidence, warna)}
            <div class="confidence-text-group">
                <div class="confidence-value" style="color:{warna}">{h.confidence:.1%}</div>
                <div class="confidence-sublabel">{_t("confidence_label")}</div>
            </div>
        </div>
        <div class="meta-row">
            <div class="meta-item">{_t("meta_inference_time")}&nbsp; <span>{h.waktu_inferensi_ms:.0f} ms</span></div>
            <div class="meta-item">{_t("meta_input_size")}&nbsp; <span>128×128 px</span></div>
            <div class="meta-item">{_t("meta_class_no")}&nbsp; <span>#{h.indeks + 1}</span></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if h.confidence < CONFIDENCE_THRESHOLD:
        st.markdown(
            f'<div class="warn-box">{_t("warn_low_confidence")}</div>',
            unsafe_allow_html=True,
        )


def bar_probabilitas(semua_prob: dict[str, float], top_n: int = 6) -> None:
    diurutkan = sorted(semua_prob.items(), key=lambda x: x[1], reverse=True)
    top_kelas = diurutkan[0][0]

    st.markdown(
        f'<p class="prob-section-title">{_t("prob_title")}</p>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for nama, prob in diurutkan[:top_n]:
        meta = CLASS_META.get(nama, {"emoji": "📌", "label": nama, "label_en": nama})
        is_top = nama == top_kelas
        fill_cls = "top" if is_top else "other"
        pct_cls = "top" if is_top else ""
        pct_w = f"{prob * 100:.1f}%"
        rows_html += f"""
        <div class="prob-row">
            <div class="prob-name">{meta["emoji"]} {nama}</div>
            <div class="prob-track">
                <div class="prob-fill {fill_cls}" style="width:{pct_w}"></div>
            </div>
            <div class="prob-pct {pct_cls}">{prob:.1%}</div>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    if len(diurutkan) > top_n:
        with st.expander(_t("prob_expander", n=len(diurutkan) - top_n)):
            sisa_html = ""
            for nama, prob in diurutkan[top_n:]:
                meta = CLASS_META.get(nama, {"emoji": "📌", "label": nama, "label_en": nama})
                pct_w = f"{prob * 100:.1f}%"
                sisa_html += f"""
                <div class="prob-row">
                    <div class="prob-name">{meta["emoji"]} {nama}</div>
                    <div class="prob-track">
                        <div class="prob-fill other" style="width:{pct_w}"></div>
                    </div>
                    <div class="prob-pct">{prob:.1%}</div>
                </div>"""
            st.markdown(sisa_html, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'<h1 class="sidebar-logo">{_t("sidebar_header")}</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-logo-sub">{_t("sidebar_sub")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<hr style="border-color:#E2E8F0; margin:0 0 12px">', unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="sidebar-section-label">{_t("sidebar_arch")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
        <div style="font-size:12px; color:#64748B; line-height:1.8">
            Model &nbsp;<span style="color:#0F172A;font-weight:600">CNN-KAN</span><br>
            Input &nbsp;&nbsp;<span style="color:#0F172A;font-weight:600">128×128 RGB</span><br>
            Backbone &nbsp;<span style="color:#0F172A;font-weight:600">5 Conv Blocks</span><br>
            Head &nbsp;&nbsp;<span style="color:#0F172A;font-weight:600">2× DenseKAN</span><br>
            Output &nbsp;<span style="color:#0F172A;font-weight:600">Softmax 12</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="sidebar-section-label" style="margin-top:20px">{_t("sidebar_categories")}</div>',
            unsafe_allow_html=True,
        )

        chips = "".join(
            f'<span class="kelas-chip">{v["emoji"]} {_get_label(k)}</span>'
            for k, v in CLASS_META.items()
        )
        st.markdown(f'<div style="line-height:2">{chips}</div>', unsafe_allow_html=True)

        st.markdown(
            '<hr style="border-color:#E2E8F0; margin:20px 0 10px">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:10px; color:#94A3B8; text-align:center; letter-spacing:1px;font-weight:600">'
            f'{_t("sidebar_footer")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<hr style="border-color:#E2E8F0; margin:16px 0 6px">',
            unsafe_allow_html=True,
        )
        lang = st.selectbox(
            _t("lang_label"),
            options=["id", "en"],
            format_func=lambda x: _t(f"lang_{x}"),
            key="lang",
            label_visibility="visible",
        )
        if lang != st.session_state.get("lang"):
            st.session_state.lang = lang
            st.rerun()


# ─────────────────────────── Halaman Utama ───────────────────────────────────


def halaman_utama() -> None:
    st.set_page_config(
        page_title=_t("page_title"),
        page_icon="♻️",
        layout="wide",
    )

    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Hero ──
    st.markdown(
        f"""
    <div class="hero">
        <div class="hero-eyebrow">{_t("hero_eyebrow")}</div>
        <h1 class="hero-title">{_t("hero_title_1")} <span>{_t("hero_title_span")}</span><br>{_t("hero_title_2")}</h1>
        <p class="hero-sub">{_t("hero_sub")}</p>
    </div>
    <hr class="eco-divider">
    """,
        unsafe_allow_html=True,
    )

    # ── Muat Model ──
    model = muat_model(MODEL_PATH)
    if model is None:
        return

    # ── Layout dua kolom ──
    col_kiri, col_kanan = st.columns([1, 1], gap="large")

    with col_kiri:
        st.markdown(
            f'<span class="upload-label">{_t("upload_label")}</span>', unsafe_allow_html=True
        )
        sumber = st.radio(
            "Pilih sumber gambar",
            [_t("source_upload"), _t("source_camera")],
            horizontal=True,
            label_visibility="collapsed",
        )

        gambar: Image.Image | None = None
        nama_file: str = ""

        if sumber == _t("source_camera"):
            cam = st.camera_input(_t("camera_label"), label_visibility="collapsed")
            if cam:
                gambar = Image.open(io.BytesIO(cam.read()))
                nama_file = "kamera.jpg"
        else:
            file_up = st.file_uploader(
                _t("file_uploader_label"),
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
            )
            if file_up:
                gambar = Image.open(io.BytesIO(file_up.read()))
                nama_file = file_up.name

        if gambar:
            st.image(gambar, width="stretch", caption=f"📎 {nama_file}")

            st.markdown(
                f"""
                <div style="display:flex;gap:16px;font-size:13px;color:#64748B;margin-top:8px;flex-wrap:wrap">
                    <span>📏 <b style="color:#0F172A">{gambar.size[0]}</b> × <b style="color:#0F172A">{gambar.size[1]}</b> px</span>
                    <span>🎨 {_t("metadata_mode")}: <b style="color:#0F172A">{gambar.mode}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_kanan:
        if gambar is None:
            st.markdown(
                f"""
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-text">{_t("empty_state")}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            if st.button(_t("btn_classify"), width="stretch"):
                with st.spinner(_t("spinner_inference")):
                    hasil = prediksi(model, gambar)
                kartu_hasil(hasil)
                bar_probabilitas(hasil.semua_probabilitas, top_n=6)

    # ── Batch ──
    st.markdown(
        '<hr class="eco-divider" style="margin-top:2rem">', unsafe_allow_html=True
    )
    with st.expander(_t("batch_expander")):
        st.markdown(
            f'<p style="font-size:13px;color:#64748B;margin-bottom:12px">{_t("batch_desc")}</p>',
            unsafe_allow_html=True,
        )
        files_batch = st.file_uploader(
            _t("batch_uploader"),
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="batch",
            label_visibility="collapsed",
        )

        if files_batch:
            hasil_batch: list[tuple[str, HasilPrediksi]] = []
            bar = st.progress(0, text=_t("batch_progress"))
            for i, f in enumerate(files_batch):
                img = Image.open(io.BytesIO(f.read()))
                h = prediksi(model, img)
                hasil_batch.append((f.name, h))
                bar.progress(
                    (i + 1) / len(files_batch),
                    text=_t("batch_progress_n", i=i + 1, n=len(files_batch)),
                )
            bar.empty()

            # Grid gambar
            cols_n = 3
            for i in range(0, len(hasil_batch), cols_n):
                cols = st.columns(cols_n)
                for j, (nama, h) in enumerate(hasil_batch[i : i + cols_n]):
                    with cols[j]:
                        files_batch[i + j].seek(0)
                        img_show = Image.open(io.BytesIO(files_batch[i + j].read()))
                        st.image(img_show, width="stretch")
                        meta = CLASS_META.get(
                            h.kelas, {"emoji": "📌", "label": h.kelas, "label_en": h.kelas}
                        )
                        warna = (
                            "#16A34A"
                            if h.confidence >= CONFIDENCE_THRESHOLD
                            else "#F59E0B"
                        )
                        st.markdown(
                            f"<div style='font-size:13px;margin-top:6px;background:#FFFFFF;"
                            f"border:1px solid #E2E8F0;border-radius:10px;padding:8px 10px'>"
                            f"<b style='color:#0F172A'>{meta['emoji']} {h.kelas}</b><br>"
                            f"<span style='color:{warna};font-weight:600'>{h.confidence:.1%}</span>"
                            f"<span style='color:#94A3B8'> · {h.waktu_inferensi_ms:.0f}ms</span></div>",
                            unsafe_allow_html=True,
                        )

            # Tabel
            st.markdown(
                f'<p class="prob-section-title" style="margin-top:20px">{_t("batch_summary")}</p>',
                unsafe_allow_html=True,
            )
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(
                [
                    {
                        _t("tbl_file"): nama,
                        _t("tbl_prediction"): f"{CLASS_META.get(h.kelas, {}).get('emoji', '')}{h.kelas}",
                        _t("tbl_label"): _get_label(h.kelas),
                        _t("tbl_confidence"): f"{h.confidence:.2%}",
                        _t("tbl_time"): f"{h.waktu_inferensi_ms:.0f}",
                    }
                    for nama, h in hasil_batch
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)


if __name__ == "__main__":
    halaman_utama()
