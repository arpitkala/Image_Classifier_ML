"""
🏅 Sports Classifier AI — Streamlit Web Application
=====================================================
Upload any sports image and get instant AI predictions
using a fine-tuned MobileNetV2 model.

Run:
    streamlit run app.py
"""

import os
import json
import time
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Page config (MUST be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="🏅 Sports Classifier AI",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient header ──────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 .4rem 0; }
.hero p  { font-size: 1.05rem; opacity: .85; margin: 0; }

/* ── Cards ────────────────────────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.8rem 2rem;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.result-card .sport-name {
    font-size: 2rem; font-weight: 700; letter-spacing: .03em;
}
.result-card .confidence {
    font-size: 1.1rem; opacity: .9; margin-top: .3rem;
}

.top3-card {
    background: #f8f9ff;
    border: 1px solid #e1e5ff;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: .6rem;
}
.top3-card .rank        { font-size: .8rem; color: #888; font-weight: 600; }
.top3-card .class-name  { font-size: 1.05rem; font-weight: 700; color: #1a1a2e; }
.top3-card .conf-text   { font-size: .9rem; color: #555; }
.top3-bar {
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    margin-top: .5rem;
}

/* ── Upload area ──────────────────────────────────────────────────────── */
.upload-hint {
    background: #f0f4ff;
    border: 2px dashed #667eea;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #555;
    margin-bottom: 1rem;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
.sidebar-title { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; }
.sidebar-step  { font-size: .9rem; color: #444; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model & class loading                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MODEL_PATH      = Path('model/model.h5')
CLASS_NAMES_PATH = Path('model/class_names.json')
IMG_SIZE        = 224


@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Load and cache the Keras model (runs once per session)."""
    import tensorflow as tf   # Lazy import — speeds up cold start
    if not MODEL_PATH.exists():
        return None
    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model


@st.cache_data(show_spinner=False)
def load_class_names():
    """Load class names from JSON, with a sensible fallback."""
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    # Fallback — common sports list
    return [
        'air hockey', 'ampute football', 'archery', 'arm wrestling',
        'axe throwing', 'balance beam', 'barell racing', 'baseball',
        'basketball', 'baton twirling', 'bike polo', 'billiards',
        'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding',
        'buzkashi', 'canoe slalom', 'cheerleading', 'cricket',
        'crossfit', 'curling', 'cycling', 'discus throw', 'fencing',
        'field hockey', 'figure skating', 'football', 'formula-1',
        'frisbee', 'golf', 'gymnastics', 'hammer throw', 'handball',
        'high jump', 'hockey', 'horse racing', 'hurdles', 'ice climbing',
        'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo',
        'kabaddi', 'karate', 'kayaking', 'lacrosse', 'log rolling',
        'long jump', 'luge', 'motorcycle racing', 'muay thai', 'naginata',
        'netball', 'octopush', 'paddleball', 'parkour', 'pole vault',
        'polo', 'pommel horse', 'rings', 'rock climbing', 'roller derby',
        'rowing', 'rugby', 'sailing', 'shot put', 'shuffleboard',
        'skeleton', 'ski jumping', 'skiing', 'skydiving', 'snowboarding',
        'softball', 'speed skating', 'squash', 'sumo wrestling', 'surfing',
        'swimming', 'table tennis', 'taekwondo', 'tennis', 'track cycling',
        'trampoline', 'triathlon', 'tug of war', 'ultimate', 'uneven bars',
        'volleyball', 'water polo', 'weightlifting', 'wheelchair basketball',
        'wheelchair racing', 'wingsuit flying', 'wrestling',
    ]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Prediction helpers                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, normalize to [0,1], add batch dimension."""
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image_array: np.ndarray, class_names: list) -> dict:
    """
    Run model inference.

    Returns:
        dict with predicted_class, confidence, top3_predictions
    """
    preds    = model.predict(image_array, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]

    return {
        'predicted_class' : class_names[top3_idx[0]],
        'confidence'      : float(preds[top3_idx[0]]),
        'top3_predictions': [
            {'class': class_names[i], 'confidence': float(preds[i])}
            for i in top3_idx
        ]
    }


def confidence_color(conf: float) -> str:
    """Return a color string based on confidence level."""
    if conf >= 0.80: return '#27ae60'   # Green
    if conf >= 0.50: return '#f39c12'   # Orange
    return '#e74c3c'                    # Red


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Sidebar                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown('<p class="sidebar-title">📖 How to Use</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-step">

**Step 1 →** Click <b>Browse files</b> and upload a sports image (JPG / PNG / WEBP).<br><br>
**Step 2 →** Press the <b>🔍 Classify Sport</b> button.<br><br>
**Step 3 →** View the predicted sport, confidence score, and top-3 alternatives.

</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sidebar-title">🧠 Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-step">
<b>Architecture:</b> MobileNetV2 (Fine-tuned)<br>
<b>Input size:</b> 224 × 224 px<br>
<b>Classes:</b> ~100 sports<br>
<b>Trained on:</b> Kaggle Sports Classification dataset
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sidebar-title">⚡ Tips</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-step">
• Use clear, well-lit images.<br>
• Center the athlete or equipment.<br>
• Avoid heavily cropped images.<br>
• Works best on single-sport images.
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main UI                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Hero banner
st.markdown("""
<div class="hero">
    <h1>🏅 Sports Classifier AI</h1>
    <p>Upload a sports image and let deep learning identify the sport in milliseconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
class_names = load_class_names()

with st.spinner('Loading AI model…'):
    model = load_model_cached()

if model is None:
    st.error(
        '⚠️ **Model not found!**\n\n'
        'Please run the notebook first to train and save `model/model.h5`, '
        'then restart this app.',
        icon='🚨'
    )
    st.info(
        '**Quick start:**\n'
        '1. Open `notebook.ipynb` in Google Colab or Jupyter.\n'
        '2. Run all cells (this trains the model and saves it).\n'
        '3. Download `model/model.h5` + `model/class_names.json`.\n'
        '4. Place them next to `app.py` and re-run `streamlit run app.py`.',
        icon='📋'
    )
    st.stop()

# ── Layout ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap='large')

with left_col:
    st.subheader('📤 Upload Image')

    st.markdown('<div class="upload-hint">Supported formats: JPG · PNG · WEBP · BMP</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label='Choose a sports image',
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        label_visibility='collapsed'
    )

    if uploaded_file:
        try:
            pil_img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f'❌ Could not open image: {e}')
            st.stop()

        st.image(pil_img, caption='Uploaded Image', use_container_width=True)

        # Image metadata
        w, h = pil_img.size
        mode = pil_img.mode
        st.caption(f'📐 {w} × {h} px  •  Mode: {mode}  •  Size: {uploaded_file.size / 1024:.1f} KB')

        predict_btn = st.button(
            '🔍 Classify Sport',
            type='primary',
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style="text-align:center; color:#888; padding: 3rem 0;">
            ⬆️ Upload an image to get started
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

with right_col:
    st.subheader('🎯 Prediction Results')

    if uploaded_file and predict_btn:

        # ── Run inference ────────────────────────────────────────────────────
        with st.spinner('🤖 Analysing image…'):
            try:
                t0          = time.perf_counter()
                img_array   = preprocess_image(pil_img)
                result      = predict(model, img_array, class_names)
                elapsed_ms  = (time.perf_counter() - t0) * 1000
            except Exception as e:
                st.error(f'❌ Prediction failed: {e}')
                st.stop()

        # ── Primary result card ──────────────────────────────────────────────
        sport  = result['predicted_class'].replace('-', ' ').title()
        conf   = result['confidence']
        color  = confidence_color(conf)

        st.markdown(f"""
        <div class="result-card">
            <div class="sport-name">🏆 {sport}</div>
            <div class="confidence">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Latency
        st.caption(f'⚡ Inference time: {elapsed_ms:.0f} ms')

        # Confidence interpretation
        if conf >= 0.80:
            st.success('✅ High confidence prediction')
        elif conf >= 0.50:
            st.warning('⚠️ Moderate confidence — consider the top-3 alternatives')
        else:
            st.error('❌ Low confidence — image may be ambiguous or blurry')

        st.markdown('---')

        # ── Top-3 predictions ────────────────────────────────────────────────
        st.markdown('#### 📊 Top-3 Predictions')

        medals = ['🥇', '🥈', '🥉']
        for rank, pred in enumerate(result['top3_predictions']):
            pct   = pred['confidence'] * 100
            name  = pred['class'].replace('-', ' ').title()
            bar_w = max(int(pct), 2)

            st.markdown(f"""
            <div class="top3-card">
                <div class="rank">{medals[rank]} Rank #{rank+1}</div>
                <div class="class-name">{name}</div>
                <div class="conf-text">{pct:.2f}%</div>
                <div class="top3-bar" style="width:{bar_w}%"></div>
            </div>
            """, unsafe_allow_html=True)

        # Expander with raw probabilities
        with st.expander('🔬 Full probability distribution'):
            import tensorflow as tf
            img_tf   = tf.constant(img_array)
            all_prob = model.predict(img_tf, verbose=0)[0]
            top10    = np.argsort(all_prob)[-10:][::-1]

            st.table({
                'Sport'      : [class_names[i].title() for i in top10],
                'Probability': [f'{all_prob[i]*100:.3f}%' for i in top10]
            })

    elif not uploaded_file:
        st.markdown("""
        <div style="text-align:center; color:#aaa; padding: 4rem 0;">
            <div style="font-size:4rem;">🏀</div>
            <p style="font-size:1.1rem; margin-top:1rem;">
                Results will appear here after uploading an image.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file and not predict_btn:
        st.info('👈 Click **🔍 Classify Sport** to analyse the uploaded image.')


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(
    '<div style="text-align:center; color:#aaa; font-size:.85rem;">'
    'Built with ❤️ using TensorFlow · MobileNetV2 · Streamlit'
    '</div>',
    unsafe_allow_html=True
)
