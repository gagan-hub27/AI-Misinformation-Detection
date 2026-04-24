import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from utils.embeddings import get_embedding
from utils.ocr_utils import extract_text_from_image, extract_text_with_boxes
from utils.video_utils import extract_text_from_video
from utils.news_api import get_latest_news
from utils.clip_utils import analyze_image_clip   # 🔥 NEW

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Misinformation Detector", layout="centered")
st.title("🧠 AI Misinformation Detection System")

# ---------- SESSION ----------
if "result" not in st.session_state:
    st.session_state.result = None

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- QUALITY SCORE ----------
def get_text_quality_score(text):
    words = text.split()
    if len(words) == 0:
        return 0

    length_score = min(len(words) / 30, 1)
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)

    meaningful = [w for w in words if len(w) > 3 and w.isalpha()]
    meaningful_ratio = len(meaningful) / len(words)

    score = (0.4 * length_score) + (0.3 * alpha_ratio) + (0.3 * meaningful_ratio)
    return round(score, 2)

# ---------- TRUSTED ----------
trusted_sources = ["bbc", "ndtv", "reuters", "cnn", "the hindu", "times of india"]

def is_trusted(text):
    return any(src in text.lower() for src in trusted_sources)

# ---------- HIGHLIGHT ----------
def highlight_text(text):
    fake_words = ["miracle", "shocking", "hoax", "instant", "breaking"]
    words = text.split()

    out = []
    for w in words:
        if w.lower() in fake_words:
            out.append(f"<span style='color:red'><b>{w}</b></span>")
        else:
            out.append(w)

    return " ".join(out)

# ---------- PREDICT ----------
def predict(text, source="text"):
    try:
        quality = get_text_quality_score(text)

        # ❌ DO NOT BLOCK OCR
        if source == "text" and quality < 0.4:
            return None, 0.0, quality

        emb = get_embedding(text).numpy()
        emb = scaler.transform(emb)

        pred = model.predict(emb)[0]
        prob = model.predict_proba(emb)[0].max()

        confidence = prob * max(quality, 0.3)

        if source in ["image", "video"]:
            confidence = max(confidence, 0.5)

        if len(text.split()) < 20:
            confidence = min(confidence, 0.75)

        if is_trusted(text):
            pred = 1
            confidence = max(confidence, 0.85)

        return pred, confidence, quality

    except Exception as e:
        st.warning(f"⚠️ Prediction error: {e}")
        return None, 0.0, 0.0

# ---------- HYBRID IMAGE (🔥 BIG UPGRADE) ----------
def hybrid_image_prediction(image_file, extracted_text):
    clip_result = analyze_image_clip(image_file)

    clip_fake = clip_result.get("fake news", 0)
    clip_real = clip_result.get("real news", 0)

    text_pred, text_conf, quality = predict(extracted_text, "image")

    if text_pred is None:
        final_pred = 0 if clip_fake > clip_real else 1
        final_conf = max(clip_fake, clip_real)
        return final_pred, final_conf

    final_fake = (clip_fake * 0.5) + ((1 - text_pred) * text_conf * 0.5)
    final_real = (clip_real * 0.5) + (text_pred * text_conf * 0.5)

    final_pred = 1 if final_real > final_fake else 0
    final_conf = max(final_real, final_fake)

    return final_pred, final_conf

# ---------- SUMMARY ----------
def ai_summary(pred, confidence):
    if pred == 1:
        return f"Content appears reliable with {int(confidence*100)}% confidence."
    else:
        return f"Content shows signs of misinformation with {int(confidence*100)}% confidence."

# ---------- RESULT ----------
def show_result(pred, confidence, text):
    st.session_state.result = (pred, confidence, text)

def display_result():
    if st.session_state.result:
        pred, confidence, text = st.session_state.result

        st.progress(min(int(confidence * 100), 100))
        st.write(f"Confidence: {confidence:.2f}")

        if confidence > 0.85:
            st.success("High Confidence")
        elif confidence > 0.6:
            st.warning("Moderate Confidence")
        else:
            st.error("Low Confidence")

        if pred == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.subheader("🧠 AI Summary")
        st.info(ai_summary(pred, confidence))

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "🖼️ Image", "🎥 Video", "📰 Live News"])

# ---------- TEXT ----------
with tab1:
    text = st.text_area("Enter News Text")

    if st.button("Analyze Text"):
        if text.strip():
            pred, conf, _ = predict(clean_text(text), "text")
            show_result(pred, conf, text)
        else:
            st.warning("Enter some text")

# ---------- IMAGE ----------
with tab2:
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file:
        st.image(image_file)

        if st.button("Analyze Image"):

            # OCR + BOXES
            text, boxed_img = extract_text_with_boxes(image_file)

            if boxed_img is not None:
                st.image(boxed_img, caption="Detected Text")

            text = clean_text(text)

            st.subheader("📄 Extracted Text")
            st.markdown(highlight_text(text), unsafe_allow_html=True)

            pred, conf = hybrid_image_prediction(image_file, text)

            show_result(pred, conf, text)

# ---------- VIDEO ----------
with tab3:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file:
        if st.button("Analyze Video"):

            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            text = extract_text_from_video(temp_path)
            text = clean_text(text)

            st.subheader("📄 Extracted Text")
            st.markdown(highlight_text(text), unsafe_allow_html=True)

            pred, conf, _ = predict(text, "video")

            show_result(pred, conf, text)

            os.remove(temp_path)

# ---------- NEWS ----------
# ---------- NEWS (GOOGLE STYLE + FIX 🔥) ----------
with tab4:
    st.subheader("📰 Latest News Feed")

    if st.button("Fetch Latest News"):
        news_list = get_latest_news()

        if not news_list:
            st.warning("⚠️ Using demo news")
            news_list = [
                {
                    "title": "Government releases official report",
                    "description": "Economic growth confirmed",
                    "source": "NDTV",
                    "image": ""
                },
                {
                    "title": "Miracle cure spreads online",
                    "description": "Unverified claims circulate",
                    "source": "Unknown",
                    "image": ""
                }
            ]

        # 🔥 SHOW COUNT
        st.write(f"📰 Showing {len(news_list)} articles")

        # 🔥 GRID LAYOUT
        cols = st.columns(2)

        for i, news in enumerate(news_list):
            with cols[i % 2]:

                st.markdown(f"### 📰 {news['title']}")

                # IMAGE (if available)
                if news.get("image"):
                    st.image(news["image"], use_container_width=True)

                st.write(news["description"])

                st.markdown(f"**Source:** `{news.get('source', 'Unknown')}`")

                # 🔥 ANALYZE BUTTON
                if st.button(f"Analyze News {i}", key=f"news_{i}"):
                    text = news["title"] + " " + news["description"]

                    pred, conf, _ = predict(clean_text(text), "text")

                    show_result(pred, conf, text)

                st.markdown("---")

# ---------- DISPLAY ----------
display_result()

# ---------- CONFUSION ----------
st.subheader("📊 Model Evaluation")

if st.button("Show Confusion Matrix"):
    cm = np.load("conf_matrix.npy")
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    st.pyplot(plt)

