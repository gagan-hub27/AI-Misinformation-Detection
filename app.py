import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from utils.embeddings import get_embedding
from utils.ocr_utils import extract_text_from_image
from utils.video_utils import extract_text_from_video
from utils.news_api import get_latest_news

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Misinformation Detector", layout="centered")
st.title("🧠 AI Misinformation Detection System")

# ---------- SESSION STATE ----------
if "result" not in st.session_state:
    st.session_state.result = None

# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- TEXT VALIDATION (NEW 🔥) ----------
def is_valid_text(text):
    words = text.split()

    if len(words) < 8:
        return False

    valid_chars = sum(c.isalnum() for c in text)
    if len(text) == 0 or valid_chars / len(text) < 0.6:
        return False

    return True

# ---------- TRUSTED SOURCES ----------
trusted_sources = ["bbc", "ndtv", "reuters", "cnn", "the hindu", "times of india"]

def is_trusted(text):
    return any(src in text.lower() for src in trusted_sources)

# ---------- PREDICTION ----------
def predict(text):
    try:
        emb = get_embedding(text).numpy()
        emb = scaler.transform(emb)

        pred = model.predict(emb)[0]
        prob = model.predict_proba(emb)[0].max()

        # 🔥 CONTROL CONFIDENCE
        if len(text.split()) < 12:
            prob = min(prob, 0.75)

        if is_trusted(text):
            pred = 1
            prob = max(prob, 0.85)

        return pred, prob

    except Exception as e:
        st.warning(f"⚠️ Prediction fallback used: {e}")
        return 0, 0.5

# ---------- EXPLANATION ----------
def explain(text, pred, confidence):
    reasons = []
    text_lower = text.lower()

    if is_trusted(text):
        reasons.append("The content includes references to trusted sources, increasing credibility.")

    if any(w in text_lower for w in ["official", "report", "announced"]):
        reasons.append("Formal language suggests structured and verified news reporting.")

    if any(w in text_lower for w in ["miracle", "shocking", "hoax", "instant"]):
        reasons.append("Sensational keywords indicate possible misinformation.")

    if len(text.split()) < 6:
        reasons.append("The content is too short for reliable classification.")

    reasons.append("Prediction is based on BERT semantic understanding and machine learning classification.")

    if confidence > 0.85:
        reasons.append("High confidence indicates strong similarity with training patterns.")
    else:
        reasons.append("Moderate confidence suggests mixed characteristics.")

    return reasons[:4]

# ---------- RESULT ----------
def show_result(text):
    pred, confidence = predict(text)
    st.session_state.result = (pred, confidence, text)

def display_result():
    if st.session_state.result:
        pred, confidence, text = st.session_state.result

        st.progress(min(int(confidence * 100), 100))
        st.write(f"Confidence: {confidence:.2f}")

        if pred == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.subheader("🔍 Explanation")
        for r in explain(text, pred, confidence):
            st.write("-", r)

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "🖼️ Image", "🎥 Video", "📰 Live News"])

# ================= TEXT ================= #
with tab1:
    text = st.text_area("Enter News Text")

    if st.button("Analyze Text"):
        if text.strip():
            show_result(text)
        else:
            st.warning("Enter some text")

# ================= IMAGE ================= #
with tab2:
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file:
        st.image(image_file)

        if st.button("Analyze Image"):
            extracted_text = extract_text_from_image(image_file)
            extracted_text = clean_text(extracted_text)

            if not is_valid_text(extracted_text):
                st.warning("⚠️ OCR text is too noisy or unclear. Try a clearer image.")
            else:
                st.subheader("📄 Extracted Text")
                st.write(extracted_text)
                show_result(extracted_text)

# ================= VIDEO ================= #
with tab3:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file:
        if st.button("Analyze Video"):
            temp_path = "temp_video.mp4"

            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            try:
                extracted_text = extract_text_from_video(temp_path)
                extracted_text = clean_text(extracted_text)

                if not is_valid_text(extracted_text):
                    st.warning("⚠️ Video text is not clear enough for analysis.")
                else:
                    st.subheader("📄 Extracted Text")
                    st.write(extracted_text[:500])
                    show_result(extracted_text)

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# ================= LIVE NEWS ================= #
with tab4:
    if st.button("Fetch Latest News"):
        try:
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

            for i, news in enumerate(news_list):
                st.markdown(f"### {news['title']}")
                st.write(news["description"])

                if news.get("image"):
                    st.image(news["image"])

                st.write(f"📰 Source: {news.get('source', 'Unknown')}")

                if st.button(f"Analyze {i}", key=f"news_{i}"):
                    text = news["title"] + " " + news["description"]
                    show_result(text)

        except Exception as e:
            st.error(f"❌ API Error: {e}")

# ---------- SHOW RESULT ----------
display_result()

# ================= CONFUSION MATRIX ================= #
st.subheader("📊 Model Evaluation")

if st.button("Show Confusion Matrix"):
    cm = np.load("conf_matrix.npy")

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    st.pyplot(plt)