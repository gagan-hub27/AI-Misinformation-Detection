# 🧠 AI Misinformation Detection System

## 📌 Overview
The AI Misinformation Detection System is an intelligent application designed to identify and classify news content as real or fake using advanced machine learning and natural language processing techniques. The system supports multiple input formats including text, images, videos, and live news feeds.

---

## 🚀 Features
- 📝 Text-based fake news detection
- 🖼️ Image analysis using OCR (Tesseract)
- 🎥 Video text extraction and analysis
- 📰 Real-time news analysis using News API
- 📊 Model evaluation with confusion matrix
- 🔍 Explainable AI output with reasoning

---

## 🧠 Technologies Used
- Python
- Streamlit (Frontend)
- BERT (Transformers)
- Scikit-learn (Logistic Regression)
- OpenCV (Video processing)
- Tesseract OCR (Image text extraction)
- NewsAPI (Live news fetching)

---

## ⚙️ How It Works
1. Input data is collected from user (text/image/video/news).
2. Text is extracted (OCR for images/videos).
3. Cleaned text is converted into embeddings using BERT.
4. A trained machine learning model classifies the content.
5. The system displays prediction, confidence, and explanation.

---

## 📊 Model Performance
- Accuracy: ~99%
- Evaluation: Confusion Matrix and Classification Report
- Dataset: Fake.csv & True.csv (combined dataset)

---

## 📂 Project Structure
