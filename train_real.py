import pandas as pd
import numpy as np
import joblib
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from utils.embeddings import get_embedding

# ---------- SETTINGS ----------
EMBEDDING_FILE = "embeddings.npy"
LABEL_FILE = "labels.npy"
RANDOM_STATE = 42

# ---------- LOAD DATA ----------
print("📂 Loading dataset...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df = df[["text", "label"]]

# ---------- CLEAN DATA (🔥 IMPORTANT) ----------
df = df.dropna()
df = df[df["text"].str.strip() != ""]

# 🔥 Remove very short / useless text (CRITICAL for OCR performance)
df = df[df["text"].str.split().str.len() > 10]

# ⚡ Limit dataset size
df = df.head(4000)

print(f"📊 Dataset size: {len(df)}")

# ---------- CLASS BALANCE CHECK ----------
print("\n📊 Class Distribution:")
print(df["label"].value_counts())

# ---------- EMBEDDINGS ----------
if os.path.exists(EMBEDDING_FILE) and os.path.exists(LABEL_FILE):
    print("\n⚡ Loading saved embeddings...")
    X = np.load(EMBEDDING_FILE)
    y = np.load(LABEL_FILE)

else:
    print("\n🧠 Generating embeddings... (first time ⏳)")
    start_time = time.time()

    X = []
    y = df["label"].values

    for i, text in enumerate(df["text"]):
        try:
            emb = get_embedding(text).numpy()[0]
        except:
            emb = np.zeros(768)

        X.append(emb)

        if i % 200 == 0:
            print(f"Processed {i} samples...")

    X = np.array(X)

    # Save safely
    np.save(EMBEDDING_FILE, X)
    np.save(LABEL_FILE, y)

    print(f"⏱ Time taken: {round(time.time() - start_time, 2)} sec")

# ---------- FEATURE SCALING ----------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ---------- MODEL ----------
print("\n🤖 Training model...")

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    solver="lbfgs"
)

model.fit(X_train, y_train)

# ---------- EVALUATION ----------
print("\n📈 Evaluating model...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Accuracy:", round(acc, 4))
print("\n📊 Confusion Matrix:\n", cm)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------- EXTRA CHECK (🔥 IMPORTANT FOR YOUR CASE) ----------
fake_recall = cm[0][0] / (cm[0][0] + cm[0][1])
real_recall = cm[1][1] / (cm[1][0] + cm[1][1])

print(f"\n🎯 Fake Recall: {round(fake_recall, 3)}")
print(f"🎯 Real Recall: {round(real_recall, 3)}")

# Warn if biased
if fake_recall < 0.8:
    print("⚠️ Model weak at detecting FAKE news (important for project)")

# ---------- SAVE ----------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
np.save("conf_matrix.npy", cm)

print("\n🔥 Model, Scaler & Confusion Matrix saved successfully!")