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

# ---------- LOAD DATA ----------
print("📂 Loading dataset...")
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)
df = df[["text", "label"]]

# ⚡ Increase for better accuracy (if system allows)
df = df.head(4000)

print(f"📊 Dataset size: {len(df)}")

# ---------- EMBEDDINGS (CACHED) ----------
if os.path.exists(EMBEDDING_FILE) and os.path.exists(LABEL_FILE):
    print("\n⚡ Loading saved embeddings (FAST)...")
    X = np.load(EMBEDDING_FILE)
    y = np.load(LABEL_FILE)

else:
    print("\n🧠 Generating BERT embeddings... (first time only ⏳)")
    start_time = time.time()

    X = []
    for i, text in enumerate(df["text"]):
        emb = get_embedding(text).numpy()[0]
        X.append(emb)

        if i % 200 == 0:
            print(f"Processed {i} samples...")

    X = np.array(X)
    y = df["label"].values

    # Save embeddings
    np.save(EMBEDDING_FILE, X)
    np.save(LABEL_FILE, y)

    print(f"⏱ Time taken: {round(time.time() - start_time, 2)} sec")

# ---------- FEATURE SCALING (NEW 🔥) ----------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- MODEL ----------
print("\n🤖 Training model...")

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"   # 🔥 handles imbalance
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

# ---------- SAVE ----------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")   # 🔥 save scaler
np.save("conf_matrix.npy", cm)

print("\n🔥 Model, Scaler & Confusion Matrix saved successfully!")