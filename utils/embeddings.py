from transformers import BertTokenizer, BertModel
import torch
import re

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL (OFFLINE MODE) ----------
try:
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        local_files_only=True
    )

    model = BertModel.from_pretrained(
        'bert-base-uncased',
        local_files_only=True
    )

except Exception as e:
    print("⚠️ Local model not found. Download once with internet.")
    raise e

model.to(device)
model.eval()


# ---------- TEXT CLEANING ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- EMBEDDING FUNCTION ----------
def get_embedding(text):
    try:
        # Handle empty input
        if not text or not text.strip():
            return torch.zeros((1, 768))

        # Clean text before embedding 🔥
        text = preprocess_text(text)

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference (fast + memory safe)
        with torch.no_grad():
            outputs = model(**inputs)

        # 🔥 Better pooling (CLS + Mean hybrid)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        mean_embedding = outputs.last_hidden_state.mean(dim=1)

        embedding = (cls_embedding + mean_embedding) / 2

        return embedding.cpu()

    except Exception as e:
        print("Embedding error:", e)
        return torch.zeros((1, 768))