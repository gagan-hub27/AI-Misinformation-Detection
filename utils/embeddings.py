import torch
import re
from functools import lru_cache
from transformers import BertTokenizer, BertModel

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL (ONCE) ----------
try:
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        local_files_only=True
    )

    model = BertModel.from_pretrained(
        "bert-base-uncased",
        local_files_only=True
    )

except Exception as e:
    print("⚠️ BERT model not found locally. Run once with internet.")
    raise e

model.to(device)
model.eval()

# ---------- TEXT PREPROCESS ----------
def preprocess_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    # remove OCR junk patterns
    text = re.sub(r'[^a-z0-9\s.,]', ' ', text)
    text = re.sub(r'\b[a-z]{1,2}\b', ' ', text)  # remove tiny useless words
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ---------- NORMALIZE FOR CACHE ----------
def normalize_for_cache(text: str) -> str:
    # ensures similar texts hit same cache
    text = preprocess_text(text)
    return text[:500]   # avoid huge cache keys


# ---------- EMBEDDING CORE ----------
def _embed(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 🔥 Hybrid pooling (CLS + Mean)
    cls = outputs.last_hidden_state[:, 0, :]
    mean = outputs.last_hidden_state.mean(dim=1)

    emb = (cls + mean) / 2

    return emb.cpu()


# ---------- CACHE (FAST 🔥) ----------
@lru_cache(maxsize=3000)
def _cached_embedding(text: str):
    return _embed(text)


# ---------- PUBLIC FUNCTION ----------
def get_embedding(text: str):
    try:
        if not text or not text.strip():
            return torch.zeros((1, 768))

        norm_text = normalize_for_cache(text)

        if not norm_text:
            return torch.zeros((1, 768))

        return _cached_embedding(norm_text)

    except Exception as e:
        print("Embedding error:", e)
        return torch.zeros((1, 768))