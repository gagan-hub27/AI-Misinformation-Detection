from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# ---------- DEVICE ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD MODEL ONCE (🔥 IMPORTANT) ----------
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()


# ---------- LABEL SET (🔥 IMPROVED PROMPTS) ----------
LABELS = [
    "a real news article screenshot",
    "a fake news image",
    "a misleading social media post",
    "a verified news report",
    "a propaganda or manipulated content"
]


# ================= MAIN FUNCTION ================= #
def analyze_image_clip(image_file):
    try:
        image = Image.open(image_file).convert("RGB")

        inputs = processor(
            text=LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1).cpu().numpy()[0]

        result = dict(zip(LABELS, probs))

        return result

    except Exception as e:
        print("CLIP error:", e)
        return {}