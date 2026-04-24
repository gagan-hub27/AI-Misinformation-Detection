import cv2
import pytesseract
import numpy as np
import re

# ---------- TEXT CLEAN ----------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,:%-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- FRAME SIMILARITY CHECK (avoid duplicates) ----------
def is_similar(text, existing_texts):
    for t in existing_texts:
        if text[:50] in t or t[:50] in text:
            return True
    return False


# ================= VIDEO OCR ================= #
def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    texts = []
    frame_count = 0

    if not cap.isOpened():
        return ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 30th frame
        if frame_count % 30 == 0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Better preprocessing 🔥
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11, 2
                )

                # OCR
                text = pytesseract.image_to_string(
                    thresh,
                    config="--oem 3 --psm 6"
                )

                text = clean_text(text)

                # Filter weak/noisy text
                if len(text.split()) > 5 and not is_similar(text, texts):
                    texts.append(text)

            except Exception as e:
                print("OCR error:", e)

        frame_count += 1

        # Limit processing (fast demo)
        if frame_count > 300:
            break

    cap.release()

    return " ".join(texts)