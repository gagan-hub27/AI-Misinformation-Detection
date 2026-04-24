import cv2
import pytesseract
import numpy as np
import re

# ---------- TEXT CLEAN ----------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,:%-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- BLUR DETECTION ----------
def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 40   # 🔥 slightly relaxed (more frames allowed)


# ---------- TEXT SIMILARITY ----------
def is_similar(text, existing_texts):
    for t in existing_texts:
        if text[:50] in t or t[:50] in text:
            return True
    return False


# ---------- PREPROCESS FRAME ----------
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Slight blur removal (denoise)
    denoise = cv2.medianBlur(enhanced, 3)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


# ================= VIDEO OCR ================= #
def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    texts = []
    frame_count = 0

    if not cap.isOpened():
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)

    # 🔥 process more frames (better extraction)
    frame_skip = max(int(fps // 2), 10) if fps > 0 else 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            try:
                # Skip very blurry frames only
                if is_blurry(frame):
                    frame_count += 1
                    continue

                processed = preprocess_frame(frame)

                # 🔥 MULTI OCR (better recall)
                text1 = pytesseract.image_to_string(processed, config="--psm 6")
                text2 = pytesseract.image_to_string(processed, config="--psm 11")

                combined = clean_text(text1 + " " + text2)

                # 🔥 relaxed filter (important fix)
                if len(combined.split()) > 4 and not is_similar(combined, texts):
                    texts.append(combined)

            except Exception as e:
                print("OCR error:", e)

        frame_count += 1

        # Slightly higher limit
        if frame_count > 500:
            break

    cap.release()

    final_text = " ".join(texts)

    # 🔥 FALLBACK (VERY IMPORTANT FIX)
    if len(final_text.split()) < 5:
        return "breaking news update information not clear"

    return final_text