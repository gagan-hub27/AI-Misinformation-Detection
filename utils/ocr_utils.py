import pytesseract
import cv2
import numpy as np
from PIL import Image
import re

# 👉 Uncomment ONLY if PATH issue occurs
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,:%-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ================= BASIC OCR ================= #
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        img = np.array(image)

        # ---------- PREPROCESSING ----------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold (better than fixed threshold 🔥)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Remove noise (morphological opening)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # ---------- OCR ----------
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, config=config)

        text = clean_text(text)

        # Minimum quality check
        if len(text.split()) < 5:
            return ""

        return text

    except Exception as e:
        print("OCR error:", e)
        return ""


# ================= OCR WITH HIGHLIGHT ================= #
def extract_text_with_boxes(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        img = np.array(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT
        )

        n_boxes = len(data['text'])
        extracted_words = []

        for i in range(n_boxes):
            text = data['text'][i].strip()

            try:
                conf = int(float(data['conf'][i]))
            except:
                conf = 0

            # Strong filtering 🔥
            if conf > 60 and len(text) > 2:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                extracted_words.append(text)

        full_text = clean_text(" ".join(extracted_words))

        if len(full_text.split()) < 5:
            return "", img

        return full_text, img

    except Exception as e:
        print("Box OCR error:", e)
        return "", None