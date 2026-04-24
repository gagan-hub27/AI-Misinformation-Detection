import pytesseract
import cv2
import numpy as np
from PIL import Image
import re

# 👉 Set path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- TEXT CLEAN ----------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,:%-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- IMAGE PREPROCESSING (🔥 IMPROVED) ----------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Resize (VERY IMPORTANT for OCR accuracy)
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpening 🔥
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


# ================= MAIN OCR ================= #
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        img = np.array(image)

        processed = preprocess_image(img)

        # 🔥 MULTI-PASS OCR
        text1 = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
        text2 = pytesseract.image_to_string(processed, config="--oem 3 --psm 11")

        text = clean_text(text1 + " " + text2)

        # 🔥 RELAXED FILTER (IMPORTANT FIX)
        if len(text.split()) < 4:
            return "breaking news update information unclear"

        return text

    except Exception as e:
        print("OCR error:", e)
        return ""


# ================= OCR WITH BOXES ================= #
def extract_text_with_boxes(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        img = np.array(image)

        processed = preprocess_image(img)

        data = pytesseract.image_to_data(
            processed,
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

            if conf > 60 and len(text) > 2:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                extracted_words.append(text)

        full_text = clean_text(" ".join(extracted_words))

        # 🔥 fallback fix
        if len(full_text.split()) < 4:
            return "text not clear from image", img

        return full_text, img

    except Exception as e:
        print("Box OCR error:", e)
        return "", None