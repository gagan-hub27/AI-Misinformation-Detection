import requests
import os
import time
import re

API_KEY = os.getenv("NEWS_API_KEY")


# ---------- SAFE REQUEST (RETRY + RATE LIMIT 🔥) ----------
def fetch_with_retry(url, params, retries=3):
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                print("⚠️ Rate limited. Waiting...")
                time.sleep(5)
            else:
                print(f"Retry {i+1}: Status {response.status_code}")

        except Exception as e:
            print(f"Retry {i+1} failed:", e)

        time.sleep(2)

    return None


# ---------- TEXT CLEAN ----------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- SIMILARITY CHECK ----------
def is_similar(title, seen_titles):
    for t in seen_titles:
        if title[:50] in t or t[:50] in title:
            return True
    return False


# ---------- FILTER FUNCTION (RELAXED 🔥) ----------
def is_valid_article(title, desc):
    if not title or not desc:
        return False

    # 🔥 RELAXED CONDITIONS
    if len(title.split()) < 3:
        return False

    if len(desc.split()) < 5:
        return False

    if any(x in title.lower() for x in ["click here", "watch", "subscribe"]):
        return False

    return True


# ================= MAIN FUNCTION ================= #
def get_latest_news():
    try:
        if not API_KEY:
            print("⚠️ API key not found")
            return get_demo_news()

        url = "https://newsapi.org/v2/everything"

        params = {
            "q": "india OR politics OR technology OR world",
            "pageSize": 20,  # fetch more from API
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": API_KEY
        }

        data = fetch_with_retry(url, params)

        if not data or data.get("status") != "ok":
            print("⚠️ API failed, using demo news")
            return get_demo_news()

        articles = []
        seen_titles = []

        for article in data.get("articles", []):
            title = clean_text(article.get("title", ""))
            desc = clean_text(article.get("description", ""))

            # 🔥 FILTER
            if not is_valid_article(title, desc):
                continue

            # 🔥 REMOVE DUPLICATES
            if is_similar(title, seen_titles):
                continue

            seen_titles.append(title)

            articles.append({
                "title": title,
                "description": desc,
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "image": article.get("urlToImage", "")
            })

            # 🔥 INCREASE LIMIT (IMPORTANT)
            if len(articles) >= 10:
                break

        # fallback
        if not articles:
            return get_demo_news()

        return articles

    except Exception as e:
        print("Error fetching news:", str(e))
        return get_demo_news()


# ---------- DEMO FALLBACK ----------
def get_demo_news():
    return [
        {
            "title": "Government releases official economic report",
            "description": "Latest data confirms steady economic growth across multiple sectors.",
            "source": "NDTV",
            "url": "",
            "image": ""
        },
        {
            "title": "Shocking miracle cure spreads on social media",
            "description": "Experts warn about misleading claims circulating without scientific proof.",
            "source": "Unknown",
            "url": "",
            "image": ""
        }
    ]