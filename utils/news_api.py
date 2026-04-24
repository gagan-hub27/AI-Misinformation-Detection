import requests
import os

API_KEY = os.getenv("NEWS_API_KEY")


def get_latest_news():
    try:
        if not API_KEY:
            print("⚠️ API key not found")
            return []

        url = "https://newsapi.org/v2/everything"

        params = {
            "q": "india OR politics OR technology OR world",
            "pageSize": 10,  # get more → filter better
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": API_KEY
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print("Error:", response.status_code)
            return []

        data = response.json()

        if data.get("status") != "ok":
            print("API Error:", data.get("message"))
            return []

        articles = []

        for article in data.get("articles", []):
            title = article.get("title", "")
            desc = article.get("description", "")

            # 🔥 FILTER BAD NEWS
            if not title or not desc:
                continue

            if len(title.split()) < 4:
                continue

            articles.append({
                "title": title.strip(),
                "description": desc.strip(),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "image": article.get("urlToImage", "")
            })

            # limit to 5 clean articles
            if len(articles) >= 5:
                break

        return articles

    except Exception as e:
        print("Error fetching news:", str(e))
        return []