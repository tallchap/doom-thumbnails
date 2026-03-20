"""Brave Search API integration for source image gathering."""

import requests

from config import BRAVE_API_KEY


def search_images_brave(queries):
    """Search Brave Image API for source images."""
    if not BRAVE_API_KEY:
        return []
    results = []
    for query in queries:
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/images/search",
                headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
                params={"q": query, "count": 5, "safesearch": "off"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get("results", []):
                    thumb_url = ""
                    thumb_obj = r.get("thumbnail", {})
                    if isinstance(thumb_obj, dict):
                        thumb_url = thumb_obj.get("src", "")
                    props = r.get("properties", {})
                    img_url = ""
                    if isinstance(props, dict):
                        img_url = props.get("url", "")
                    if not img_url:
                        img_url = r.get("url", "")
                    if thumb_url or img_url:
                        results.append({
                            "url": img_url,
                            "thumbnail": thumb_url or img_url,
                            "title": r.get("title", ""),
                            "query": query,
                        })
        except Exception as e:
            print(f"Brave search error for '{query}': {e}")
    return results


def download_image_bytes(url, timeout=10):
    """Download an image from URL, return bytes or None."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
            return resp.content
    except Exception:
        pass
    return None
