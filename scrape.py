"""
MSOE Web Scraper for NLP Project
- Scrapes all pages listed in https://www.msoe.edu/sitemap.xml
- Crawls https://catalog.msoe.edu/ by following internal links
- Saves raw HTML and extracted text to output folders
"""

import os
import re
import time
import hashlib
import logging
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "scraped_data")

SITEMAP_URL = "https://www.msoe.edu/sitemap.xml"
CATALOG_ROOT = "https://catalog.msoe.edu/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 0.5  # seconds between requests — be polite
REQUEST_TIMEOUT = 30
MAX_CATALOG_PAGES = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "scraper.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def setup_dirs():
    dirs = {
        "sitemap_raw": os.path.join(OUTPUT_DIR, "sitemap", "raw_html"),
        "sitemap_text": os.path.join(OUTPUT_DIR, "sitemap", "text"),
        "catalog_raw": os.path.join(OUTPUT_DIR, "catalog", "raw_html"),
        "catalog_text": os.path.join(OUTPUT_DIR, "catalog", "text"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def url_to_filename(url: str) -> str:
    """Turn a URL into a safe, unique filename."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    if parsed.query:
        path += "_" + parsed.query.replace("&", "_").replace("=", "-")
    path = re.sub(r'[<>:"/\\|?*]', "_", path)
    if len(path) > 180:
        path = path[:140] + "_" + hashlib.md5(url.encode()).hexdigest()[:12]
    return path


def fetch(url: str, session: requests.Session) -> requests.Response | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        log.warning("Failed to fetch %s: %s", url, exc)
        return None


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def save_page(url: str, html: str, raw_dir: str, text_dir: str):
    fname = url_to_filename(url)
    raw_path = os.path.join(raw_dir, fname + ".html")
    text_path = os.path.join(text_dir, fname + ".txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(extract_text(html))


# ─── Sitemap scraper ────────────────────────────────────────────────────

def get_sitemap_urls(session: requests.Session) -> list[str]:
    log.info("Fetching sitemap from %s", SITEMAP_URL)
    resp = fetch(SITEMAP_URL, session)
    if resp is None:
        return []
    root = ET.fromstring(resp.content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text.strip() for loc in root.findall(".//s:loc", ns) if loc.text]
    log.info("Found %d URLs in sitemap", len(urls))
    return urls


def scrape_sitemap(session: requests.Session, dirs: dict):
    urls = get_sitemap_urls(session)
    total = len(urls)
    for i, url in enumerate(urls, 1):
        log.info("[sitemap %d/%d] %s", i, total, url)
        resp = fetch(url, session)
        if resp is None:
            continue
        save_page(url, resp.text, dirs["sitemap_raw"], dirs["sitemap_text"])
        time.sleep(REQUEST_DELAY)
    log.info("Sitemap scraping complete — saved %d pages", total)


# ─── Catalog crawler ────────────────────────────────────────────────────

def is_catalog_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "catalog.msoe.edu" and parsed.scheme in ("http", "https")


def discover_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"])
        clean, _ = urldefrag(full)
        if is_catalog_url(clean):
            links.add(clean)
    return links


def crawl_catalog(session: requests.Session, dirs: dict):
    visited: set[str] = set()
    queue: list[str] = [CATALOG_ROOT]
    page_count = 0

    while queue and page_count < MAX_CATALOG_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        log.info("[catalog %d] %s", page_count + 1, url)
        resp = fetch(url, session)
        if resp is None:
            continue

        save_page(url, resp.text, dirs["catalog_raw"], dirs["catalog_text"])
        page_count += 1

        new_links = discover_links(resp.text, url) - visited
        queue.extend(new_links)
        time.sleep(REQUEST_DELAY)

    log.info("Catalog crawling complete — saved %d pages", page_count)


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    dirs = setup_dirs()
    session = requests.Session()

    log.info("=" * 60)
    log.info("Starting MSOE scraper")
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("=" * 60)

    log.info("Phase 1: Scraping sitemap URLs from msoe.edu")
    scrape_sitemap(session, dirs)

    log.info("Phase 2: Crawling catalog.msoe.edu")
    crawl_catalog(session, dirs)

    log.info("=" * 60)
    log.info("All done! Data saved to %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
