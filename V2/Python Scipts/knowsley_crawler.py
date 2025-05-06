import requests
import re
import time
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from xml.etree import ElementTree
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load spaCy English model for keyword lemmatization
nlp = spacy.load("en_core_web_sm")

# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
SITEMAP_URL = "https://www.knowsley.gov.uk/sitemap.xml"
DOMAIN = "www.knowsley.gov.uk"
OUTPUT_DIR = "output"
COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_output.json")
HEADERS = {"User-Agent": "KnowsleyCrawler/1.0"}
CRAWL_VERSION = time.strftime("%Y-%m-%d")

# ───────────────────────────────
# REGEX MATCHING
# ───────────────────────────────
UK_PHONE_REGEX = re.compile(r"\b((?:\+44\s?7\d{3}|(?<!\d)07\d{3}|0151|0800|0300)[\s.-]?\d{3}[\s.-]?\d{3,4})\b")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# ───────────────────────────────
# FILTERING RULES
# ───────────────────────────────
EXCLUDE_URL_PATTERNS = [
    "privacy", "/print", "/pdf", "/share",
    "https://www.knowsley.gov.uk/website-patterns-components-and-design-library"
]
EXCLUDE_EXTENSIONS = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png", ".gif"]

# ───────────────────────────────
# STOPWORDS (used for keyword filtering)
# ───────────────────────────────
STOPWORDS = set([
    "the", "and", "a", "to", "of", "in", "for", "on", "with", "is", "by", "this", "that", "at", "an", "as",
    "be", "or", "it", "from", "are", "we", "you", "can", "our", "your", "if", "will", "has", "have", "was", "were",
    "but", "not", "they", "their", "which", "all", "more", "any", "so", "when", "what", "how", "do", "does"
])

# ───────────────────────────────
# INITIAL SETUP
# ───────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────────────
# CRAWLER FUNCTIONS
# ───────────────────────────────

# Get all URLs from the sitemap
def fetch_sitemap():
    print("Fetching sitemap...")
    resp = requests.get(SITEMAP_URL, headers=HEADERS)
    root = ElementTree.fromstring(resp.content)
    return [loc.text for loc in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]

# Check if a URL should be excluded
def is_valid_url(url):
    if any(url.startswith(pattern) for pattern in EXCLUDE_URL_PATTERNS):
        return False
    if any(url.lower().endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return False
    return True

# Extract breadcrumbs for department/topic tagging
def parse_breadcrumbs(soup):
    breadcrumbs = []
    nav = soup.find("nav", class_=lambda x: x and "breadcrumb" in x)
    if nav:
        for li in nav.find_all("li"):
            a = li.find("a")
            if a and a.text.strip().lower() not in ["home", "you are here"]:
                breadcrumbs.append(a.text.strip())
    department = breadcrumbs[0] if breadcrumbs else "Unknown"
    topics = breadcrumbs[1:] if len(breadcrumbs) > 1 else []
    return department, topics

# Pull external (non-Knowsley) links from page
def extract_external_links(content_block):
    links = []
    for a in content_block.find_all("a", href=True):
        href = a['href']
        text = a.get_text(strip=True)
        context = a.find_parent().get_text(strip=True) if a.find_parent() else ""
        link_info = {
            "url": href,
            "text": text,
            "context": context
        }
        if 'btn-start' in a.get("class", []):
            link_info["important"] = True
        links.append(link_info)
    return links

# Lemmatize and filter content text for keywords
def lemmatize_words(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in STOPWORDS and len(token.lemma_) > 2]

# Top frequent words
def extract_keywords(text, n=3):
    if not text:
        return []
    lemmas = lemmatize_words(text.lower())
    most_common = Counter(lemmas).most_common(n)
    return [word for word, _ in most_common]

# Top TF-IDF keywords
def extract_keywords_tfidf(text, n=3):
    if not text:
        return []
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:n]]

# Common two-word phrases
def extract_bigrams(text, n=3):
    lemmas = lemmatize_words(text.lower())
    bigrams = zip(lemmas, lemmas[1:])
    bigram_counts = Counter([" ".join(pair) for pair in bigrams])
    return [phrase for phrase, _ in bigram_counts.most_common(n)]

# Extract useful content and metadata from a single page
def extract_data(url, html):
    soup = BeautifulSoup(html, "html.parser")

    # Skip service landing pages
    article = soup.find("article", class_="service-landing-page")
    if article:
        print(f"[SKIP] Service landing page detected at {url}")
        return None

    # Main extraction targets
    title_tag = soup.find("h1")
    summary_tag = soup.find(class_="lgd-page-title-block__subheader")
    content_block = soup.find("div", class_="lgd-region lgd-region--content region region-content")

    # Clean out unnecessary sections
    if content_block:
        for tag in content_block.find_all(class_="feedback"):
            tag.decompose()
        for tag in content_block.find_all(class_="lgd-footer"):
            tag.decompose()

    # Pull alert banners (if present)
    alert_banner = soup.find("div", class_="block block-localgov-alert-banner block-localgov-alert-banner-block")
    alert_text = alert_banner.get_text(separator="\n", strip=True) if alert_banner else ""

    # Extract visible content
    title = title_tag.get_text(strip=True) if title_tag else ""
    summary = summary_tag.get_text(strip=True) if summary_tag else ""
    intro = summary
    text = content_block.get_text(separator="\n", strip=True) if content_block else ""

    # Extract metadata
    phones = list(set(UK_PHONE_REGEX.findall(html)))
    emails = list(set(EMAIL_REGEX.findall(html)))
    links = extract_external_links(content_block) if content_block else []
    department, topics = parse_breadcrumbs(soup)
    keywords = extract_keywords(text)
    tfidf_keywords = extract_keywords_tfidf(text)
    bigrams = extract_bigrams(text)

    return {
        "url": url,
        "title": title,
        "source": title,
        "summary": summary,
        "intro": intro,
        "text": text,
        "alert": alert_text,
        "has_alert": bool(alert_text),
        "phone": phones,
        "email": emails,
        "links": links,
        "department": department,
        "topics": topics,
        "keywords": keywords,
        "tfidf_keywords": tfidf_keywords,
        "bigrams": bigrams,
        "crawl_version": CRAWL_VERSION
    }

# Crawl all pages and compile full dataset
def crawl():
    urls = fetch_sitemap()
    print(f"Found {len(urls)} URLs in sitemap.")

    combined_data = []
    department_data = defaultdict(list)
    count = 0

    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Processing: {url}")
        if not is_valid_url(url):
            print(f"[SKIP] Invalid or excluded URL: {url}")
            continue

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            continue

        data = extract_data(url, html)
        if data:
            combined_data.append(data)
            department_data[data['department']].append(data)
            print(f"[OK] Saved page data ({data['title']})")
            count += 1

        time.sleep(1)  # Respectful delay between requests

    # Save combined output
    with open(COMBINED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    # Save output split by department
    for dept, records in department_data.items():
        safe_name = dept.replace(" ", "_").lower()
        filename = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(records)} pages to {filename}")

    print(f"\n✅ Crawl complete. {count} pages saved.")

# Run the crawler
if __name__ == "__main__":
    crawl()
