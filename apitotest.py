import hashlib
import pandas as pd
import requests  # For API calls if needed

def generate_id(title, text):
    # Create a hash based on title + text for uniqueness
    return hashlib.md5((title + text).encode('utf-8')).hexdigest()

def process_article(article):
    title = article.get('title', 'No Title')
    author = article.get('author', 'Unknown Author')
    text = article.get('content', article.get('description', ''))
    if not text:
        return None
    article_id = generate_id(title, text)
    return {
        'id': article_id,
        'title': title,
        'author': author,
        'text': text
    }

# Example: list of articles from an API or scraping
raw_articles = [
    {
        "title": "Government Launches New Policy",
        "author": "Jane Doe",
        "content": "The government announced a new policy today aimed at..."
    },
    {
        "title": "Economy Booms in 2025",
        "author": None,
        "description": "Experts say the economy is growing due to innovation in..."
    }
]

# Process and filter out invalid entries
processed_data = [process_article(article) for article in raw_articles if process_article(article)]

# Convert to DataFrame and save
df = pd.DataFrame(processed_data)
df.to_csv("formatted_news.csv", index=False)

print("Formatted CSV created with columns: id, title, author, text")
