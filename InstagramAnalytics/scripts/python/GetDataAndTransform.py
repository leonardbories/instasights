import pyodbc
import pandas as pd
from collections import Counter
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
import re
import requests
import nltk
import os
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# === CONFIGURATION AND INITIALIZATION ===
# API access, database connection, account IDs, sentiment analyzer
# API-Zugriff, SQL-Verbindung, Account-IDs und Sentiment-Analyzer

API_TOKEN = "INSERT_YOUR_API_TOKEN_HERE"
SQL_CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=DESKTOP-B1VB63R\\SQLEXPRESS;"
    "Database=InstagramAnalytics;"
    "Trusted_Connection=yes;"
)

accounts_config = [
    {
        "username": "brand1",
        "dataset_profile": "iBTaTuIcbIjUSR9oE",
        "dataset_posts": "Yu6OKsqwfJh6OSOn7",
        "dataset_comments": "dHt18V2Smsw3bgh5L",
    },
]

vader = SentimentIntensityAnalyzer()


# === APIFY API ACCESS ===
# Functions to fetch and back up Instagram data from Apify API
# Funktionen zum Abrufen und Backup von Instagram-Daten über die Apify API

import os
import json
from datetime import datetime

import os
import json
from datetime import datetime

# Lokaler Zielpfad für Backups
BACKUP_BASE_PATH = r"C:\Users\marto\OneDrive\Bilder\Website\Portfolio\InstagramAnalytics\backups"

def get_dataset(dataset_id, label=None):
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={API_TOKEN}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Fehler beim Abrufen von Dataset {dataset_id}: {response.status_code}")
    
    data = response.json()

    # Backup speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label or dataset_id
    backup_dir = os.path.join(BACKUP_BASE_PATH, safe_label)
    os.makedirs(backup_dir, exist_ok=True)

    backup_path = os.path.join(backup_dir, f"data_{timestamp}.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Backup erstellt: {backup_path}")
    return data

def get_scraper_data_sync(results_type, direct_urls, results_limit=100):
    
    print(f"[INFO] Starte Scraper für type='{results_type}' mit {len(direct_urls)} URL(s)")

    api_url = f"https://api.apify.com/v2/acts/apify~instagram-scraper/run-sync-get-dataset-items?token={API_TOKEN}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "addParentData": False,
        "directUrls": direct_urls,
        "enhanceUserSearchWithFacebookPage": False,
        "isUserReelFeedURL": False,
        "isUserTaggedFeedURL": False,
        "onlyPostsNewerThan": "90 days",
        "resultsLimit": results_limit,
        "resultsType": results_type,
        "searchLimit": 1,
        "searchType": "hashtag"  # irrelevant bei directUrls
    }

    response = requests.post(api_url, headers=headers, json=payload)
    print("[INFO] Status:", response.status_code)

    if response.status_code != 201:
        raise Exception(f"[ERROR] Sync-Run fehlgeschlagen: {response.text}")
    
    return response.json()




# === DATABASE HELPER FUNCTIONS ===
# Checking for existing posts and comments
# Überprüfen existierender Posts und Kommentare

def get_existing_post_ids(cursor, account_id):
    cursor.execute("SELECT PostID FROM Post WHERE AccountID = ?", account_id)
    return set(row[0] for row in cursor.fetchall())

def comment_exists(cursor, comment_id):
    cursor.execute("SELECT 1 FROM Comment WHERE CommentID = ?", comment_id)
    return cursor.fetchone() is not None




# === TEXT PREPROCESSING AND NLP ===
# Cleaning text, topic modeling with BERTopic, sentiment analysis
# Bereinigung von Texten, Topic Modeling mit BERTopic, Sentimentanalyse

def preprocess_texts(texts):
    stop_words = set(stopwords.words("english"))
    cleaned_texts = []
    for text in texts:
        text = re.sub(r"[^\w\s]", "", text.lower())
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
        cleaned_texts.append(" ".join(filtered))
    return cleaned_texts

def train_bertopic_model(texts):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.1, random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=1,
        verbose=True
    )
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics, embeddings

def compute_frequencies(cleaned_texts):
    all_words = []
    for doc in cleaned_texts:
        all_words.extend(doc.split())
    return Counter(all_words)

def analyze_sentiment(text):
    if not text:
        return None
    score = vader.polarity_scores(text)
    return round(score['compound'], 3)


# === DATABASE INSERT FUNCTIONS ===
# Insert statements for accounts, posts, hashtags, comments, etc.
# Insert-Statements für Accounts, Posts, Hashtags, Kommentare etc.

def insert_account(cursor, profile):
    account_id = int(profile['id'])
    name = profile.get('fullName')
    bio = profile.get('biography')
    url = profile.get('url')
    profile_pic = profile.get('profilePicUrl')

    cursor.execute("""
        MERGE INTO Account AS target
        USING (SELECT ? AS AccountID, ? AS Name, ? AS Biography, ? AS URL, ? AS ProfilePicUrl) AS source
        ON target.AccountID = source.AccountID
        WHEN MATCHED THEN
            UPDATE SET Name = source.Name, Biography = source.Biography, URL = source.URL, ProfilePicUrl = source.ProfilePicUrl
        WHEN NOT MATCHED THEN
            INSERT (AccountID, Name, Biography, URL, ProfilePicUrl)
            VALUES (source.AccountID, source.Name, source.Biography, source.URL, source.ProfilePicUrl);
    """, account_id, name, bio, url, profile_pic)

    return account_id

def insert_account_snapshot(cursor, profile, account_id):
    followers = profile.get('followersCount')
    retrieved_at = datetime.now()
    cursor.execute("""
        INSERT INTO AccountSnapshot (AccountID, Followers, RetrievedAt)
        VALUES (?, ?, ?)
    """, account_id, followers, retrieved_at)

def insert_post(cursor, row, account_id, followers):
    caption = row.get('caption')
    likes = row.get('likesCount', 0)
    comments = row.get('commentsCount', 0)
    engagement = (likes + comments) / followers if followers else None
    post_type = row.get('type', 'image').lower()
    video_duration = row.get('videoDuration') if 'videoDuration' in row else None
    video_plays = row.get('videoPlayCount') if 'videoPlayCount' in row else None
    video_views = row.get('videoViewCount') if 'videoViewCount' in row else None
    view_retention = video_views / video_plays if video_plays and video_plays != 0 else None
    timestamp = pd.to_datetime(row.get('timestamp') or datetime.now())

    cursor.execute("""
    IF NOT EXISTS (SELECT 1 FROM Post WHERE PostID = ?)
    INSERT INTO Post (
        PostID, AccountID, URL, Caption, DisplayUrl,
        Likes, Comments, Engagement, Date, PostType,
        VideoDuration, VideoPlays, VideoViews, ViewRetention
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    int(row['id']), int(row['id']), account_id, row.get('url'), caption, row.get('displayUrl'),
    likes, comments, engagement, timestamp, post_type,
    video_duration, video_plays, video_views, view_retention)

    insert_hashtags(cursor, int(row['id']), row.get('hashtags'))

def insert_hashtags(cursor, post_id, hashtags_list):
    if not hashtags_list or not isinstance(hashtags_list, list):
        return

    for tag in hashtags_list:
        tag_cleaned = tag.strip().lower()
        if tag_cleaned:  # Avoid empty entries
            cursor.execute("""
                INSERT INTO Hashtag (PostID, Tag)
                VALUES (?, ?)
            """, post_id, tag_cleaned)


def insert_comment_and_replies(cursor, comment, post_id, sentiment, topic_id):
    if not comment.get('text'):
        return
    comment_id = int(comment['id'])
    cursor.execute("""
        IF NOT EXISTS (SELECT 1 FROM Comment WHERE CommentID = ?)
        INSERT INTO Comment (CommentID, PostID, Text, LikesCount, RepliesCount, OwnerUserName, OwnerID, SentimentScore, TopicID, Timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        comment_id, comment_id, post_id, comment.get('text'), comment.get('likesCount', 0),
        comment.get('repliesCount', 0), comment.get('ownerUsername'),
        comment.get('owner', {}).get('id'), sentiment, topic_id,
        comment.get("timestamp"))

    for reply in comment.get('replies', []):
        reply_sentiment = analyze_sentiment(reply.get('text'))
        reply_id = int(reply['id'])
        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM Reply WHERE ReplyID = ?)
            INSERT INTO Reply (ReplyID, CommentID, Text, OwnerUserName, OwnerID, Timestamp, LikesCount, SentimentScore, TopicID)
            VALUES (?, ?, ?, ?, ?, ?, ? , ? ,?)
        """,
            reply_id, reply_id, comment_id, reply.get('text'), reply.get('ownerUsername'),
            reply.get('owner', {}).get('id'), reply.get("timestamp"),
            reply.get('likesCount', 0), reply_sentiment, topic_id)

def insert_bertopic_map_to_sql(cursor, topic_model):
    topics_info = topic_model.get_topic_info()
    topic_reps = topic_model.get_topics()
    coords_2d = topic_model.topic_embeddings_[:, :2]

    for i, row in topics_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        label = row["Name"]
        freq = int(row["Count"])
        keywords = ", ".join([w for w, _ in topic_reps.get(topic_id, [])[:5]])
        try:
            x, y = map(float, coords_2d[topic_id])
        except IndexError:
            continue
        cursor.execute("""
            MERGE INTO Topic AS target
            USING (SELECT ? AS TopicID) AS source
            ON target.TopicID = source.TopicID
            WHEN MATCHED THEN
                UPDATE SET x = ?, y = ?, Freq = ?, Label = ?, Keywords = ?
            WHEN NOT MATCHED THEN
                INSERT (TopicID, x, y, Freq, Label, Keywords)
                VALUES (?, ?, ?, ?, ?, ?);
        """, topic_id, x, y, freq, label, keywords, topic_id, x, y, freq, label, keywords)

def insert_keyword_topic_weight(cursor, topic_model, account_id, frequency_map=None, top_n=10):
    topic_keywords = topic_model.get_topics()
    for topic_id, words in topic_keywords.items():
        if topic_id == -1:
            continue
        for word, weight in words[:top_n]:
            frequency = frequency_map.get(word, None) if frequency_map else None
            cursor.execute("""
                MERGE INTO KeywordTopicWeight AS target
                USING (SELECT ? AS AccountID, ? AS TopicID, ? AS Keyword) AS source
                ON target.AccountID = source.AccountID AND target.TopicID = source.TopicID AND target.Keyword = source.Keyword
                WHEN MATCHED THEN
                    UPDATE SET Weight = ?, Frequency = ?
                WHEN NOT MATCHED THEN
                    INSERT (AccountID, TopicID, Keyword, Weight, Frequency)
                    VALUES (?, ?, ?, ?, ?);
            """, account_id, topic_id, word, weight, frequency,
                 account_id, topic_id, word, weight, frequency)

def transform_sqldata(cursor):
    cursor.execute("""
    UPDATE Post
        SET PostHour = DATEPART(HOUR, PostTime) 
    FROM Post;

    UPDATE Comment
        SET CommentHour = DATEPART(HOUR, Timestamp)
    FROM Comment;

    UPDATE Reply
        SET ReplyHour = DATEPART(HOUR, Timestamp)
    FROM Reply;

    UPDATE R
    SET R.TimeDiff = DATEDIFF(MINUTE, C.Timestamp, R.Timestamp)
    FROM Reply AS R
    INNER JOIN Comment AS C ON R.CommentID = C.CommentID;

    UPDATE C
    SET C.TimeDiff = DATEDIFF(MINUTE, P.Date, C.Timestamp)
    FROM Comment AS C
    INNER JOIN Post AS P ON C.PostID = P.PostID;

    UPDATE Comment
    SET 
        TimeBucket_Label = 
            CASE 
                WHEN TimeDiff IS NULL THEN 'Unbekannt'
                WHEN TimeDiff < 10 THEN '0–10 min'
                WHEN TimeDiff < 30 THEN '10–30 min'
                WHEN TimeDiff < 60 THEN '30–60 min'
                WHEN TimeDiff < 180 THEN '1–3 Std'
                ELSE '3+ Std'
            END,
        TimeBucket_Order =
            CASE 
                WHEN TimeDiff IS NULL THEN 0
                WHEN TimeDiff < 10 THEN 1
                WHEN TimeDiff < 30 THEN 2
                WHEN TimeDiff < 60 THEN 3
                WHEN TimeDiff < 180 THEN 4
                ELSE 5
            END;
    """)


# === MAIN ACCOUNT PROCESSING ===
# Combined flow of scraping, extracting, inserting, and model training
# Kombinierter Ablauf von Scraping, Datenextraktion, Einfügen und Modelltraining

def prepare_account_data(account_urls, cursor):
    all_comment_entries = []

    profile_data_list = get_scraper_data_sync("details", account_urls, results_limit=10)

    def extract_shortcode(url):
        match = re.search(r"/(p|reel)/([^/]+)/", url)
        return match.group(2) if match else None

    for profile in profile_data_list:
        username = profile.get("username", "unknown_user")
        account_id = insert_account(cursor, profile)
        insert_account_snapshot(cursor, profile, account_id)

        posts = get_scraper_data_sync("posts", [f"https://www.instagram.com/{username}/"], results_limit=100)
        existing_post_ids = get_existing_post_ids(cursor, account_id)
        new_posts = [p for p in posts if int(p["id"]) not in existing_post_ids]

        if not new_posts:
            continue

        post_urls = [p["url"] for p in new_posts]
        comments_raw = get_scraper_data_sync("comments", post_urls, results_limit=100)

        shortcode_to_postid = {
            extract_shortcode(p["url"]): int(p["id"])
            for p in new_posts
        }

        for post in new_posts:
            insert_post(cursor, post, account_id, profile.get("followersCount"))

        for comment in comments_raw:
            comment_shortcode = extract_shortcode(comment.get("postUrl", ""))
            post_id = shortcode_to_postid.get(comment_shortcode)

            if post_id is not None and comment.get("text"):
                all_comment_entries.append({
                    "text": comment.get("text"),
                    "post_id": post_id,
                    "comment": comment,
                    "account_id": account_id
                })

    return all_comment_entries

def train_topic_model_on_comments(comment_entries):
    texts = [entry["text"] for entry in comment_entries if entry.get("text")]

    if not texts:
        print("[INFO] Keine Texte zum Verarbeiten")
        return None, [], {}

    texts_cleaned = preprocess_texts(texts)
    topic_model, topics, _ = train_bertopic_model(texts_cleaned)
    word_counts = compute_frequencies(texts_cleaned)

    return topic_model, topics, word_counts


def insert_comments_and_topics(cursor, comment_entries, topic_model, topics, word_counts):
    insert_bertopic_map_to_sql(cursor, topic_model)

    for (entry, topic_id) in zip(comment_entries, topics):
        comment = entry["comment"]
        post_id = entry["post_id"]
        account_id = entry["account_id"]

        if not comment_exists(cursor, comment["id"]):
            sentiment = analyze_sentiment(comment.get("text"))
            insert_comment_and_replies(cursor, comment, post_id, sentiment, topic_id)

    insert_keyword_topic_weight(cursor, topic_model, account_id, word_counts)
    transform_sqldata(cursor)



# === MAIN ENTRY POINT ===
# Connect to DB, process account list, store results
# Verbindung zur DB, Account-Liste verarbeiten, Ergebnisse speichern

def main():
    conn = pyodbc.connect(SQL_CONN_STR)
    cursor = conn.cursor()

    account_urls = [
        "https://www.instagram.com/indiranagar_runclub/",
        "https://www.instagram.com/pudhechala.runclub/",
        "https://www.instagram.com/ontourclub/",
        "https://www.instagram.com/bhagclub/",
        "https://www.instagram.com/shutupandmove._/"
    ]

    try:
        comment_entries = prepare_account_data(account_urls, cursor)
        topic_model, topics, word_counts = train_topic_model_on_comments(comment_entries)
        if topic_model:
            insert_comments_and_topics(cursor, comment_entries, topic_model, topics, word_counts)
    except Exception as e:
        print(f"[ERROR] Verarbeitung fehlgeschlagen: {e}")

    conn.commit()
    conn.close()
    print("[INFO] Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    main()
