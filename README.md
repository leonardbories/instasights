# 📊 Instagram ETL & Data Analytics Project

This portfolio project demonstrates a complete ETL (Extract, Transform, Load) pipeline to collect, process, and analyze Instagram data using the Apify API.

## 🚀 Features

- Data extraction from Instagram via Apify (profiles, posts, comments)
- Natural Language Processing with VADER sentiment analysis
- Storage of structured data in a SQL Server database
- Topic modeling and keyword weight analysis for comments
- Backup of raw JSON data
- Dashboard (Power BI)

## 🧰 Tech Stack

- Python (Pandas, NLTK, SentenceTransformers)
- SQL Server (via pyodbc)
- Apify API for Instagram scraping
- Power BI or Tableau for visualization

## 📂 Project Structure

InstagramAnalytics/
├── backups/ # API answers
├── icon/ # used icons in my project
├── scripts/ # ETL logic
├── Instagram Analytics_.pbix # Power BI dashboard
└── README.md # project info


## 🛡️ API Token

**DO NOT COMMIT YOUR API TOKEN**  
In the code: replace `API_TOKEN = "INSERT_YOUR_API_TOKEN_HERE"` with your actual token **locally only**.
