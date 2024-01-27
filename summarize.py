import requests
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Logging setup
logging.basicConfig(level=logging.INFO)

# Claude API
CLAUDE_URL = "https://api.claude.com/summarize"
CLAUDE_KEY = "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"

# MongoDmongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'

# Replace with your actual MongoDB connection string, database, and collection names
mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'

client = MongoClient(mongo_connection_string)
db = client['microplastik']
collection = db['papers']



def summarize(text):
    """Call Claude API to summarize text"""
    headers = {"Authorization": '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'}
    data = {"text": text}
    try:
        response = requests.post(CLAUDE_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["summary"]
    except requests.RequestException as e:
        logging.error(f"Error calling Claude API: {e}")
        return None

def store_summary(original_text, summary):
    """Store original text and summary in MongoDB"""
    if summary:
        try:
            doc = {"original_text": original_text, "summary": summary}
            collection.insert_one(doc)
        except Exception as e:
            logging.error(f"Error storing summary in MongoDB: {e}")

def process_texts(texts):
    """Summarize a list of texts and store the summaries"""
    for text in texts:
        summary = summarize(text)
        store_summary(text, summary)

if __name__ == "__main__":
    texts = ["Text 1 to process", "Text 2 to process", "Text 3 to process"]
    process_texts(texts)
