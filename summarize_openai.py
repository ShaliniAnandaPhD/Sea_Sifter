import requests
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Logging setup
logging.basicConfig(level=logging.INFO)

# OpenAI API
OPENAI_URL = "https://api.openai.com/v1/engines/davinci-codex/completions"
OPENAI_KEY = "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"  # Replace with your actual OpenAI API key

# MongoDB Connection
mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'
client = MongoClient(mongo_connection_string)
db = client['microplastik']
collection = db['papers']

def summarize(text):
    """Call OpenAI API to summarize text"""
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": "Summarize this text: " + text,
        "temperature": 0.7,
        "max_tokens": 150  # Adjust as needed
    }
    try:
        response = requests.post("https://api.openai.com/v1/engines/davinci/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except requests.RequestException as e:
        logging.error(f"Error calling OpenAI API: {e}")
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
