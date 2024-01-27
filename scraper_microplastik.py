import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient

# MongoDB Connection
# Replace with your actual MongoDB connection string, database, and collection names
mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'

client = MongoClient(mongo_connection_string)
db = client['microplastik']
collection = db['papers']

def scrape_article(url):
    """
    Scrape the content of the article from the given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example of scraping: Extracting the title of the article
        # Modify the selection as per the structure of your target webpage
        title = soup.find('h1').text

        # Return the extracted data as a dictionary
        # You can add more fields based on what you need to extract
        return {'url': url, 'title': title}

    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None

def main():
    """
    Main function to iterate over a range of URLs and scrape articles.
    """
    base_url = 'https://microplastics.springeropen.com/'  # Replace with the base URL of the articles
    for i in range(10000):  # Loop over 1000 articles (adjust as needed)
        article_url = f"{base_url}/{i}"  # Construct the URL; modify based on URL structure
        try:
            article_data = scrape_article(article_url)
            if article_data:
                collection.insert_one(article_data)  # Insert the article data into MongoDB
        except Exception as e:
            print(f"An error occurred while scraping {article_url}: {e}")

if __name__ == '__main__':
    main()

