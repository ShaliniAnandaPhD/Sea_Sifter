from flask import Flask, request, render_template
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB Connection
mongo_connection_string = 'your_connection_string'
client = MongoClient(mongo_connection_string)
db = client['microplastik']
collection = db['papers']

@app.route('/', methods=['GET', 'POST'])
def search():
    search_results = []
    if request.method == 'POST':
        query = request.form['query']
        search_results = collection.find({"$text": {"$search": query}})
    return render_template('search.html', search_results=search_results)

if __name__ == '__main__':
    app.run(debug=True)
