from pymongo import MongoClient
from py2neo import Graph, Node, Relationship, NodeMatcher
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# MongoDB Connection String
mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'

# Connect to MongoDB
client = MongoClient(mongo_connection_string)
db = client.your_database_name  # Replace with your MongoDB database name
collection = db.your_collection_name  # Replace with your MongoDB collection name

# Connect to Neo4j
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "your_password"))  # Update with your Neo4j credentials
matcher = NodeMatcher(graph)

# Process and Transfer Data from MongoDB to Neo4j
for doc in collection.find():
    try:
        # Create or find existing location node
        location_node = matcher.match("Location", name=doc['location_name']).first()
        if not location_node:
            location_node = Node("Location", name=doc['location_name'], latitude=doc['lat'], longitude=doc['long'])
            graph.create(location_node)
        
        # Create or find existing density node
        density_node = matcher.match("MicroplasticDensity", density=doc['density']).first()
        if not density_node:
            density_node = Node("MicroplasticDensity", density=doc['density'])
            graph.create(density_node)

        # Create relationship if it doesn't exist
        if not graph.match_one((location_node, density_node), "HAS_DENSITY"):
            relation = Relationship(location_node, "HAS_DENSITY", density_node)
            graph.create(relation)

        logging.info(f"Processed document: {doc['_id']}")

    except Exception as e:
        logging.error(f"Error processing document {doc['_id']}: {e}")

print("Data transfer completed.")
