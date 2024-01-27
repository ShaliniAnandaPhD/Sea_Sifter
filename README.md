# Seasifter

Seasifter is a tool for visualizing and analyzing global microplastics pollution data to assist with clean-up efforts. 

## Features

* Interactive map showing microplastics accumulation density data across oceans and coastlines
* Ability to click on a location to view plastic count, breakdown of types, and time series data
* Searchable knowledge graph linking key entities like plastic types, remediation methods, research papers etc.
* Query interface to find relevant research papers on plastic removal approaches suitable for a location
* Future prediction models showing projected accumulation levels based on past trends

## Data Sources

* **Open microplastics contamination data** - stored in MongoDB, used to power core map visualizations 
* **Plastic remediation research papers** - scraped and indexed in Neo4j graph database, accessible via search
* **Supplemental datasets** - weather, ocean currents etc. used for future predictions  

## Architecture

* **Frontend** - React dashboard with Leaflet map, Streamlit search and query interfaces  
* **Backend** - Flask API pulls data from MongoDB and Neo4j to serve frontend visualizations and search
* **Data Pipeline** - Scripts for scraping papers, transferring and indexing data, generating knowledge graph
* **Model Training** - Notebooks for developing ML models to predict future plastic accumulation  

## Getting Started  

* Prereq: Python 3.7+, Node.js
* `pip install -r requirements.txt`  
* `npm install`
* `npm run start` - start React front-end
* Configure .env with credentials
* `python app.py` - launch backend Flask server
* Load data: `python data_pipeline.py` 

Feel free to use the issue tracker and contribute! Together we can tackle the microplastics crisis.
