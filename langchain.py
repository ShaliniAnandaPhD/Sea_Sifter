import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import SinglePromptChain
from pymongo import MongoClient
import networkx as nx
import matplotlib.pyplot as plt

# Initialize language model (OpenAI GPT-3)
llm = OpenAI(api_key="your-openai-api-key")

# Setup for the summarization chain
summarization_chain = SinglePromptChain(
    llm=llm, 
    prompt_function=lambda text: f"Summarize this text in 3 sentences: \n{text}",
    post_process_function=lambda resp: resp.strip()
)

# MongoDB connection for the search agent
client = MongoClient("mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/")
db = client["microplastik"]
papers = db["papers"]

# Search function
def search(query):
    return list(papers.find({"$text": {"$search": query}}))

# Graph construction
def build_graph():
    G = nx.Graph()
    G.add_node("Microplastics")
    G.add_node("Environmental Impact")
    G.add_edge("Microplastics", "Environmental Impact")
    return G

# Streamlit app for UI
st.title("MicroPlastiK Research Explorer")

query = st.text_input("Enter your search query")
if query:
    results = search(query)
    for paper in results:
        summary = summarization_chain(paper["content"])
        st.subheader(paper["title"])
        st.write(summary)

# Graph visualization
G = build_graph()
plt.figure(figsize=(10, 7))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray')
st.pyplot(plt)
