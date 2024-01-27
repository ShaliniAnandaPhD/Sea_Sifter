import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load Spacy NLP Model
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    """Extract entities using Spacy NER."""
    doc = nlp(text)
    return [(X.text, X.label_) for X in doc.ents]

def build_graph(entity_pairs):
    """Build a graph from the entity pairs."""
    G = nx.Graph()
    for text, kind in entity_pairs:
        G.add_node(text, type=kind)
        for other_text, _ in entity_pairs:
            # This is a simple way to create edges: linking each entity to every other
            # You might want to use more sophisticated methods depending on your needs
            if other_text != text:
                G.add_edge(text, other_text)
    return G

def visualize_graph(G):
    """Visualize the entity graph."""
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['type']
                                                      for u, v in G.edges()}, font_color='red')
    plt.axis('off')
    plt.show()

# Example text summaries
summaries = [
    "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
    "Google LLC is an American multinational technology company that specializes in Internet-related services and products."
]

# Process each summary
all_entities = []
for summary in summaries:
    entities = extract_entities(summary)
    all_entities.extend(entities)

# Build and visualize the graph
G = build_graph(all_entities)
visualize_graph(G)
