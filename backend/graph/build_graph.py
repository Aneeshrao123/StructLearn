import json
import networkx as nx

GRAPH_PATH = "backend/graph/graph_data.json"

def load_graph():
    with open(GRAPH_PATH, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node, edges in data.items():
        for child in edges:
            G.add_edge(node, child)

    return G
