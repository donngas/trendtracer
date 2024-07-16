import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def get_cooc_network(file):

    #Assuming the CSV file has no header and each row contains keywords separated by commas
    df = pd.read_csv(file, header=None)

    print("File successfully loaded.")

    #Extract keywords from the first column
    keywords_per_article = df.iloc[:, 0].str.split(',')

    #Initialize an empty co-occurrence matrix
    co_occurrence_matrix = {}

    #Build co-occurrence matrix based on pairs of keywords within each article
    for keywords in keywords_per_article:
        for pair in combinations(keywords, 2):
            if pair not in co_occurrence_matrix:
                co_occurrence_matrix[pair] = 0
            co_occurrence_matrix[pair] += 1

    print("Co-oc matrix built.")

    #Create a NetworkX graph
    G = nx.Graph()

    #Add edges to the graph
    for pair, weight in co_occurrence_matrix.items():
        keyword1, keyword2 = pair
        G.add_edge(keyword1, keyword2, weight=weight)

    #Compute degree centrality for each node (keyword)
    degree_centralities = nx.degree_centrality(G)

    #Scale node sizes based on degree centrality (multiplied by a factor for better visibility)
    node_sizes = [3000 * degree_centralities[node] for node in G.nodes()]

    print("About to draw graph.")

    #Draw the network graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)  # positions for all nodes
    nx.draw(G, pos, node_color='skyblue', node_size=node_sizes, with_labels=True, font_size=8, font_weight='bold', edge_color='gray')
    plt.title('Keyword Co-occurrence Network with Emphasized Importance')
    plt.show()

if __name__ == "__main__":
    get_cooc_network("examplekeywords.csv")