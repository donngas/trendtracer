import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot as offline_plot
import os

# Set colors
node_colors_set = ['blue', 'red', 'orange', 'yellow', 'grey']
edge_colors_set = ['mediumseagreen', 'darkkhaki', 'lightsteelblue']
convert_colors_set = ['seagreen', 'darkgoldenrod', 'black']

# Function to assign colors to nodes based on their ranks
def assign_colors_by_rank(sorted_nodes):
    global node_colors_set
    color_map = {}
    max_rank = len(sorted_nodes)
    for rank, node in enumerate(sorted_nodes, start=1):
        if rank <= max_rank * 0.1:
            color_map[node] = node_colors_set[0]
        elif rank <= max_rank * 0.2:
            color_map[node] = node_colors_set[1]
        elif rank <= max_rank * 0.5:
            color_map[node] = node_colors_set[2]
        elif rank <= max_rank * 0.8:
            color_map[node] = node_colors_set[3]
        else:
            color_map[node] = node_colors_set[4]
    return color_map

# Function to assign colors to edges based on their weights
def assign_edge_colors_by_weight(G):
    global edge_colors_set
    edge_color_map = {}
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    weight_percentiles = pd.Series(weights).rank(pct=True)
    
    for (u, v), percentile in zip(G.edges(), weight_percentiles):
        if percentile >= 0.95:
            edge_color_map[(u, v)] = edge_colors_set[0]
        elif percentile >= 0.8:
            edge_color_map[(u, v)] = edge_colors_set[1]
        else:
            edge_color_map[(u, v)] = edge_colors_set[2]
    
    return edge_color_map

# Function to retrieve edge color with both words as string
def get_edge_color(u, v, edge_color_map):
    if (u, v) in edge_color_map:
        return edge_color_map[(u, v)]
    elif (v, u) in edge_color_map:
        return edge_color_map[(v, u)]
    else:
        return 'black'

# Function to convert edge color into more visible color for hoverinfo
def convert_edge_color(color, edge_colors_set, convert_colors_set):
    if color == edge_colors_set[0]:
        return convert_colors_set[0]
    elif color == edge_colors_set[1]:
        return convert_colors_set[1]
    else:
        return convert_colors_set[2]

def get_cooc_network(file, saving_directory, max_keywords=300):
    global edge_colors_set

    # Try to load CSV file
    try:
        df = pd.read_csv(file, header=0)
        column_name = df.columns[1]
        df = df.drop(columns=df.columns[0])
        df = df.reset_index(drop=True)
        print("[Cooc] File successfully loaded.")
    except Exception as error_while_loading:
        print("[Cooc] Error while loading keywords as csv.")
        print("[Cooc] Error:", error_while_loading)
        return

    # Try to create co-oc matrix
    try:
        keywords_per_article = df.iloc[:, 0].str.split(', ').tolist()
        co_occurrence_matrix = {}
        error_while_creating_iter = 0

        for keywords in keywords_per_article:
            if not isinstance(keywords, list):
                continue

            keywords_filtered = [word for word in keywords if len(word) > 2]
            count_dict = {}
            keywords_filtered = [word for word in keywords_filtered if count_dict.setdefault(word, 0) < 2 and not count_dict.update({word: count_dict[word] + 1})]

            try:
                for pair in combinations(keywords_filtered, 2):
                    if pair not in co_occurrence_matrix:
                        co_occurrence_matrix[pair] = 0
                    co_occurrence_matrix[pair] += 1
            except:
                error_while_creating_iter += 1
                continue
        
        print("[Cooc] Co-occurrence matrix built.")
        print("[Cooc]", error_while_creating_iter, "exceptions occurred during creation. Accountable rows were skipped, if any.")

    except Exception as error_while_creating:
        print("[Cooc] Error while creating co-oc matrix, couldn't complete.")
        print("[Cooc] Error:", error_while_creating)
        return

    # Try to draw the graph
    try:
        G = nx.Graph()
        for pair, weight in co_occurrence_matrix.items():
            keyword1, keyword2 = pair
            G.add_edge(keyword1, keyword2, weight=weight)

        # Compute degree centrality for each node (keyword)
        degree_centralities = nx.degree_centrality(G)

        # Sort nodes by degree centrality (highest to lowest)
        sorted_nodes = sorted(G.nodes(), key=lambda x: degree_centralities[x], reverse=True)

        # Keep only the top N keywords (e.g., 300 most important ones)
        top_nodes = sorted_nodes[:max_keywords]

        # Remove edges involving nodes that are not in the top N
        edges_to_remove = [(u, v) for u, v in G.edges() if u not in top_nodes or v not in top_nodes]
        G.remove_edges_from(edges_to_remove)

        # Assign colors based on rank
        color_map = assign_colors_by_rank(top_nodes)
        edge_color_map = assign_edge_colors_by_weight(G)

        # Compute positions for each node using spring layout
        pos = nx.spring_layout(G, k=1.5, iterations=200, weight='weight')

        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x = [x0, x1, None]
            edge_y = [y0, y1, None]
            edge_color = edge_color_map[(edge[0], edge[1])]
            line_width = 1.3 if edge_color == edge_colors_set[0] else (0.8 if edge_color == edge_colors_set[1] else 0.5)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=line_width, color=edge_color),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        annotations = []
        for node in top_nodes:
            x, y = pos[node]
            border_color = color_map[node]
            connected_keywords = list(G.neighbors(node))
            connected_keywords.sort(key=lambda n: G[node][n]['weight'], reverse=True)
            connected_keywords_text = ''

            for i in range(0, len(connected_keywords), 5):
                for j in range(i, min(i + 5, len(connected_keywords))):
                    current_connected_keyword = str(connected_keywords[j])
                    current_edge_color = get_edge_color(str(node), current_connected_keyword, edge_color_map)
                    current_edge_color = convert_edge_color(current_edge_color, edge_colors_set, convert_colors_set)

                    # Apply edge color to annotation text
                    if current_edge_color == convert_colors_set[0]:
                        current_connected_keyword = f'<span style="color: {convert_colors_set[0]};">'+current_connected_keyword+'</span>'
                    elif current_edge_color == convert_colors_set[1]:
                        current_connected_keyword = f'<span style="color: {convert_colors_set[1]};">'+current_connected_keyword+'</span>'
                    else:
                        current_connected_keyword = f'<span style="color: {convert_colors_set[2]};">'+current_connected_keyword+'</span>'

                    if i == 0 and j == i:
                        connected_keywords_text += current_connected_keyword
                    elif j == i:
                        connected_keywords_text += ',<br>'+current_connected_keyword
                    else:
                        connected_keywords_text += ', '+current_connected_keyword

            annotations.append(
                dict(
                    x=x, y=y,
                    xref='x1', yref='y1',
                    text=node,
                    hovertext=f"<b>{node}</b><br>Connected keywords:<br>{connected_keywords_text}",
                    showarrow=False,
                    font=dict(family='Arial', size=12, color='black'),
                    align='center',
                    bgcolor='lightgray',
                    bordercolor=border_color,
                    borderwidth=1,
                    borderpad=4
                )
            )

        fig = go.Figure(data=edge_traces,
                        layout=go.Layout(
                            title='Keyword Co-occurrence Network for <b>'+column_name.capitalize()+'</b> Category',
                            title_font=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=annotations,
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)))

        filename = os.path.join(saving_directory, f'{column_name}.html')
        offline_plot(fig, filename=filename, auto_open=False)

        print("[Cooc] Successfully saved co-occurrence network for", file)

    except Exception as error_while_drawing:
        print("[Cooc] Error while drawing the graph")
        print("[Cooc] Error:", error_while_drawing)
        return

if __name__ == "__main__":
    get_cooc_network("examplekeywords.csv", "./resources/graphs/")
